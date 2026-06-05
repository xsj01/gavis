#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from ..utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from ..utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from ..utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from ..utils.graphics_utils import BasicPointCloud
from ..utils.general_utils import strip_symmetric, build_scaling_rotation

from copy import deepcopy
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import gc
import math

from .fixed_gaussian_helper import sample_shell_points, sample_volume_points

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(
        self,
        sh_degree: int,
        object_boundaries=[[-3, 3], [-3, 3], [-3, 3]],
    ):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        #####
        

        
        # assert object_boundaries != None
        self.object_boundaries = object_boundaries
        self._opacity_temp = None
        #####
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.N = 0

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            #####
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            # #####
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    def get_xyz_no_background(self):
        return self._xyz[self.N:]
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    #####
    # def get_visibility(self, view=None):
    #     if self._angular_vis and view != None:
    #         return self._angular_vis.query_angular_visibility(view, self._xyz)
    #     else:
    #         return self._visibility

    #####

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    #####
    # def set_visibility(self, new_vis):
    #     self._visibility = new_vis
    #     torch.cuda.empty_cache()

    # def update_visibility(self, new_vis):
    #     self._visibility *= new_vis
    #     torch.cuda.empty_cache()

    # def reset_angular_vis(self):
    #     self._angular_vis.reset_angular_visibility()

    # def query_angular_vis(self, view):
    #     assert self._angular_vis != None
    #     return self._angular_vis.query_angular_visibility(view, self._xyz)

    # def update_angular_vis(self, view, field_outputs=None, transmittance=None, kappa=1.5):
    #     self._angular_vis.build_angular_visibility(view, self._xyz, field_outputs, transmittance, kappa)

    def opaque(self):
        N = 50 * 50 * 50  # self.fixed_points_num #self._xyz.shape[0]
        self._opacity[:self.N] = inverse_sigmoid(
            torch.full((self._opacity[:self.N].shape[0], 1), 0.5, device="cuda")
        )

    def scale(self, scale=5.0):
        N = 50 * 50 * 50
        self._scaling[:self.N] = float(np.log(np.sqrt(0.0018)))

    def zero_fixed_opacity(self, N=50 * 50 * 50):
        with torch.no_grad():
            self._opacity[:N] = inverse_sigmoid(
                torch.full((self._opacity[:N].shape[0], 1), 0.0, device="cuda")
            )
        torch.cuda.empty_cache()

    def zero_gaussian_grads(self, N=50 * 50 * 50):
        self.fixed_points_num = 50 * 50 * 50
        N = self.fixed_points_num  # self._xyz.shape[0]
        self._xyz.grad[:N] = 0
        self._features_dc.grad[:N] = 0
        self._features_rest.grad[:N] = 0
        self._scaling.grad[:N] = 0
        self._rotation.grad[:N] = 0
        self._opacity.grad[:N] = 0

    def set_opacity(self, opacities):
        N = 50 * 50 * 50
        if self._opacity_temp == None:
            self._opacity_temp = opacities[:N]
        else:
            temp = opacities[:N]
            mask = temp > 0
            self._opacity_temp[mask] = temp[mask]
        torch.cuda.empty_cache()

    def format_opacity(self):
        assert self._opacity_temp != None
        N = 50 * 50 * 50
        with torch.no_grad():
            self._opacity[:N] = inverse_sigmoid(self._opacity_temp)
        self._opacity_temp = None
        torch.cuda.empty_cache()

    #####

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def add_pcd_gaussians(self, points, colors):
        fused_color = RGB2SH(torch.tensor(colors).float().cuda())
        fused_point_cloud = torch.tensor(points).float().cuda()
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        self._xyz = nn.Parameter(torch.cat((self._xyz, fused_point_cloud), dim=0))
        self._features_dc = nn.Parameter(torch.cat(
            (self._features_dc, features[:, :, 0:1].transpose(1, 2).contiguous()), dim=0
        ))
        self._features_rest = nn.Parameter(torch.cat(
            (self._features_rest, features[:, :, 1:].transpose(1, 2).contiguous()),
            dim=0,
        ))

        scale = 0.0008 # 0.00009 # 0.040 #0.0018, #0.0008 = for initial tests

        scale_x = torch.full((fused_point_cloud.shape[0], 1), scale, device="cuda")
        scale_y = torch.full((fused_point_cloud.shape[0], 1), scale, device="cuda")
        scale_z = torch.full((fused_point_cloud.shape[0], 1), scale, device="cuda")

        _background_scaling = torch.cat((scale_x, scale_y, scale_z), dim=1)
        _background_scaling = torch.log((torch.sqrt(_background_scaling)))

        self._scaling = nn.Parameter(torch.cat((self._scaling, _background_scaling), dim=0))
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        self._rotation = nn.Parameter(torch.cat((self._rotation, rots), dim=0))
        opacities = inverse_sigmoid(
            torch.full((fused_point_cloud.shape[0], 1), 0.1, device="cuda")
        )
        self._opacity = nn.Parameter(torch.cat((self._opacity, opacities), dim=0))

        # visbilities = torch.ones(
        #     (fused_point_cloud.shape[0], 1), dtype=torch.float
        # ).cuda()
        # self._visibility = torch.cat((self._visibility, visbilities), dim=0)

        self.max_radii2D = torch.cat(
            (
                self.max_radii2D,
                torch.zeros((fused_point_cloud.shape[0]), device="cuda"),
            ),
            dim=0,
        )
        
    def add_background(self, boundaries=np.asarray([[-3, 3], [-3, 3], [-3, 3]]), use_surface=False, scale = 0.0005, density=1250):
        self.background_on = True

        x = boundaries[0]
        y = boundaries[1]
        z = boundaries[2]

        #determine count based on the size of the bounding box:
        boundaries = np.asarray(boundaries, dtype=float)
        lengths = boundaries[:, 1] - boundaries[:, 0]  
        lx, ly, lz = lengths       
       
        

        if use_surface:
            #pts/ unit cubed
            N = density * 2*(lx * ly + ly * lz + lx * lz)
        else:
            #pts/unit sq
            N = density*lx*ly*lz

        self.N = int(np.ceil(N))   # or round(), np.floor(), etc.—but make it int

        boundary_volume = lx * ly * lz
        gaussian_volume_to_boundary_volume_ratio = 1.5
        scale = (
            gaussian_volume_to_boundary_volume_ratio * boundary_volume * 0.75 * (1/math.pi) * (1.0/self.N)
                 )**(1/3)
        # breakpoint()
        if use_surface:
            grid_points = sample_shell_points(boundaries, 0.10, self.N)  
        else:
            grid_points = sample_volume_points(boundaries, self.N)
    
        print("grid_points to add: ", grid_points.shape)
        print("self.N: ", self.N)
        print("self.gaussians size before adding background: ", self._xyz.shape)

        color_grid_points = np.ones_like(grid_points, dtype=float)
        fused_point_cloud = torch.tensor(grid_points).float().cuda()
        fused_color = RGB2SH(torch.tensor(color_grid_points).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        self._xyz = torch.cat((fused_point_cloud, self._xyz), dim=0)
        self._features_dc = torch.cat(
            (features[:, :, 0:1].transpose(1, 2).contiguous(), self._features_dc), dim=0
        )
        self._features_rest = torch.cat(
            (features[:, :, 1:].transpose(1, 2).contiguous(), self._features_rest),
            dim=0,
        )

        #scale = 0.0005 # 0.00009 # 0.040 #0.0018, #0.0008 = for initial tests

        scale_x = torch.full((grid_points.shape[0], 1), scale, device="cuda")
        scale_y = torch.full((grid_points.shape[0], 1), scale, device="cuda")
        scale_z = torch.full((grid_points.shape[0], 1), scale, device="cuda")

        self._background_scaling = torch.cat((scale_x, scale_y, scale_z), dim=1)
        self._background_scaling = torch.log(self._background_scaling)

        self._scaling = torch.cat((self._background_scaling, self._scaling), dim=0)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        self._rotation = torch.cat((rots, self._rotation), dim=0)
        opacities = inverse_sigmoid(
            torch.full((fused_point_cloud.shape[0], 1), 0.0, device="cuda")
        )
        self._opacity = torch.cat((opacities, self._opacity), dim=0)

        # visbilities = torch.ones(
        #     (fused_point_cloud.shape[0], 1), dtype=torch.float
        # ).cuda()
        # self._visibility = torch.cat((visbilities, self._visibility), dim=0)

        # torch.cuda.empty_cache()
        # self._angular_vis.spherical_visibility = None

        self.max_radii2D = torch.cat(
            (
                torch.zeros((fused_point_cloud.shape[0]), device="cuda"),
                self.max_radii2D,
            ),
            dim=0,
        )
        
        print("self.gaussians size after adding background: ", self._xyz.shape)

    # def remove_background(self, N=55 * 55 * 55):
    #     self.background_on = False
    #     self.using_background = False

    #     print("self.N: ", self.N)
    #     print("self.gaussians size before pruning: ", self._xyz.shape)
    #     self._xyz = nn.Parameter(self._xyz[self.N:])
    #     self._features_dc = nn.Parameter(self._features_dc[self.N:])
    #     self._features_rest = nn.Parameter(self._features_rest[self.N:])
    #     self._opacity = nn.Parameter(self._opacity[self.N:])
    #     self._scaling = nn.Parameter(self._scaling[self.N:])
    #     self._rotation = nn.Parameter(self._rotation[self.N:])

    #     # self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
    #     # self.denom = self.denom[valid_points_mask]
    #     self.max_radii2D = self.max_radii2D[self.N:]
    #     print("self.gaussians size after pruning: ", self._xyz.shape)

    #     #####
    #     torch.cuda.empty_cache()
    #     self._visibility = self._visibility[self.N:]
    #     # self._visibility = optimizable_tensors["visibility"]
    #     torch.cuda.empty_cache()

    #     #####
    #     torch.cuda.empty_cache()
    #     if self._angular_vis.spherical_visibility != None:
    #         self._angular_vis.spherical_visibility = (
    #             self._angular_vis.spherical_visibility[self.N:]
    #         )
    #     torch.cuda.empty_cache()
        
    def prune_visible_background(self, visibility, threshold=0.95):
        # visibility is only size of fixed gaussians
        #remove fixed gaussians only
        mask = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=visibility.device)
        mask[:self.N] = (visibility.squeeze() <= threshold)
        
        visibility_mask = mask[:self.N]

        
        removed_count = torch.sum(~mask[:self.N]).item()
        print("Number of fixed gaussians removed:", removed_count)
        self.N = self.N - removed_count
        print("self.N: ", self.N)
        print("self.gaussians size before pruning: ", self._xyz.shape)

        self._xyz = nn.Parameter(self._xyz[mask])
        self._features_dc = nn.Parameter(self._features_dc[mask])
        self._features_rest = nn.Parameter(self._features_rest[mask])
        self._opacity = nn.Parameter(self._opacity[mask])
        self._scaling = nn.Parameter(self._scaling[mask])
        self._rotation = nn.Parameter(self._rotation[mask])
        self.max_radii2D = self.max_radii2D[mask]
        print("self.gaussians size after pruning: ", self._xyz.shape)
        #####
        torch.cuda.empty_cache()
        # self._visibility = self._visibility[mask]
        
        # self.pos_visibility = (visibility[mask])[:self.N].clone()
        
        torch.cuda.empty_cache()
        
        num_pruned = torch.sum(~mask[:self.N]).item()
        return num_pruned, visibility_mask

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        torch.cuda.empty_cache()
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = (
            (torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # dist2 = torch.clamp_min(
        #     distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
        #     0.0000001,
        # )
        torch.cuda.empty_cache()
        dist2 = torch.full((fused_point_cloud.shape[0],), 0.00018, device="cuda")
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        # scale = 0.0018
        # scales = torch.log(torch.sqrt(torch.ones_like(dist2)*scale))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacity = 0.1  # 0.1

        opacities = inverse_sigmoid(
            opacity
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # self.add_background()

        # if self.using_background:
        # #####
        #     print("adding background")
        #     # grid dimensions
        #     x = [-1.0, 1.0]
        #     y = [-1.1, 1.1]#[-20, 20]
        #     z = [-1.1, 1.8]

        #     x = self.object_boundaries[0]
        #     y = self.object_boundaries[1]
        #     z = self.object_boundaries[2]

        #     # scale resolution
        #     grid_size = 50
        #     self.fixed_points_num = grid_size* grid_size* grid_size

        #     x = np.linspace(x[0], x[1], grid_size)
        #     y = np.linspace(y[0], y[1], grid_size)
        #     z = np.linspace(z[0], z[1], grid_size)

        #     xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        #     grid_points = np.stack((xv, yv, zv), axis=-1)

        #     grid_points = grid_points.reshape(-1, 3)

        #     color_grid_points = np.ones_like(grid_points, dtype=float)

        #     self.spatial_lr_scale = spatial_lr_scale
        #     fused_point_cloud = torch.tensor(np.concatenate((grid_points, np.asarray(pcd.points)), axis=0)).float().cuda()
        #     fused_color = RGB2SH(torch.tensor(np.concatenate((color_grid_points, np.asarray(pcd.colors)), axis=0)).float().cuda())
        #     features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        #     features[:, :3, 0 ] = fused_color
        #     features[:, 3:, 1:] = 0.0

        #     print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        #     scale = 0.0018 #0.00009 # 0.0090
        #     dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        #     scale_x = torch.full((grid_points.shape[0], 1), scale, device='cuda')
        #     scale_y = torch.full((grid_points.shape[0], 1), scale, device='cuda')
        #     scale_z = torch.full((grid_points.shape[0], 1), scale, device='cuda')
        #     self._background_scaling = torch.cat((scale_x, scale_y, scale_z), dim=1)
        #     self._background_scaling = torch.log((torch.sqrt(self._background_scaling)))
        #     scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        #     scales = torch.cat((self._background_scaling, scales), dim=0)
        #     rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        #     rots[:, 0] = 1

        #     opacity = 0.0 # 0.1
        #     opacities = inverse_sigmoid(opacity * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        #     opacities[0:self.fixed_points_num] = inverse_sigmoid( 0.3* torch.ones((opacities[0:self.fixed_points_num].shape[0] , 1), device="cuda"))

        #     self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        #     self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        #     self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        #     self._scaling = nn.Parameter(scales.requires_grad_(True))
        #     self._rotation = nn.Parameter(rots.requires_grad_(True))
        #     self._opacity = nn.Parameter(opacities.requires_grad_(True))
        #     self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        #####

        #####
        # self.add_background()

        #####

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
            #####
            # {'params': [self._visibility], 'lr': training_args.opacity_lr, "name": "visibility"},
            #####
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        #####
        # visibilities = self._visibility.detach().cpu().numpy()
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation),
            axis=1,
        )
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        #####
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def save_fixed_ply(self, path):
        """Save only the first self.N (fixed / background) gaussians to PLY + .meta with N."""
        mkdir_p(os.path.dirname(path) or ".")
        n = int(self.N)
        with open(path + ".meta", "w", encoding="utf-8") as f:
            f.write(str(n))
        if n <= 0:
            return

        xyz = self._xyz[:n].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc[:n]
            .detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest[:n]
            .detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity[:n].detach().cpu().numpy()
        scale = self._scaling[:n].detach().cpu().numpy()
        rotation = self._rotation[:n].detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def _load_ply_prefix_tensors(self, path):
        """Load PLY into tensors (fixed gaussians only), same layout as load_ply."""
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        return {
            "xyz": torch.tensor(xyz, dtype=torch.float, device="cuda"),
            "f_dc": torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous(),
            "f_rest": torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous(),
            "opacity": torch.tensor(opacities, dtype=torch.float, device="cuda"),
            "scaling": torch.tensor(scales, dtype=torch.float, device="cuda"),
            "rotation": torch.tensor(rots, dtype=torch.float, device="cuda"),
        }

    def prepend_tensors_to_optimizer(self, prefix_dict):
        """Prepend fixed-gaussian tensors to each optimizer param group (fixed, learned order).

        Uses model attributes (self._xyz, etc.) as the source of truth for the learned tail.
        If optimizer param groups still reference a stale/smaller tensor than the model
        (e.g. after load without full optimizer sync), concatenating only group["params"][0]
        would shrink the model — see prepend_fixed_ply.
        """
        model_attr = {
            "xyz": "_xyz",
            "f_dc": "_features_dc",
            "f_rest": "_features_rest",
            "opacity": "_opacity",
            "scaling": "_scaling",
            "rotation": "_rotation",
        }
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            name = group["name"]
            prefix_tensor = prefix_dict[name]
            model_param = getattr(self, model_attr[name])
            opt_param = group["params"][0]
            if model_param.data.shape != opt_param.data.shape:
                raise RuntimeError(
                    f"prepend_tensors_to_optimizer: model vs optimizer shape mismatch for {name!r}: "
                    f"model {tuple(model_param.shape)} vs optimizer param {tuple(opt_param.shape)}. "
                    "Fix optimizer to point at the same tensors as the model before prepending."
                )
            # Always use model tensor data so we never prepend onto a stale copy.
            new_tensor = torch.cat((prefix_tensor, model_param.data), dim=0)
            stored_state = self.optimizer.state.get(opt_param, None)
            if stored_state is None:
                stored_state = self.optimizer.state.get(model_param, None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (torch.zeros_like(prefix_tensor), stored_state["exp_avg"]), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (torch.zeros_like(prefix_tensor), stored_state["exp_avg_sq"]),
                    dim=0,
                )
                if opt_param in self.optimizer.state:
                    del self.optimizer.state[opt_param]
                if model_param is not opt_param and model_param in self.optimizer.state:
                    del self.optimizer.state[model_param]
                group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))
            optimizable_tensors[name] = group["params"][0]
        return optimizable_tensors

    def prepend_fixed_ply(self, path):
        """Load fixed gaussians from PLY (+ optional .meta for N) and prepend to current (learned) model."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Fixed gaussian PLY not found: {path}")
        meta_path = path + ".meta"
        n_from_meta = None
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                n_from_meta = int(f.read().strip())
        prefix = self._load_ply_prefix_tensors(path)
        n_fixed = prefix["xyz"].shape[0]
        if n_from_meta is not None and n_from_meta != n_fixed:
            raise ValueError(
                f"fixed_gaussians.ply has {n_fixed} points but .meta says N={n_from_meta}"
            )
        if n_fixed == 0:
            self.N = 0
            return

        n_learned_before = self._xyz.shape[0]
        device = self._xyz.device
        # load_ply() does not set xyz_gradient_accum / denom; they stay empty CPU tensors.
        # Prepend zeros must match learned tensors' device for torch.cat.
        accum_dtype = torch.float32
        if self.xyz_gradient_accum.numel() > 0:
            accum_dtype = self.xyz_gradient_accum.dtype
        if (
            self.xyz_gradient_accum.numel() == 0
            or self.xyz_gradient_accum.shape[0] != n_learned_before
        ):
            self.xyz_gradient_accum = torch.zeros(
                (n_learned_before, 1), device=device, dtype=accum_dtype
            )
        else:
            self.xyz_gradient_accum = self.xyz_gradient_accum.to(
                device=device, dtype=accum_dtype
            )
        denom_dtype = torch.float32
        if self.denom.numel() > 0:
            denom_dtype = self.denom.dtype
        if self.denom.numel() == 0 or self.denom.shape[0] != n_learned_before:
            self.denom = torch.zeros(
                (n_learned_before, 1), device=device, dtype=denom_dtype
            )
        else:
            self.denom = self.denom.to(device=device, dtype=denom_dtype)
        radii_dtype = torch.float
        if self.max_radii2D.numel() > 0:
            radii_dtype = self.max_radii2D.dtype
        if self.max_radii2D.numel() == 0 or self.max_radii2D.shape[0] != n_learned_before:
            self.max_radii2D = torch.zeros(
                (n_learned_before,), device=device, dtype=radii_dtype
            )
        else:
            self.max_radii2D = self.max_radii2D.to(device=device, dtype=radii_dtype)

        d = {
            "xyz": prefix["xyz"],
            "f_dc": prefix["f_dc"],
            "f_rest": prefix["f_rest"],
            "opacity": prefix["opacity"],
            "scaling": prefix["scaling"],
            "rotation": prefix["rotation"],
        }

        self._xyz = nn.Parameter(
            torch.cat((d["xyz"], self._xyz), dim=0).requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.cat((d["f_dc"], self._features_dc), dim=0).requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.cat((d["f_rest"], self._features_rest), dim=0).requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.cat((d["opacity"], self._opacity), dim=0).requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.cat((d["scaling"], self._scaling), dim=0).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.cat((d["rotation"], self._rotation), dim=0).requires_grad_(True)
        )

        z = torch.zeros((n_fixed, 1), device=device, dtype=self.xyz_gradient_accum.dtype)
        self.xyz_gradient_accum = torch.cat((z, self.xyz_gradient_accum), dim=0)
        z_denom = torch.zeros((n_fixed, 1), device=device, dtype=self.denom.dtype)
        self.denom = torch.cat((z_denom, self.denom), dim=0)
        self.max_radii2D = torch.cat(
            (
                torch.zeros((n_fixed,), device=device, dtype=self.max_radii2D.dtype),
                self.max_radii2D,
            ),
            dim=0,
        )
        self.N = n_fixed

        n_after = self._xyz.shape[0]
        if n_after != n_learned_before + n_fixed:
            raise RuntimeError(
                f"prepend_fixed_ply: expected total points {n_learned_before + n_fixed} "
                f"(learned {n_learned_before} + fixed {n_fixed}), got {n_after}"
            )

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    #####
    # def reset_visibility(self, use_angular_vis=False):
    #     if use_angular_vis:
    #         self.reset_angular_vis()
    #     else:
    #         visibilities_new = torch.ones_like(self.get_visibility())
    #         self._visibility = visibilities_new
    #     torch.cuda.empty_cache()

    #####

    def add_depth_image(self, c2ws, depths, images):
        print(f"Updating depth 3DGS")
        assert depths != None
        H = depths.shape[0]
        W = depths.shape[1]
        x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
        FX = 256.00000000000006
        FY = 256.00000000000006
        CX = W // 2
        CY = H // 2
        xx = (x_grid - CX) / FX
        yy = (y_grid - CY) / FY
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        pts = None
        rgb = None
        widths = None

        for idx, transform in enumerate(c2ws):
            xx = (x_grid - CX) / FX
            yy = (y_grid - CY) / FY
            xx = xx.reshape(-1)
            yy = yy.reshape(-1)

            depth = depths[idx].reshape(-1)
            color = images[idx][:, :, :3].reshape(-1, 3)

            R = (color[:, 0])[depth >= 0]
            G = (color[:, 1])[depth >= 0]
            B = (color[:, 2])[depth >= 0]
            color = np.concatenate(
                (R[..., np.newaxis], G[..., np.newaxis], B[..., np.newaxis]), axis=1
            )
            xx = xx[depth >= 0]
            yy = yy[depth >= 0]
            depth = depth[depth >= 0]

            pts_cam = np.stack([xx * depth, yy * depth, depth], axis=-1)

            pix_ones = np.ones((xx.shape[0], 1), dtype=np.float32)
            pts4 = np.concatenate((pts_cam, pix_ones), axis=1)

            c2w = np.copy(transform)
            pts_cur = (c2w @ pts4.T).T[:, :3]
            radius = (depth / FX).reshape(-1, 1)

            if not isinstance(pts, np.ndarray):
                pts = pts_cur
                rgb = color
                widths = radius
            else:
                pts = np.concatenate((pts, pts_cur), axis=0)
                rgb = np.concatenate((rgb, color), axis=0)
                widths = np.concatenate((widths, radius), axis=0)
        self._xyz = torch.cat(
            (self._xyz, torch.tensor(pts, dtype=torch.float, device="cuda")), dim=0
        )

        fused_color = RGB2SH(torch.tensor(rgb).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        self._features_dc = torch.cat(
            (features[:, :, 0:1].transpose(1, 2).contiguous(), self._features_dc), dim=0
        )
        self._features_rest = torch.cat(
            (features[:, :, 1:].transpose(1, 2).contiguous(), self._features_rest),
            dim=0,
        )

        rots = torch.zeros((pts.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        self._rotation = torch.cat((rots, self._rotation), dim=0)

        widths = torch.tensor(widths, dtype=torch.float, device="cuda")
        scaling = torch.log(torch.sqrt(torch.cat((widths, widths, widths), dim=1)))

        self._scaling = torch.cat((self._scaling, scaling), dim=0)
        self._opacity = inverse_sigmoid(
            torch.cat(
                (
                    self._opacity,
                    torch.full(
                        (pts.shape[0], 1), 0.5, dtype=torch.float, device="cuda"
                    ),
                ),
                dim=0,
            )
        )
        self.max_radii2D = torch.cat(
            (
                self.max_radii2D,
                torch.full((pts.shape[0], 1), 0.0, dtype=torch.float, device="cuda"),
            ),
            dim=0,
        )

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree
 
        #####

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group["params"][0]]

                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        #####
        #####

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        #####

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        #####

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

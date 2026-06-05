import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

import time

class Angular_Visibility_Mesh():
    def __init__(self, num_theta=5, num_phi=10) -> None:
        super(Angular_Visibility_Mesh, self).__init__()
        
        self.num_theta = num_theta
        self.num_phi = num_phi
        self.theta_vals = torch.linspace(0, 2*torch.pi, self.num_theta).cuda()
        self.phi_vals = torch.linspace(0, torch.pi, self.num_phi).cuda()
        self._generate_sphere_grid()

        self.spherical_visibility = None
        # self.time_record = {
        #     "build_time": [],
        #     "query_time": []
        # }

    def _generate_sphere_grid(self):
        theta_grid, phi_grid = torch.meshgrid(self.theta_vals, self.phi_vals, indexing='ij')
        x = torch.cos(theta_grid) * torch.sin(phi_grid)
        y = torch.sin(theta_grid) * torch.sin(phi_grid)
        z = torch.cos(phi_grid)
        self.spherical_directions = torch.stack((x, y, z), dim=2).reshape(-1, 3)

    def _sphere_grid_mapping(self, theta, phi):
        theta_idx = torch.floor((theta - self.theta_vals[0]) / (self.theta_vals[-1]) * (self.num_theta)).long()
        phi_idx = torch.floor((phi - self.phi_vals[0]) / (self.phi_vals[-1]) * (self.num_phi )).long()
        idx = theta_idx * self.num_phi + phi_idx
        return idx

    def _generate_rays_from_indices(self, camera, means):
        """
        compute ray origin & direction from camera object

        Args:
            camera: 3dgs camera object

        Returns:
            camera_origin: 3 dimensional tensor in world coordinates
            camera_direction: 3 dimensional tensor in world coordinates

        """

        # Transform camera-space position to world space
        world_pos = camera.camera_center

        epsilon = 1e-8
        # breakpoint()
        directions = means - world_pos
        
        camera_rays = directions / (torch.norm(directions, dim=1) + epsilon).unsqueeze(1)

        assert camera_rays.shape == (means.shape[0], 3)
        
        return camera_rays

    def _likelihood_on_sphere(self, dot_val, kappa):
        # sigma 1.3, is identical to dot product from -1, to 1
        # print(sigma)
        # alpha = torch.acos(dot_val.clip(-1+1e-7, 1-1e-7))  # clamp to valid domain
        # ang_vis = torch.exp(-0.5*(alpha/sigma)**2)
        
        # note: reimplement using kappa = 1.5, and y = exp(kappa*dot_val - kappa)
        # ang_vis = torch.where(dot_val < 0.0, torch.tensor(0.0), dot_val)
        
        ang_vis = torch.exp(kappa*dot_val - kappa)
        
        return ang_vis

    def _update_sphere(self, camera_rays, transmittance, kappa):
        N = camera_rays.shape[0]
        D = self.spherical_directions.shape[0]
        
        spherical_directions_dupl = self.spherical_directions.unsqueeze(0).repeat(N, 1, 1)
        camera_rays_dupl = camera_rays.unsqueeze(1).repeat(1, D, 1)

        similarity = torch.sum(spherical_directions_dupl * camera_rays_dupl, dim=-1)

        assert torch.max(similarity).item() <= 1. + 1e-3
        assert torch.min(similarity).item() >= -1. - 1e-3
        assert similarity.shape == (N, D)

        likelihood = 1.0 - (self._likelihood_on_sphere(similarity, kappa) * (transmittance).unsqueeze(1))
        
        if likelihood.ndim == 3:
            likelihood = likelihood.squeeze(1)
            
        self.spherical_visibility += torch.log(likelihood)

    def pdf(self, similarity, spherical_directions_dupl):
        _, nearest_indices = similarity.max(1) 
        spherical_directions_nearest = spherical_directions_dupl[torch.arange(similarity.shape[0]), nearest_indices]
        phis_nearest = torch.acos(spherical_directions_nearest[:, 2])
        thetas_nearest = torch.atan2(spherical_directions_nearest[:, 1], spherical_directions_nearest[:, 0])
        thetas_nearest = thetas_nearest % (2*torch.pi)

        indices = self._sphere_grid_mapping(thetas_nearest, phis_nearest)
        
        row_indices = torch.arange(similarity.shape[0])
        pdf = self.spherical_visibility[row_indices, indices]

        return 1.0 - torch.exp(pdf)
        
    def V0(self):
        return torch.sum(self.spherical_visibility, dim=-1) / self.spherical_visibility.shape[-1]

    def build_angular_visibility(self, camera, means, field_outputs=None, transmittance=None, kappa=1.5):
        """
            update visibility tables based on new pose info, 

            Args:
                camera: train view 3dgs camera object containing all extrinsics, intrinsics and other camera info
                field_outputs: gaussian parameter outputs from rasterizing viewing pose

            Returns:
            
        """
        # start_time = time.time()
        N = means.shape[0]
        if self.spherical_visibility == None:
            self.spherical_visibility = torch.zeros((N, self.num_theta * self.num_phi), dtype=torch.float32).cuda()

        camera_rays = self._generate_rays_from_indices(camera, means)
        # torch.cuda.empty_cache()
        if field_outputs is not None:
            transmittance = field_outputs["debug_T"]
            ray_indices = field_outputs["pixel_idx"]
            gaussian_ids = field_outputs["gaussian_id"]

            ray_indices = ray_indices[gaussian_ids != -1]
            transmittance = transmittance[gaussian_ids != -1]
            gaussian_ids = gaussian_ids[gaussian_ids != -1]

            transmittance_per_gaussian = torch.zeros((N), dtype=torch.float32).cuda()

            transmittance_per_gaussian.index_copy_(0, gaussian_ids.long(), transmittance)
        elif transmittance is not None:
            transmittance_per_gaussian = transmittance.clone().detach()
        else:
            raise ValueError("Either field_outputs or transmittance must be provided")

        self._update_sphere(camera_rays, transmittance_per_gaussian, kappa)
        # end_time = time.time()
        # self.time_record["build_time"].append(end_time - start_time)
        # print("avg build time: ", sum(self.time_record["build_time"]) / len(self.time_record["build_time"]))

    def query_angular_visibility(self, camera, means):
        """
            update visibility tables based on new pose info, 

            Args:
                camera: query view 3dgs camera object containing all extrinsics, intrinsics and other camera info
                field_outputs: gaussian parameter outputs from rasterizing qeury pose

            Returns: angular visibility values
            
        """
        # start_time = time.time()
        

        camera_rays = self._generate_rays_from_indices(camera, means)

        D = self.spherical_directions.shape[0]
        N = camera_rays.shape[0]
        camera_rays_dupl = camera_rays.unsqueeze(1).repeat(1, D, 1)
        spherical_directions_dupl = self.spherical_directions.unsqueeze(0).repeat(N, 1, 1)
        similarity = torch.sum(spherical_directions_dupl* camera_rays_dupl, dim=-1)
        
        assert torch.max(similarity).item() <= 1. + 1e-3
        assert torch.min(similarity).item() >= -1.0 - 1e-3
        assert similarity.shape == (N, D)

        # end_time = time.time()
        # self.time_record["query_time"].append(end_time - start_time)
        # print("avg query time: ", sum(self.time_record["query_time"]) / len(self.time_record["query_time"]))
        return self.pdf(similarity, spherical_directions_dupl)
        # return self.V0()
        # likelihood = self._likelihood_on_sphere(similarity)

    def postprocess(self):
        pass
        
    def reset_angular_visibility(self, ):
        """
            reset visibility tables for new 3DGS representaiton
            
        """
        # del self.spherical_visibility
        self.spherical_visibility = None
        # torch.cuda.empty_cache()
    
    def apply_particle_mask(self, mask):
        self.spherical_visibility = self.spherical_visibility[mask]

    def save_state(self, path):
        torch.save({"spherical_visibility": self.spherical_visibility}, path)

    def load_state(self, path):
        state = torch.load(path, map_location="cuda")
        self.spherical_visibility = state["spherical_visibility"]
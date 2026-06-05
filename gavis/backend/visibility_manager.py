from contextlib import contextmanager
import os

import numpy as np
import torch
import math
import copy
import gc

from diff_gaussian_rasterization import GaussianRasterizationSettings as DiffGaussianRasterizationSettings
from diff_gaussian_rasterization import GaussianRasterizer as DiffGaussianRasterizer

from gavis_rasterizer import GaussianRasterizationSettings as GAVISRasterizationSettings
from gavis_rasterizer import GaussianRasterizer as GAVISRasterizer

from .scene.gaussian_model import GaussianModel
from .utils.sh_utils import eval_sh

from .renderer import *

from .renderer.entropy_renderers import *
from .mesh_angular_vis import Angular_Visibility_Mesh
from .sh_angular_vis import Angular_Visibility_SH
from .position_vis import Position_Visibility

import matplotlib.pyplot as plt


class VisibilityManager:
    def __init__(self, pc, pipe, use_angular_vis_learned=True,  kappa=1.5, num_fixed_gaussians = 0,
                 fixed_gaussians=False, surface_fixed = True,  use_angular_vis_fixed = False, background_kappa=None, bounding_box=None,
                 cfg = None):

        if not fixed_gaussians:
            assert not use_angular_vis_fixed
            assert not surface_fixed

        self.cfg = cfg
        self.pc = pc
        self.pipe = pipe
        self.kappa = kappa
        self.background_kappa = background_kappa
        self.bounding_box = bounding_box
        self.starting_num_fixed_gaussians = num_fixed_gaussians
        self.num_fixed_gaussians = num_fixed_gaussians


        #defien all flags needed
        self.use_angular_vis_learned = use_angular_vis_learned
        self.fixed_gaussians = fixed_gaussians
        self.surface_fixed = surface_fixed
        self.use_angular_vis_fixed = use_angular_vis_fixed

        # Internal state: for learned gaussians we store a tensor,
        # for fixed gaussians we store a separate tensor.
        self.visibility_learned = None
        self.visibility_fixed = None
        self.visibility = None

        #mesh object will be instantiated on the first preprocess
        self._mesh_angular_vis_learned = None
        self._mesh_angular_vis_fixed = None

        if self.use_angular_vis_fixed:
            if self.cfg.vis.mode.startswith("pos") or self.cfg.vis.fixed_gaussian_kappa == 0.0:
                print("using pos vis for fixed gaussians")
                self._mesh_angular_vis_fixed = Position_Visibility()
            elif self.cfg.vis.mode == "mesh":
                print("using mesh vis for fixed gaussians")
                self._mesh_angular_vis_fixed = Angular_Visibility_Mesh(num_theta=self.cfg.vis.bg_num_theta, num_phi=self.cfg.vis.bg_num_phi)
            elif self.cfg.vis.mode == "sh":
                print("using sh vis for fixed gaussians")
                self._mesh_angular_vis_fixed = Angular_Visibility_SH(kappa=self.cfg.vis.fixed_gaussian_kappa, **self.cfg.vis.sh)
            else:
                raise ValueError(f"Unknown visibility mode {self.cfg.vis.mode}, should be 'mesh' or 'sh'")

        if self.use_angular_vis_learned:
            if self.cfg.vis.mode.startswith("pos"):
                print("using pos vis")
                self._mesh_angular_vis_learned = Position_Visibility()
            elif self.cfg.vis.mode == "mesh":
                print("using mesh vis")
                self._mesh_angular_vis_learned = Angular_Visibility_Mesh(num_theta=self.cfg.vis.bg_num_theta, num_phi=self.cfg.vis.bg_num_phi)
            elif self.cfg.vis.mode == "sh":
                print("using sh vis")
                self._mesh_angular_vis_learned = Angular_Visibility_SH(kappa=self.cfg.vis.kappa, **self.cfg.vis.sh)
            else:
                raise ValueError(f"Unknown visibility mode {self.cfg.vis.mode}, should be 'mesh' or 'sh'")

    @contextmanager
    def _temporary_random_seed(self, seed):
        if seed is None:
            yield
            return

        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    @torch.no_grad()
    def _update_learned_visibility(self, all_transmittance, camera_views):
        """
        Updates the visibility for learned (dynamic) gaussians.
        Depending on use_angular_vis, it either uses a direct (position-based)
        update or an angular update via entropy computation.
        """

        #filter out first n columns of the transmittances
        if self.fixed_gaussians:
            transmittance_learned = all_transmittance[:, self.num_fixed_gaussians: ]
        else:
            transmittance_learned = all_transmittance

        if self.use_angular_vis_learned:
            m_learned = self._mesh_angular_vis_learned
            skip_sh_learned = isinstance(
                m_learned, Angular_Visibility_SH
            ) and m_learned.skip_build_after_checkpoint_load
            if not skip_sh_learned:
                m_learned.reset_angular_visibility()
            for i, camera in enumerate(camera_views):
                if skip_sh_learned:
                    continue
                if self.fixed_gaussians:
                    m_learned.build_angular_visibility(
                        camera,
                        self.pc.get_xyz[self.num_fixed_gaussians:, :],
                        transmittance=transmittance_learned[i, :],
                        kappa=self.kappa,
                    )
                else:
                    m_learned.build_angular_visibility(
                        camera,
                        self.pc.get_xyz,
                        transmittance=transmittance_learned[i, :],
                        kappa=self.kappa,
                    )
            if not skip_sh_learned:
                m_learned.postprocess()
            if skip_sh_learned:
                m_learned.clear_checkpoint_build_skip()

            #placeholder for downstream methods
            self.visibility_learned = -1 * torch.ones_like(transmittance_learned[0, :])

        else:
            viz =  1 - transmittance_learned
            product = torch.prod(viz, dim = 0)
            learned_visibility = 1 - product


            self.visibility_learned = learned_visibility
    
    @torch.no_grad()
    def _update_fixed_visibility(self, all_transmittance, camera_views):
        """
        Updates visibility for fixed gaussians.
        Treats surface and volume fixed gaussians differently
        """

        #retain out first n columns of the transmittances
        transmittance_fixed = all_transmittance[:, :self.num_fixed_gaussians]

        if self.use_angular_vis_fixed:
            m_fixed = self._mesh_angular_vis_fixed
            skip_sh_fixed = isinstance(
                m_fixed, Angular_Visibility_SH
            ) and m_fixed.skip_build_after_checkpoint_load
            if not skip_sh_fixed:
                m_fixed.reset_angular_visibility()
            for i, camera in enumerate(camera_views):
                if skip_sh_fixed:
                    continue
                m_fixed.build_angular_visibility(
                    camera,
                    self.pc.get_xyz[: self.num_fixed_gaussians, :],
                    transmittance=transmittance_fixed[i, :],
                    kappa=self.background_kappa,
                )
            if skip_sh_fixed:
                m_fixed.clear_checkpoint_build_skip()

            #placeholder for downstream methods
            # self.visibility_fixed = -1 * torch.ones_like(transmittance_fixed[0, :])

        # else:
        viz =  1 - transmittance_fixed
        product = torch.prod(viz, dim = 0)
        fixed_visibility = 1 - product

        self.visibility_fixed = fixed_visibility

    # --- Public API Methods ---

    # create a deep copy of the gaussian model, this is used to avoid overwriting the gaussian model in the scene
    # this gaussian model will contain both fixed and learned gaussians
    def set_pc(self, pc):
        self.pc = None
        self.pc = copy.deepcopy(pc)

    # save the gaussian model to a file
    def save_pc(self, path):
        self.pc.save_ply(path)

    def save_visibility_state(self, dirpath):
        os.makedirs(dirpath, exist_ok=True)
        torch.save(
            {
                "visibility": self.visibility,
                "visibility_fixed": self.visibility_fixed,
                "visibility_learned": self.visibility_learned,
                "num_fixed_gaussians": self.num_fixed_gaussians,
            },
            os.path.join(dirpath, "visibility_tensors.pt"),
        )
        if self._mesh_angular_vis_learned is not None:
            self._mesh_angular_vis_learned.save_state(
                os.path.join(dirpath, "angular_vis_learned.pt")
            )
        if self._mesh_angular_vis_fixed is not None:
            self._mesh_angular_vis_fixed.save_state(
                os.path.join(dirpath, "angular_vis_fixed.pt")
            )

    def load_visibility_state(self, dirpath):
        data = torch.load(
            os.path.join(dirpath, "visibility_tensors.pt"), map_location="cuda"
        )
        self.visibility = data["visibility"]
        self.visibility_fixed = data["visibility_fixed"]
        self.visibility_learned = data["visibility_learned"]
        self.num_fixed_gaussians = data["num_fixed_gaussians"]
        print("########################################################")
        print("Loaded visibility state from: ", dirpath, "num_fixed_gaussians: ", self.num_fixed_gaussians)
        print("########################################################")
        path_learned = os.path.join(dirpath, "angular_vis_learned.pt")
        if self._mesh_angular_vis_learned is not None and os.path.isfile(path_learned):
            self._mesh_angular_vis_learned.load_state(path_learned)
        path_fixed = os.path.join(dirpath, "angular_vis_fixed.pt")
        if self._mesh_angular_vis_fixed is not None and os.path.isfile(path_fixed):
            self._mesh_angular_vis_fixed.load_state(path_fixed)

    #call once per timestep
    @torch.no_grad()
    def preprocess(self, camera_views, bg_color, scaling_modifier=1.0):
        """
        Refines the visibility state using a list of training camera views.
        For each view, the internal visibility state is updated.

        """

        # Reset internal visibility states.
        self.visibility_learned = None
        self.visibility_fixed = None
        self.visibility = None
        self.num_fixed_gaussians = self.starting_num_fixed_gaussians

        # Populate virtual gaussians, seed is used to ensure reproducibility
        if self.fixed_gaussians:
            virtual_gaussians_seed = getattr(self.cfg.vis, "virtual_gaussians_seed", 1)
            with self._temporary_random_seed(virtual_gaussians_seed):
                self.pc.add_background(boundaries = self.bounding_box, use_surface=self.surface_fixed, scale = self.cfg.vis.background_scale, density = self.cfg.vis.background_density)
            self.num_fixed_gaussians = self.pc.N
        #for scene with m fixed gaussians and n learned gaussians and camera_views of length a:
        #obtain transmittances for each of the gaussians for each view and concatenate into tensor of size: (n+m) X a
        # print('background density', self.cfg.vis.background_density)
        transmittance_list = []
        for camera in camera_views:
            transmittance = update_position_visibility(camera, self.pc, self.pipe, torch.tensor([0,0,0], dtype = torch.float), scaling_modifier=1.0, only_retrieve_transmittance=True)
            transmittance_list.append(transmittance)

        all_transmittance = torch.stack(transmittance_list, dim = 0).squeeze(2)


        #update visibilities
        self._update_learned_visibility(all_transmittance, camera_views)


        if self.fixed_gaussians:
            self._update_fixed_visibility(all_transmittance, camera_views)
            #prune fixed gaussians
            # print("fixed gaussian visibility size before prune", self.visibility_fixed.shape)
            num_pruned, visibility_mask = self.pc.prune_visible_background(self.visibility_fixed, threshold = 0.95)
            original_fixed_gaussians = self.visibility_fixed.shape[0]
            print(f"original fixed gaussians: {original_fixed_gaussians}")
            print(f"pruned gaussians: {num_pruned}")
            print(
                f"pruned percentage: "
                f"{(num_pruned / original_fixed_gaussians * 100.0) if original_fixed_gaussians > 0 else 0.0:.2f}%"
            )
            if self.use_angular_vis_fixed:
                # self._mesh_angular_vis_fixed.spherical_visibility = self._mesh_angular_vis_fixed.spherical_visibility[visibility_mask]
                self._mesh_angular_vis_fixed.apply_particle_mask(visibility_mask)
            self.visibility_fixed = self.visibility_fixed[visibility_mask]
            self.num_fixed_gaussians = self.visibility_fixed.shape[0]
            # print("fixed gaussian visibility size after prune", self.visibility_fixed.shape)
    
            self.visibility  = torch.cat([self.visibility_fixed, self.visibility_learned], dim = 0)
        else:
            self.visibility = self.visibility_learned
        
        self.visibility = torch.zeros((self.pc.get_xyz.shape[0], 1), dtype=torch.float32, device="cuda")

    #call once for each candidate view in a timestep
    @torch.no_grad()
    def query_visibility(self, camera, bg_color, scaling_modifier=1.0):
        """
        returns queried visibility of size [N + M, 1]
        For a query camera view, returns the current visibility state for gaussians displaying position visibility 
        and queries position specific current visibility state for gaussians displaying angular visibility
        """
        assert self.visibility is not None
        
        if self.cfg.vis.disable_visibility:
            return torch.ones((self.pc.get_xyz.shape[0]), dtype=torch.float32, device="cuda")
        if self.fixed_gaussians and self.num_fixed_gaussians > 0:
            # access direct attributes of VisibilityManager for position visibility and access attributes of mesh grid object for angular visibility
            if self.use_angular_vis_learned and self.use_angular_vis_fixed:
                self.visibility = torch.zeros((self.pc.get_xyz.shape[0]), dtype=torch.float32, device="cuda")
                self.visibility[:self.num_fixed_gaussians] = self._mesh_angular_vis_fixed.query_angular_visibility(camera, self.pc.get_xyz[:self.num_fixed_gaussians])
                self.visibility[self.num_fixed_gaussians:] = self._mesh_angular_vis_learned.query_angular_visibility(camera, self.pc.get_xyz[self.num_fixed_gaussians:])

            elif self.use_angular_vis_learned and not self.use_angular_vis_fixed:
                self.visibility[self.num_fixed_gaussians:] = self._mesh_angular_vis_learned.query_angular_visibility(camera, self.pc.get_xyz[self.num_fixed_gaussians:]).unsqueeze(1)
                self.visibility[:self.num_fixed_gaussians] = self.visibility_fixed.unsqueeze(1)
                    
            elif not self.use_angular_vis_learned and self.use_angular_vis_fixed:
                self.visibility[:self.num_fixed_gaussians] = self._mesh_angular_vis_fixed.query_angular_visibility(camera, self.pc.get_xyz[:self.num_fixed_gaussians])
                self.visibility[self.num_fixed_gaussians:] = self.visibility_learned.unsqueeze(1)
        elif self.use_angular_vis_learned:
            return self._mesh_angular_vis_learned.query_angular_visibility(camera, self.pc.get_xyz)
        else:
            return self.visibility_learned

        #if fixed and leanred gaussians were oth using position visibility, no additional logic is needed
        return self.visibility
    #--Visualization methods


def project_gaussians_to_image(gaussian_positions, camera):
    """
    Projects 3D gaussian positions to the 2D image plane using the camera's full_proj_transform.
    
    Args:
        gaussian_positions: Tensor of shape (N, 3) containing 3D coordinates.
        camera: An instance of Camera with a full_proj_transform attribute.
        
    Returns:
        image_coords: Tensor of shape (N, 2) with pixel coordinates.
    """
    N = gaussian_positions.shape[0]
    # Convert to homogeneous coordinates by appending ones.
    ones = torch.ones((N, 1), device=gaussian_positions.device)
    homo_points = torch.cat((gaussian_positions, ones), dim=1)  # shape: (N, 4)

    # Apply the full projection transform.
    # full_proj_transform is assumed to be of shape (4,4)
    proj_points = homo_points @ camera.full_proj_transform.t()  # shape: (N, 4)

    # Perform perspective division.
    proj_points = proj_points / proj_points[:, 3:4]

    # Optionally, convert from normalized device coordinates to pixel coordinates.
    # Assume image coordinates are in [0, width] and [0, height]
    x_ndc = proj_points[:, 0]  # in [-1, 1]
    y_ndc = proj_points[:, 1]  # in [-1, 1]
    # Map NDC to pixel coordinates:
    x_pix = (x_ndc + 1) / 2 * camera.image_width
    y_pix = (1 - (y_ndc + 1) / 2) * camera.image_height  # flip y-axis if necessary

    image_coords = torch.stack((x_pix, y_pix), dim=1)
    return image_coords

# Example visualization:
def plot_gaussian_projection(gaussian_positions, visibility, camera):
    """
    Projects the 3D gaussian positions and plots them over the image.
    
    Args:
        gaussian_positions: Tensor of shape (N, 3)
        visibility: Tensor of shape (N, 1) with visibility values.
        camera: An instance of Camera.
    """
    image_coords = project_gaussians_to_image(gaussian_positions, camera)
    image_coords_np = image_coords.detach().cpu().numpy()
    visibility_np = visibility.squeeze().detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(image_coords_np[:, 0], image_coords_np[:, 1],
                c=visibility_np, cmap='viridis', s=30)
    plt.xlim(0, camera.image_width)
    plt.ylim(0, camera.image_height)
    plt.gca().invert_yaxis()  # To match image coordinate convention
    plt.title("Projected Gaussians Colored by Visibility")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.colorbar(label="Visibility")
    plt.show()

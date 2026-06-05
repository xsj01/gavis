import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

import time

from .mesh_angular_vis import Angular_Visibility_Mesh
class Position_Visibility(Angular_Visibility_Mesh):
    def __init__(self, kappa=0.0) -> None:
        self.visibility=None
        self.n_views = 0
        assert kappa == 0.0, f"kappa must be 0.0 for Position_Visibility, got {kappa}"
    
    def V0(self):
        raise NotImplementedError("V0 is not implemented for Angular_Visibility_SH.")

    def build_angular_visibility(self, camera, means, field_outputs=None, transmittance=None, kappa=1.):
        """
            update visibility tables based on new pose info, 

            Args:
                camera: train view 3dgs camera object containing all extrinsics, intrinsics and other camera info
                field_outputs: gaussian parameter outputs from rasterizing viewing pose

            Returns:
            
        """
        # start_time = time.time()
        N = means.shape[0]
        if self.visibility is None:
            self.visibility = torch.zeros(N, dtype=torch.float32).cuda()

        # camera_rays = self._generate_rays_from_indices(camera, means)
        # torch.cuda.empty_cache()
        if field_outputs is not None:
            transmittance = field_outputs["debug_T"]
            # ray_indices = field_outputs["pixel_idx"]
            gaussian_ids = field_outputs["gaussian_id"]

            # ray_indices = ray_indices[gaussian_ids != -1]
            transmittance = transmittance[gaussian_ids != -1]
            gaussian_ids = gaussian_ids[gaussian_ids != -1]

            transmittance_per_gaussian = torch.zeros((N), dtype=torch.float32).cuda()

            transmittance_per_gaussian.index_copy_(0, gaussian_ids.long(), transmittance)
        elif transmittance is not None:
            transmittance_per_gaussian = transmittance.clone().detach()
        else:
            raise ValueError("Either field_outputs or transmittance must be provided")

        self._update_sphere(None, transmittance_per_gaussian, kappa)
        self.n_views += 1 #camera.camera_center.shape[0]
    
    def _update_sphere(self, camera_rays, transmittance, kappa):
        # kappa = round(kappa, 5)
        assert kappa == 0.0, f"kappa must be 0.0 for Position_Visibility, got {kappa}"
        
        # N = camera_rays.shape[0]
        # D = self.spherical_directions.shape[0]
        # self.visibility = 1 - (1-self.visibility) * (1-transmittance)
        self.visibility += transmittance * (1- self.visibility)

    
    def query_angular_visibility(self, camera, means) -> torch.Tensor:
        """
        Query the angular visibility for given camera and means.

        Args:
            camera: Camera object containing camera parameters.
            means: Tensor of shape (N, 3) representing points in space.

        Returns:
            spherical_visibility: Tensor of shape (num_theta, num_phi) representing visibility.
        """
        # print("Querying angular visibility using position visibility...")
        # camera_rays = self._generate_rays_from_indices(camera, means)
        # D = self.sh_coeff.shape[0]
        # N = camera_rays.shape[0]
        # N = camera.camera_center.shape[0]
        return self.visibility.clone().detach()

    def postprocess(self):
        pass
    
    def reset_angular_visibility(self, ):
        """
            reset visibility tables for new 3DGS representaiton
            
        """
        # del self.visibility
        self.visibility = None
        self.n_views = 0
    
    def apply_particle_mask(self, mask):
        self.visibility = self.visibility[mask]

    def save_state(self, path):
        torch.save(
            {"visibility": self.visibility, "n_views": self.n_views},
            path,
        )

    def load_state(self, path):
        state = torch.load(path, map_location="cuda")
        self.visibility = state["visibility"]
        self.n_views = state["n_views"]

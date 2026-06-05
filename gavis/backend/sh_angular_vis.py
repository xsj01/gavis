import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import os

from .mesh_angular_vis import Angular_Visibility_Mesh
# from sh import update_visibility, sh_product, vMF2SH, query_sh, expand_sh, sh_product_orig, sh2f_channel, f2sh_channel, sh2f_bases_dict, f2sh_bases_dict, sh_1, sh_1_minus
from .sh.sh_utils import vMF2SH, sh_product, get_vMF_coeffs, query_sh

try:
    # vMF2SH = torch.compile(vMF2SH)
    # query_sh = torch.compile(query_sh)
    pass
    # sh_product = torch.compile(sh_product)
except:
    pass

class Angular_Visibility_SH(Angular_Visibility_Mesh):
    def __init__(self, sh_degree=5, method='ghant', device='cuda', kappa=1.0, use_flatten_sh=True,
                 indiv_n=False, trans_cutoff=0.001, dtype=torch.float32) -> None:
        self.sh_degree = int(sh_degree)
        self.sh_coeff = None
        self.fourier_coeff = None
        self.n_views_overall = 0
        assert method.lower() in ['amgm', 'exp', 'ghant'], f"Invalid method: {method}. Choose from 'amgm', 'exp', or 'ghant'."
        self.method = method.lower()
        if not isinstance(kappa, list):
            kappa = [kappa]
        self.kappa = [round(k, 5) for k in kappa] # all possible kappa values that could be used during this experiment, need for precomputation of vMF coefficients
        self.device = device
        self.dtype = dtype

        self.indiv_n = indiv_n
        self.trans_cutoff = trans_cutoff
        
        self.n_views = None
        self.n_mask_geq1 = None
        self.n_mask_geq2 = None

        

        if self.method in ['ghant']:
            path = os.path.dirname(os.path.abspath(__file__))
            sh2f_bases_dict = torch.load(os.path.join(path, 'sh', "constants/coefficient_sh2f.pt"))
            f2sh_bases_dict = torch.load(os.path.join(path, 'sh', "constants/coefficient_f2sh.pt"))
            self.sh2f_bases = sh2f_bases_dict[self.sh_degree+1].to(device)#.to(camera_rays)
            self.f2sh_bases = f2sh_bases_dict[self.sh_degree+1].to(device)#.to(camera_rays)

        if self.method in ['ghant']:
            self.use_flatten_sh = False
        else:
            self.use_flatten_sh = use_flatten_sh
        
        self._warmup() # Precompute vMF coefficients and init compiled pytorch function to avoid overhead during training and testing

        # Set True by load_state(); VisibilityManager skips reset/build/postprocess once, then clears.
        self.skip_build_after_checkpoint_load = False

    def _warmup(self):
        # get_vMF_coeffs(self.sh_degree, self.kappa, device=self.device)  # Precompute vMF coefficients
        # print("Warming up Angular_Visibility_SH with kappa values: ", self.kappa)
        # print("Warning: this process should be exluded from runtime eval, it is only for precomputation of vMF coefficients")
        for k in self.kappa:
            get_vMF_coeffs(self.sh_degree, k, device=self.device, flatten=self.use_flatten_sh)
        n=400000
        # get_vMF_coeffs(self.sh_degree, self.kappa, device=self.device)  # Precompute vMF coefficients
        rand_coeff = vMF2SH(self.sh_degree, k, torch.rand((n, 3), device=self.device, dtype=torch.float32), flatten=self.use_flatten_sh)  # Warm up the function
        query_sh(rand_coeff, torch.rand((n, 3), device=self.device, dtype=torch.float32))

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
        if self.skip_build_after_checkpoint_load:
            return
        # start_time = time.time()
        N = means.shape[0]
        # if self.spherical_visibility == None:
        #     self.spherical_visibility = torch.zeros((N, self.num_theta * self.num_phi), dtype=torch.float32).cuda()

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
        
        if self.indiv_n:
            # try:
            self._update_sphere_indiv(camera_rays, transmittance_per_gaussian, kappa)
            # except:
            #     import traceback; traceback.print_exc()
            #     breakpoint()
        else:
            self._update_sphere(camera_rays, transmittance_per_gaussian, kappa)
        self.n_views_overall += 1 #camera_rays.shape[0]
    
    def _update_sphere(self, camera_rays, transmittance, kappa):
        kappa = round(kappa, 5)
        assert kappa in self.kappa, f"kappa {kappa} not in precomputed kappa values: {self.kappa}, this will cause performance issue during the exp, please register kappa at initialization"
        N = camera_rays.shape[0]
        # D = self.spherical_directions.shape[0]
        if self.use_flatten_sh:
            transmittance = transmittance.reshape(-1, 1)  # Ensure transmittance is (N, 1) for flattening
        else:
            transmittance = transmittance.reshape(-1, 1, 1)
        
        if self.sh_coeff is None:

            vMF = vMF2SH(self.sh_degree, kappa, camera_rays, flatten=self.use_flatten_sh)
            vMF *= transmittance
            self.sh_coeff = vMF
        else:
            vMF = vMF2SH(self.sh_degree, kappa, camera_rays, flatten=self.use_flatten_sh)
            vMF *= transmittance
            # prod = sh_product_orig(self.sh_coeff, expand_sh(vMF, self.sh_coeff.shape[1]-1), fft_return_real=False)
            # new_deg = prod.shape[1] - 1
            # self.sh_coeff = expand_sh(self.sh_coeff, new_deg) + expand_sh(vMF, new_deg) - prod
            # prod = sh_product(self.sh_coeff, vMF, fft_return_real=False)
            if self.method == 'ghant':
                prod = sh_product(self.sh_coeff, vMF, fft_return_real=False, bases = (self.sh2f_bases, self.f2sh_bases))
                self.sh_coeff = vMF+self.sh_coeff-prod
            else:
                self.sh_coeff += vMF
    
    def _update_sphere_indiv(self, camera_rays, transmittance, kappa):
        kappa = round(kappa, 5)
        assert kappa in self.kappa, f"kappa {kappa} not in precomputed kappa values: {self.kappa}, this will cause performance issue during the exp, please register kappa at initialization"
        Np = camera_rays.shape[0]

        # print('transmittance shape', transmittance.shape)
        valid_mask = (transmittance>=self.trans_cutoff)#.reshape(-1)

        if self.use_flatten_sh:
            transmittance = transmittance.reshape(-1, 1)  # Ensure transmittance is (N, 1) for flattening
        else:
            transmittance = transmittance.reshape(-1, 1, 1)
        
        valid_transmittance = transmittance[valid_mask].contiguous()
        valid_camera_rays = camera_rays[valid_mask].contiguous()

        if self.sh_coeff is None:
            vMF = vMF2SH(self.sh_degree, kappa, valid_camera_rays, flatten=self.use_flatten_sh)
            vMF *= valid_transmittance
            self.sh_coeff = torch.zeros(Np, *vMF.shape[1:], device=vMF.device, dtype=vMF.dtype)
            self.n_views = torch.zeros(Np, device=vMF.device, dtype=torch.int32)
            self.sh_coeff[valid_mask] = vMF
            self.n_views[valid_mask] = 1
        else:
            vMF = vMF2SH(self.sh_degree, kappa, valid_camera_rays, flatten=self.use_flatten_sh)
            vMF *= valid_transmittance

            if self.method == 'ghant':
                prod = sh_product(self.sh_coeff[valid_mask], vMF, fft_return_real=False, bases = (self.sh2f_bases, self.f2sh_bases))
                self.sh_coeff[valid_mask] = vMF+self.sh_coeff[valid_mask]-prod
            else:
                self.sh_coeff[valid_mask] += vMF
            
            self.n_views[valid_mask] += 1
              
    
    def query_angular_visibility(self, camera, means) -> torch.Tensor:
        """
        Query the angular visibility for given camera and means.

        Args:
            camera: Camera object containing camera parameters.
            means: Tensor of shape (N, 3) representing points in space.

        Returns:
            spherical_visibility: Tensor of shape (num_theta, num_phi) representing visibility.
        """
        # print("Querying angular visibility using SH method...")
        # camera_rays = self._generate_rays_from_indices(camera, means)
        # D = self.sh_coeff.shape[0]
        # N = camera_rays.shape[0]
        camera_rays = self._generate_rays_from_indices(camera, means).reshape(-1, 3)
        # camera_rays_dupl = camera_rays.unsqueeze(1).repeat(1, D, 1).reshape(-1, 3)
        
        # if self.use_flatten_sh:
        #     sh_coeff_dupl = sh_coeff.unsqueeze(0).repeat(N, 1, 1).reshape(-1, sh_coeff.shape[-1])
        # else:
        #     sh_coeff_dupl = sh_coeff.unsqueeze(0).repeat(N, 1, 1, 1).reshape(-1, sh_coeff.shape[-2], sh_coeff.shape[-1])

        # values = query_sh(sh_coeff_dupl, camera_rays).reshape(N, D)

        
        if self.indiv_n:
            # try:
            return self._get_visibility_indiv(camera_rays)
            # except:
            #     import traceback; traceback.print_exc()
            #     breakpoint()
        else:
            return self._get_visibility(camera_rays)
    
    def _get_visibility_indiv(self, camera_rays: torch.Tensor) -> torch.Tensor:
        sh_coeff = self.sh_coeff.real
        # values = torch.zeros((camera_rays.shape[0],), device=sh_coeff.device, dtype=sh_coeff.dtype)
        # values[self.n_mask_geq1] = query_sh(sh_coeff[self.n_mask_geq1].contiguous(), camera_rays[self.n_mask_geq1].contiguous())
        values = query_sh(sh_coeff, camera_rays)
        # values_clone = values.clone()
        # values[self.n_mask_le1] = 0.0
        # try:
        #     assert torch.allclose(values_clone, values)
        # except:
        #     breakpoint()
        
        if self.method == 'ghant':
            return values.clamp(min=0.0, max=1.0)
        if self.n_views_overall <= 1:
            return values
        else:
            if self.method == 'amgm':
                # n = self.n_views[self.n_mask_geq2].to(values.dtype)
                # values[self.n_mask_geq2] = (1-(1-values[self.n_mask_geq2]/n).pow(n)).clamp(min=0.0, max=1.0)
                n = self.n_views_f.clamp(min=1.0)
                values2 = (1-(1-values/n).pow(n)).clamp(min=0.0, max=1.0)
                # values2_clone = values2.clone()
                # values2[self.n_mask_le2] = values[self.n_mask_le2]
                # try:
                #     assert torch.allclose(values2_clone, values2, atol=1e-5)
                # except:
                #     breakpoint()
                return values2
            elif self.method == 'exp':
                # values[self.n_mask_geq2] =  1-torch.exp(-values[self.n_mask_geq2]).clamp(min=0.0, max=1.0)
                values2 = 1-torch.exp(-values).clamp(min=0.0, max=1.0)
                values2[self.n_mask_le2] = values[self.n_mask_le2]
                return values2
            else:
                raise ValueError(f"Unknown method: {self.method}")

    def _get_visibility(self, camera_rays: torch.Tensor) -> torch.Tensor:
        sh_coeff = self.sh_coeff.real
        values = query_sh(sh_coeff, camera_rays)
        if self.method == 'ghant':
            return values.clamp(min=0.0, max=1.0)
        if self.n_views_overall == 1:
            return values
        else:
            if self.method == 'amgm':
                return (1-(1-values/self.n_views_overall).pow(self.n_views_overall)).clamp(min=0.0, max=1.0)
            elif self.method == 'exp':
                return 1-torch.exp(-values).clamp(min=0.0, max=1.0)
            else:
                raise ValueError(f"Unknown method: {self.method}")
    
    def postprocess(self):
        if self.indiv_n:
            # self.n_mask_geq1 = (self.n_views != 0)
            # self.n_mask_geq2 = (self.n_views > 1)
            self.n_mask_le1 = (self.n_views == 0)#self.n_mask_geq1.logical_not()
            self.n_mask_le2 = (self.n_views <= 1)#self.n_mask_geq2.logical_not()
            self.n_views_f = self.n_views.to(self.dtype)

    def reset_angular_visibility(self, ):
        """
            reset visibility tables for new 3DGS representaiton
            
        """
        # del self.sh_coeff
        self.sh_coeff = None
        self.fourier_coeff = None
        self.n_views_overall = 0
        self.n_views = None
        self.n_mask_geq1 = None
        self.n_mask_geq2 = None
        self.skip_build_after_checkpoint_load = False

        # torch.cuda.empty_cache()

    def apply_particle_mask(self, mask):
        self.sh_coeff = self.sh_coeff[mask]

    def save_state(self, path):
        """Persist full SH accumulation state (build_angular_visibility + postprocess)."""
        state = {
            "sh_coeff": self.sh_coeff,
            "n_views_overall": self.n_views_overall,
            "indiv_n": self.indiv_n,
            "fourier_coeff": self.fourier_coeff,
        }
        if self.indiv_n:
            state["n_views"] = self.n_views
            state["n_mask_le1"] = getattr(self, "n_mask_le1", None)
            state["n_mask_le2"] = getattr(self, "n_mask_le2", None)
            state["n_views_f"] = getattr(self, "n_views_f", None)
        torch.save(state, path)

    def load_state(self, path):
        state = torch.load(path, map_location="cuda")
        self.sh_coeff = state["sh_coeff"]
        self.n_views_overall = state["n_views_overall"]
        self.fourier_coeff = state.get("fourier_coeff", None)
        if state.get("indiv_n", False):
            self.n_views = state.get("n_views")
            if "n_mask_le1" in state:
                self.n_mask_le1 = state["n_mask_le1"]
            if "n_mask_le2" in state:
                self.n_mask_le2 = state["n_mask_le2"]
            if "n_views_f" in state:
                self.n_views_f = state["n_views_f"]
        # Next VisibilityManager preprocess should not reset/re-accumulate SH state once.
        self.skip_build_after_checkpoint_load = True

    def clear_checkpoint_build_skip(self):
        """Clear one-shot skip used after load_state (e.g. after a full preprocess pass)."""
        self.skip_build_after_checkpoint_load = False
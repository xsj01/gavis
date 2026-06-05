import torch
import numpy as np

from typing import Dict, Optional, Tuple, Type
from jaxtyping import Float, Int
from torch import Tensor

import nerfacc

from .entropy_utils import format_color, gmm_entropy_upper_bound_huber, gmm_entropy_upper_bound


class BaseEntropyRenderer(torch.nn.Module):
    """Base Entropy Renderer"""
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        field_outputs: Dict[str, Tensor],
        num_rays: int,
        ) -> Float[Tensor, "*bs 1"]:
        """Calculate Entropy along the ray.
        """

        raise NotImplementedError

class DebugRenderer(BaseEntropyRenderer):
    def forward(
        self,
        field_outputs: Dict[str, Tensor],
        num_rays: int,
    )-> Float[Tensor, "*bs 1"]:

        ray_indices = field_outputs["pixel_idx"]
        alphas = field_outputs["alphas"]
        visibility = field_outputs["vis"]
        transmittance = field_outputs["debug_T"]
        gaussian_ids = field_outputs["gaussian_id"]


        ray_indices = ray_indices[gaussian_ids != -1]
        unique_vals, counts = torch.unique(ray_indices, return_counts=True)

        print("frequencies per ray: ", counts)
        print("max intersections: ", counts.max().item())

        alphas = alphas[gaussian_ids != -1]
        visibility = visibility[gaussian_ids != -1]
        visibility = 1 - visibility

        packed_info = nerfacc.pack_info(ray_indices.long(), n_rays = num_rays)
        weights, _ = nerfacc.render_weight_from_alpha(alphas, packed_info= packed_info.long(), ray_indices = ray_indices.long(), n_rays = num_rays)

        output = torch.zeros((num_rays), dtype=torch.float, device="cuda")
        output.index_add_(0, ray_indices.long(), weights)

        # output[output > 0.1] = 1.0

        return output

class VisibilityRenderer(BaseEntropyRenderer):
    def forward(
        self,
        field_outputs: Dict[str, Tensor],
        num_rays: int,
        angular_vis = None
    )-> Float[Tensor, "*bs 1"]:

        ray_indices = field_outputs["pixel_idx"]
        alphas = field_outputs["alphas"]
        visibility = field_outputs["vis"]
        transmittance = field_outputs["debug_T"]
        gaussian_ids = field_outputs["gaussian_id"]

        unique, counts = torch.unique(ray_indices, return_counts=True)
        # print("unique shape: ", unique.shape)
        # print("counts max: ", torch.topk(counts, k=10))

        ray_indices = ray_indices[gaussian_ids != -1]
        alphas = alphas[gaussian_ids != -1]
        visibility = visibility[gaussian_ids != -1]
        visibility = 1 - visibility

        packed_info = nerfacc.pack_info(ray_indices.long(), n_rays = num_rays)
        weights, _ = nerfacc.render_weight_from_alpha(alphas, packed_info= packed_info.long(), ray_indices = ray_indices.long(), n_rays = num_rays)
        output = torch.zeros((num_rays), dtype=torch.float, device="cuda")

        # gaussian_ids = gaussian_ids[gaussian_ids != -1]
        # unique_vals, counts = torch.unique(gaussian_ids.long(), return_counts=True, sorted=True)
        # print("unique values: ", unique_vals)
        # print("frequencies per gaussian: ", counts)
        # print("max intersections: ", counts.max().item())
        # print("mean intersections: ", torch.mean(counts.float()))
        # print("mode intersections: ", torch.mode(counts.float())[0])

        if angular_vis == None:
            output.index_add_(0, ray_indices.long(), weights * visibility)
        else:
            gaussian_ids = gaussian_ids[gaussian_ids != -1]
            angular_vis = angular_vis[gaussian_ids]
            output.index_add_(0, ray_indices.long(), weights * angular_vis)

        return output

class WeightDistributionRenderer(BaseEntropyRenderer):
    """ 
    Entropy estimation method proposed by Lee et al. 2022 RAL https://arxiv.org/abs/2209.08409
    """
    def forward(
        self,
        field_outputs: Dict[str, Tensor],
        num_rays: int,
        ) -> Float[Tensor, "*bs 1"]:
        """Calculate Entropy along the ray.
        """
        ray_indices = field_outputs["pixel_idx"]
        alphas = field_outputs["alphas"]
        transmittance = field_outputs["debug_T"]
        
        ray_indices = ray_indices[transmittance != 0]
        alphas = alphas[transmittance != 0]

        packed_info = nerfacc.pack_info(ray_indices.long(), n_rays = num_rays)

        weights, _ = nerfacc.render_weight_from_alpha(alphas, packed_info= packed_info.long(), ray_indices = ray_indices.long(), n_rays = num_rays)
        weights = weights[..., None]

        comp_logweights = nerfacc.accumulate_along_rays(
                weights[..., 0], values=torch.log(torch.clamp(weights, min=1e-6)), ray_indices=ray_indices, n_rays=num_rays
            )
        accumulated_weight = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
        )

        bg_weight = 1 - accumulated_weight
        neg_entropy = comp_logweights + torch.log(torch.clamp(bg_weight, min=1e-6)) * bg_weight
        entropy = -neg_entropy
        # entropy = torch.zeros((num_rays), dtype=torch.float, device="cuda")
        # entropy.index_add_(0, indices.long(), -weights * torch.log(torch.clamp(weights,1e-6)))

        # print("min entropy: ", torch.min(entropy))
        # print("max value: ", torch.max(entropy))
        # print("sum entropy: ", torch.sum(entropy))

        return entropy


class VisibilityEntropyRenderer(BaseEntropyRenderer):
    beta = 0.95 #0.95
    density_factor = 0.01
    sigma_un = 1.
    sigma_bg = 0.01
    rgb_un = torch.FloatTensor([0.5, 0.5, 0.5])
    rgb_bg = torch.FloatTensor([0., 0., 0.])
    d0 = 6.5  #room=6.5 #blender = 1.0
    depth_decay_factor = 2
    use_huber = True
    use_visibility = True
    use_var = True
    depth_use_equiv = True
    use_depth_corr = False
    use_vis_as_var = False

    def get_equiv_alpha(
        self,
        alpha: Float[Tensor, "*bs"], 
        visibility: Float[Tensor, "*bs"]
        ) -> Float[Tensor, "*bs"]:
        if not self.use_visibility:
            return alpha
        equiv_alpha = (visibility + self.beta * (1-visibility) ) * alpha
        equiv_alpha += (1-self.beta) * (1 - visibility) * (1 - np.exp(- self.density_factor*self.sigma_un))

        return equiv_alpha
    
    def get_gmm_flatten(self, rgb : Float[Tensor, "num_samples 3"], 
                        var_rgb : Float[Tensor, "num_samples 3"], 
                        weight_eqiv: Float[Tensor, "num_samples 1"],
                        visibility: Float[Tensor, "num_samples 1"],
                        ray_indices: Int[Tensor, "num_samples"],
                        num_rays: int,
                        ) -> Tuple[Float[Tensor, "num_samples+2*num_rays 3"], Float[Tensor, "num_samples+2*num_rays 3"], Float[Tensor, "num_samples+2*num_rays 1"], Int[Tensor, "num_samples+2*num_rays"]]:
        '''
        return flatten means, variance, weights, ray_indices
        not sorted by index!
        '''
        # print("flattening gmm")
        if not self.use_vis_as_var:
            weight_1 = weight_eqiv * visibility


            # breakpoint()    
            weight_un = torch.zeros(num_rays, 1, device=weight_eqiv.device, dtype=weight_eqiv.dtype)
            weight_un = weight_un.index_add_(0, ray_indices, weight_eqiv * (1 - visibility))

            assert var_rgb.shape[-1] == 3

            weight_sum = torch.zeros(num_rays, 1, device=weight_eqiv.device, dtype=weight_eqiv.dtype)
            weight_sum = weight_sum.index_add_(0, ray_indices, weight_eqiv)

            weight_bg = torch.clip(1 - weight_sum, min=0., max=1.)
            
            weight = torch.cat([weight_1, weight_bg, weight_un], dim=0)
            
            # print("var max: ", torch.max(var_rgb))
            # print("var min: ", torch.min(var_rgb))
            var_un = torch.ones(num_rays, 3).to(var_rgb) * self.sigma_un **2
            var_bg = torch.ones(num_rays, 3).to(var_rgb) * self.sigma_bg **2

            variance = torch.cat([var_rgb, var_bg, var_un], dim=0)

            rgb_un = torch.ones(num_rays, 3).to(rgb) * self.rgb_un.to(rgb)
            rgb_bg = torch.ones(num_rays, 3).to(rgb) * self.rgb_bg.to(rgb)
            mean = torch.cat([rgb, rgb_bg, rgb_un], dim=0)

            idx_add = torch.LongTensor(np.arange(num_rays)).to(ray_indices)
            ray_indices = torch.cat([ray_indices, idx_add, idx_add])
        
        else:

            # weight_1 = weight_eqiv

            # assert var_rgb.shape[-1] == 3

            # weight_sum = torch.zeros(num_rays, 1, device=weight_eqiv.device, dtype=weight_eqiv.dtype)
            # weight_sum = weight_sum.index_add_(0, ray_indices, weight_eqiv)

            # weight_bg = torch.clip(1 - weight_sum, min=0., max=1.)

            # weight = torch.cat([weight_1, weight_bg], dim=0)
            
            # var_bg = torch.ones(num_rays, 3).to(var_rgb) * self.sigma_bg **2

            # # var_rgb = torch.ones(num_rays, 3).to(var_rgb) * (1.0 - visibility[..., None])
            # # var_rgb = (((1.0 - visibility)*10.0)**2).expand(-1, 3)
            # var_rgb = ((1.0 - visibility)).expand(-1, 3) * 0.5
            # # var_rgb = ((1.0 - visibility)).expand(-1, 3) * 0.5


            # variance = torch.cat([var_rgb, var_bg], dim=0)

            # rgb_bg = torch.ones(num_rays, 3).to(rgb) * self.rgb_bg.to(rgb)
            # mean = torch.cat([rgb, rgb_bg], dim=0)
            # idx_add = torch.LongTensor(np.arange(num_rays)).to(ray_indices)
            # ray_indices = torch.cat([ray_indices, idx_add])

            weight_1 = weight_eqiv * visibility
  
            weight_un = torch.zeros(num_rays, 1, device=weight_eqiv.device, dtype=weight_eqiv.dtype)
            weight_un = weight_un.index_add_(0, ray_indices, weight_eqiv * (1 - visibility))

            assert var_rgb.shape[-1] == 3

            weight_sum = torch.zeros(num_rays, 1, device=weight_eqiv.device, dtype=weight_eqiv.dtype)
            weight_sum = weight_sum.index_add_(0, ray_indices, weight_eqiv)

            weight_bg = torch.clip(1 - weight_sum, min=0., max=1.)
            
            weight = torch.cat([weight_1, weight_bg, weight_un], dim=0)
            
            var_un = torch.ones(num_rays, 3).to(var_rgb) * self.sigma_un **2
            var_bg = torch.ones(num_rays, 3).to(var_rgb) * self.sigma_bg **2

            variance = torch.cat([var_rgb, var_bg, var_un], dim=0)

            # rgb_un = torch.ones(num_rays, 3).to(rgb) * self.rgb_un.to(rgb)
            rgb_un = ((1.0 - visibility)).expand(-1, 3)
            rgb_bg = torch.ones(num_rays, 3).to(rgb) * self.rgb_bg.to(rgb)
            mean = torch.cat([rgb, rgb_bg, rgb_un], dim=0)

            idx_add = torch.LongTensor(np.arange(num_rays)).to(ray_indices)
            ray_indices = torch.cat([ray_indices, idx_add, idx_add])

        return mean, variance, weight, ray_indices
    
    def get_gmm(
        self,
        field_outputs: Dict[str, Tensor],
        num_rays: int,
        )-> Float[Tensor, "*bs 1"]:

        # print("batched gmm")

        min_entropy = 0.5 * (3 * np.log(2 * np.pi * np.e) + 3* np.log (self.sigma_bg **2))

        # density = field_outputs[FieldHeadNames.DENSITY]
        # visibility = field_outputs[FieldHeadNames.VISIBILITY]
        ray_indices = field_outputs["pixel_idx"]
        alphas = field_outputs["alphas"]
        visibility = field_outputs["vis"]
        transmittance = field_outputs["debug_T"]
        gaussian_ids = field_outputs["gaussian_id"]
        rgb = field_outputs["color"]
        
        ray_indices = ray_indices[gaussian_ids != 0]
        alphas = alphas[gaussian_ids != 0]
        visibility = visibility[gaussian_ids != 0]
        rgb = rgb[gaussian_ids != 0]
        rgb = format_color(rgb)

        # print("")

        torch.cuda.empty_cache()

        if self.use_var:
            # rgb_var = field_outputs[FieldHeadNames.RGB_VARIANCE]
            pass
        else:
            rgb_var = torch.ones_like(rgb) * self.sigma_bg **2

        # equiv_density = self.get_equiv_density(density, visibility, ray_samples)
        equiv_alpha = self.get_equiv_alpha(alphas, visibility)
        

        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weight_eqiv, _ = nerfacc.render_weight_from_alpha(equiv_alpha, packed_info= packed_info.long(), ray_indices = ray_indices.long(), n_rays = num_rays)

        gmm_output = self.get_gmm_flatten(rgb, rgb_var, weight_eqiv[..., None], visibility[..., None], ray_indices, num_rays)
        gmm = gmm_output[:3]

        ray_indices = gmm_output[-1]

        return gmm, (ray_indices, num_rays)

    
    def forward(
        self,
        field_outputs: Dict[str, Tensor],
        num_rays: int,
        depth,
        angular_vis,
        var_color = None,
        ) -> Float[Tensor, "*bs 1"]:
        """Calculate Entropy along the ray.
        """
        # breakpoint()
        # min_entropy = 0.5 * (3 * np.log(2 * np.pi * np.e) + 3* np.log (self.sigma_bg **2))
        min_entropy = 0.5 * (3 * np.log(2 * np.pi * np.e) + 3* np.log (self.sigma_bg **2))

        # density = field_outputs[FieldHeadNames.DENSITY]
        # visibility = field_outputs[FieldHeadNames.VISIBILITY]
        # rgb_var = field_outputs[FieldHeadNames.RGB_VARIANCE]
        # rgb = field_outputs[FieldHeadNames.RGB]
        # breakpoint()
        ray_indices = field_outputs["pixel_idx"]
        alphas = field_outputs["alphas"]
        visibility = field_outputs["vis"]
        transmittance = field_outputs["debug_T"]
        gaussian_ids = field_outputs["gaussian_id"]
        rgb = field_outputs["color"]
        rgb_var = field_outputs["color_var"]

        # print("counts: ", counts)

        # print("transmit shape: ", transmittance.shape)
        # print("gaussian_ids shape: ", gaussian_ids.shape)


        # ray_indices = ray_indices[transmittance != 0]
        # alphas = alphas[transmittance != 0]
        # visibility = visibility[transmittance != 0]
        # visibility = 1 - visibility
        # rgb_var = rgb_var[transmittance != 0]
        # # rgb = rgb[transmittance != 0,:]
        # rgb = format_color(rgb, transmittance)

        # breakpoint()

        ray_indices = ray_indices[gaussian_ids != -1]
        torch.cuda.empty_cache()

        alphas = alphas[gaussian_ids != -1]
        torch.cuda.empty_cache()
        visibility = visibility[gaussian_ids != -1]
        torch.cuda.empty_cache()

        rgb_var = rgb_var[gaussian_ids != -1]
        torch.cuda.empty_cache()
        # rgb = rgb[transmittance != 0,:]
        # rgb = format_color(rgb, gaussian_ids)
        rgb = format_color(rgb, gaussian_ids)
        torch.cuda.empty_cache()
        gaussian_ids = gaussian_ids[gaussian_ids != -1]
        if angular_vis == None:
            visibility = 1 - visibility
        else:
            # print("using angular nvf in renderer init")
            visibility = angular_vis[gaussian_ids]

        torch.cuda.empty_cache()

        if not self.use_visibility:
            visibility = torch.ones_like(visibility)
        
        if not self.use_var:
            rgb_var = torch.ones_like(rgb) * self.sigma_bg **2
        elif var_color is not None:
            rgb_var = var_color[gaussian_ids]
        else:
            rgb_var = torch.ones_like(rgb) * self.sigma_bg **2
        torch.cuda.empty_cache()

        # equiv_density = self.get_equiv_density(density, visibility, ray_samples)
        equiv_alpha = self.get_equiv_alpha(alphas, visibility)

        # print("equiv_alpha_debug_begin")
        # print("min: ", torch.min(equiv_alpha))
        # print("max: ", torch.max(equiv_alpha))
        # print("sum: ", torch.sum(equiv_alpha))
        # print("equiv_alpha_debug_end")

        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weight_eqiv, _ = nerfacc.render_weight_from_alpha(equiv_alpha, packed_info= packed_info.long(), ray_indices = ray_indices.long(), n_rays = num_rays)

        # print("equiv_weight_debug_begin")
        # print("min: ", torch.min(weight_eqiv))
        # print("max: ", torch.max(weight_eqiv))
        # print("sum: ", torch.sum(weight_eqiv))
        # print("equiv_weight_debug_end")

        gmm_output = self.get_gmm_flatten(rgb, rgb_var, weight_eqiv[..., None], visibility[..., None], ray_indices, num_rays)
        gmm = gmm_output[:3]

        ray_indices = gmm_output[-1]
        
        if self.use_huber:
            entropy = gmm_entropy_upper_bound_huber(*gmm, ray_indices=ray_indices, num_rays=num_rays)
        else:
            entropy = gmm_entropy_upper_bound(*gmm, ray_indices=ray_indices, num_rays=num_rays)

        torch.cuda.empty_cache()

        if self.d0>0.:
            # depth needs to be passed as a flattened matrix
            # pass
            # if self.depth_use_equiv:
            #     depth = self.depth_renderer(weight_eqiv, ray_samples)
            # else:
            #     if flatten:
            #         packed_info = nerfacc.pack_info(ray_indices, num_rays)
            #         weight_orig = nerfacc.render_weight_from_density(
            #             t_starts=ray_samples.frustums.starts[..., 0],
            #             t_ends=ray_samples.frustums.ends[..., 0],
            #             sigmas=density[..., 0],
            #             packed_info=packed_info,
            #             )[0]
            #         weight_orig = weight_orig[..., None]
            #     else:
            #         weight_orig = ray_samples.get_weights(density)
                
            #     depth = self.depth_renderer(weight_orig, ray_samples)
            # depth = depth.view(-1)
            # factor = torch.ones_like(depth)
            # factor[depth<self.d0] = depth[depth<self.d0] 
            
            # breakpoint()
            # depth = depth.clip(0, self.d0*2)
            if self.use_depth_corr:
                factor = 1 - torch.exp(-depth/self.d0)
            else:
                factor = (torch.clip(depth/self.d0, 0., 1.)**self.depth_decay_factor)
            
            entropy = entropy.reshape(depth.shape)
            entropy = (entropy - min_entropy ) *factor + min_entropy
            # entropy = entropy * factor
            # entropy = torch.where(depth > 12, -30.0, entropy)

        entropy = entropy.clip(min = min_entropy)
        # entropy = entropy.clip(min = -10000000)
        # print("min entropy: ", torch.min(entropy))
        # print("max value: ", torch.max(entropy))
        # print("mean entropy: ", torch.mean(entropy))
        # breakpoint()

        # entropy = torch.where(entropy <= -23.37, 1.0, 0.0)

        # self.min_val = -25
        # self.max_val = 4
        # entropy = (entropy - self.min_val) / (self.max_val - self.min_val)

        return entropy
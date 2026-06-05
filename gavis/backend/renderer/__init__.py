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
import math
import time

from diff_gaussian_rasterization import GaussianRasterizationSettings as DiffGaussianRasterizationSettings 
from diff_gaussian_rasterization import GaussianRasterizer as DiffGaussianRasterizer

from gavis_rasterizer import GaussianRasterizationSettings as GAVISRasterizationSettings
from gavis_rasterizer import GaussianRasterizer as GAVISRasterizer


from ..scene.gaussian_model import GaussianModel
from ..utils.sh_utils import eval_sh

import nerfacc

from .entropy_renderers import *


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    render_mode="RGB",
    use_angular_vis=True,
    vars=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if render_mode == "RGB":

        raster_settings = DiffGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
            antialiasing=False
        )
        rasterizer = DiffGaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # rendered_image, radii, depth = rasterizer(

    if render_mode == "RGB":
        rendered_image, radii, depth = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth}


def render_entropy(
    viewpoint_camera,
    pc: GaussianModel,
    rgb_var: torch.Tensor,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    visibility = None,
    d0 = 1.0, # depth correlation parameter
    prior_alpha = 0.15, # prior alpha for visibility
    beta = 0.50,
    Q_0 = 1.0,
    Q_bg = 0.01,
    cfg = None,
    visibility_handler = None,
):
    """
    pixel entropy.

    Background tensor (bg_color) must be on GPU!
    """
    assert cfg is not None
    if cfg is not None:
        d0 = cfg.bn.d0
        prior_alpha = cfg.bn.alpha0
        beta = cfg.bn.beta
        Q_0 = cfg.bn.Q0
        Q_bg = cfg.bn.Qbg
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            visibility_handler.pc.get_xyz, dtype=visibility_handler.pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    # if use_surface:
    #     prior_alpha = 0.95

    # print("custom rasterizer called")
    raster_settings = GAVISRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=visibility_handler.pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False, #True,
        d0=d0,
        prior_alpha=prior_alpha,
        beta=beta,
        Q_0=Q_0,
        Q_bg=Q_bg
    )

    rasterizer = GAVISRasterizer(raster_settings=raster_settings)

    means3D = visibility_handler.pc.get_xyz
    means2D = screenspace_points
    opacity = visibility_handler.pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = visibility_handler.pc.get_covariance(scaling_modifier)
    else:
        scales = visibility_handler.pc.get_scaling
        rotations = visibility_handler.pc.get_rotation

    assert visibility_handler is not None
    bg_color = torch.tensor([0.0, 0.0, 0.0]).cuda()
    visibility = visibility_handler.query_visibility(viewpoint_camera, bg_color)

    num_gaussians = means3D.shape[0]
    _gaussian_shape_checks = [
        ("rgb_var", rgb_var),
        ("visibility", visibility),
        ("opacity", opacity),
        ("scales", scales),
        ("rotations", rotations),
    ]
    _mismatches = []
    for _name, _t in _gaussian_shape_checks:
        if _t is None:
            continue
        if _t.shape[0] != num_gaussians:
            _d0 = _t.shape[0]
            _mismatches.append(
                f"  {_name}: shape[0]={_d0} (full shape={tuple(_t.shape)}), "
                f"expected {num_gaussians} (diff={_d0 - num_gaussians:+d})"
            )
    if _mismatches:
        raise AssertionError(
            "render_entropy: first-dim mismatch vs means3D / num_gaussians=%d\n%s"
            % (num_gaussians, "\n".join(_mismatches))
        )

    image, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        visibilities=visibility,
        rgb_var=rgb_var,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp
    )
    
    return image

def update_position_visibility(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    only_retrieve_transmittance=False,
):
    """
    Update position visibility

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GAVISRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False, #True,
        d0=1.0,
        prior_alpha=0.05,
        beta=0.5,
        Q_0=1.0,
        Q_bg=0.01
    )

    rasterizer = GAVISRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    transmittance = rasterizer.get_position_visibility(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp
    )
    
    if not only_retrieve_transmittance:
        raise NotImplementedError("update_position_visibility with only_retrieve_transmittance flag=true is deprecated")
        visibility = 1.0 - transmittance
        pc.update_visibility(visibility)
    else:
        return transmittance

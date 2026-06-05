import os
import warnings
from random import randint
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch

from gavis.backend.renderer import render
from gavis.backend.scene.cameras import Camera
from gavis.backend.scene.gaussian_model import GaussianModel
from gavis.backend.utils.general_utils import get_expon_lr_func
from gavis.backend.utils.graphics_utils import BasicPointCloud, fov2focal
from gavis.backend.utils.loss_utils import l1_loss, ssim
from gavis.pose_utils import c2w_to_RT


def _default_opt(
    iterations: int,
    lambda_dssim: float,
    densify_from: int,
    densify_until: int,
    densification_interval: int,
    opacity_reset_interval: int,
    densify_grad_threshold: float,
    random_background: bool,
) -> SimpleNamespace:
    """OptimizationParams stand-in with the standard 3DGS defaults."""
    return SimpleNamespace(
        iterations=iterations,
        lambda_dssim=lambda_dssim,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30000,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.001,
        rotation_lr=0.001,
        percent_dense=0.01,
        densify_from_iter=densify_from,
        densify_until_iter=densify_until,
        densification_interval=densification_interval,
        opacity_reset_interval=opacity_reset_interval,
        densify_grad_threshold=densify_grad_threshold,
        random_background=random_background,
    )


def _make_camera_with_image(
    image: np.ndarray,
    pose: np.ndarray,
    fovx: float,
    fovy: float,
    uid: int,
    device: str,
    inv_depth: Optional[np.ndarray] = None,
    gt_alpha_mask: Optional[torch.Tensor] = None,
) -> Camera:
    """Build a Camera with the real ground-truth image attached.

    The image is the (already pre-multiplied) ``(3, H, W)`` RGB tensor,
    and ``gt_alpha_mask`` is a ``(1, H, W)`` float tensor in ``[0, 1]``
    used by the trainer to re-composite against the per-iteration
    background. ``inv_depth`` follows the convention
    ``invdepthmap = 1.0 / (depth + 1e-6)``.
    """
    R, T = c2w_to_RT(pose)
    # Camera expects (3, H, W) tensor
    img = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
    return Camera(
        colmap_id=0,
        R=R,
        T=T,
        FoVx=float(fovx),
        FoVy=float(fovy),
        image=img,
        gt_alpha_mask=gt_alpha_mask,
        image_name=f"train_{uid}",
        uid=uid,
        data_device=device,
        inv_depth=inv_depth,
    )


def _random_pcd_in_aabb(num_points: int, aabb, seed: int = 0) -> BasicPointCloud:
    rng = np.random.default_rng(seed)
    aabb = np.asarray(aabb, dtype=np.float64)
    assert aabb.shape == (3, 2)
    points = rng.uniform(low=aabb[:, 0], high=aabb[:, 1], size=(num_points, 3))
    colors = rng.uniform(low=0.0, high=1.0, size=(num_points, 3))
    normals = np.zeros_like(points)
    return BasicPointCloud(points=points, colors=colors, normals=normals)


def _pcd_from_depths(
    images: List[np.ndarray],
    poses: List[np.ndarray],
    depths: List[np.ndarray],
    fovx: float,
    fovy: float,
    H: int,
    W: int,
    pts_per_view: int = 10_000,
    seed: int = 0,
) -> BasicPointCloud:
    """Unproject depth maps into a world-space point cloud.

    For each view, build camera-space rays ``(xx*d, yy*d, d)`` with
    ``xx, yy = (px-cx)/fx, (py-cy)/fy``, transform by ``c2w``, mask to
    ``depth > 0``, and subsample to ``pts_per_view`` points per view.
    Colors come from the RGB image at the same pixel.

    Poses must be in **COLMAP / OpenCV** convention (camera looks down
    ``+z``) — this is what every other gavis API expects.
    """
    rng = np.random.default_rng(seed)
    fx = fov2focal(fovx, W)
    fy = fov2focal(fovy, H)
    cx, cy = W / 2.0, H / 2.0
    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    xx_base = ((x_grid - cx) / fx).reshape(-1)
    yy_base = ((y_grid - cy) / fy).reshape(-1)

    all_pts: List[np.ndarray] = []
    all_rgb: List[np.ndarray] = []
    for img, pose, depth in zip(images, poses, depths):
        d = np.asarray(depth, dtype=np.float64).reshape(-1)
        if d.shape[0] != xx_base.shape[0]:
            raise ValueError(
                f"depth map of shape {np.asarray(depth).shape} does not match "
                f"(H={H}, W={W}) — check that every depth matches its image."
            )
        color = np.asarray(img)[..., :3].reshape(-1, 3)
        mask = d > 0.0
        if not np.any(mask):
            continue
        xx = xx_base[mask]
        yy = yy_base[mask]
        dd = d[mask]
        pts_cam = np.stack([xx * dd, yy * dd, dd], axis=-1)
        pts_h = np.concatenate(
            [pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float64)], axis=1
        )
        c2w = np.asarray(pose, dtype=np.float64)
        pts_world = (c2w @ pts_h.T).T[:, :3]
        rgb = color[mask]

        if pts_world.shape[0] > pts_per_view:
            idx = rng.choice(pts_world.shape[0], size=pts_per_view, replace=False)
            pts_world = pts_world[idx]
            rgb = rgb[idx]
        all_pts.append(pts_world)
        all_rgb.append(rgb.astype(np.float64))

    if not all_pts:
        raise ValueError(
            "all provided depth maps were empty (no positive values); "
            "cannot initialize from depth."
        )

    pts = np.concatenate(all_pts, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    return BasicPointCloud(points=pts, colors=rgb, normals=np.zeros_like(pts))


def train_gaussian_model(
    images: List[np.ndarray],
    poses: List[np.ndarray],
    fovx: float,
    fovy: float,
    H: int,
    W: int,
    aabb,
    depths: Optional[List[np.ndarray]] = None,
    alpha_masks: Optional[List[np.ndarray]] = None,
    use_depth_init: bool = True,
    use_depth_supervision: bool = True,
    depth_init_pts_per_view: int = 10_000,
    depth_l1_weight_init: float = 1.0,
    depth_l1_weight_final: float = 0.01,
    iterations: int = 3500,
    sh_degree: int = 3,
    lambda_dssim: float = 0.2,
    densify_from: int = 500,
    densify_until: int = 15000,
    densification_interval: int = 100,
    opacity_reset_interval: int = 3000,
    densify_grad_threshold: float = 0.0002,
    random_init_pts: int = 100_000,
    white_background: bool = True,
    random_background: bool = True,
    use_isotropic_loss: bool = True,
    out_ply: Optional[str] = None,
    device: str = "cuda",
    verbose: bool = True,
) -> GaussianModel:
    """Train a 3D Gaussian Splatting model from RGB images + c2w poses.

    L1 + DSSIM + isotropic scaling penalty, periodic densify/prune and
    opacity reset, SH-degree progression every 1000 iterations.

    Parameters
    ----------
    images, poses, fovx, fovy, H, W, aabb : required
        RGB images (float32 ``[0, 1]``), c2w COLMAP/OpenCV poses, FOVs in
        radians, image size, and a scene AABB ``(3, 2)``.
    depths : list of (H, W) float arrays, optional
        Per-view metric depth maps (same length as ``images``). Required
        to actually use depth for init/supervision; if ``None`` both
        depth flags silently fall back to no-depth.
    alpha_masks : list of (H, W) float arrays, optional
        Per-view alpha masks in ``[0, 1]`` (same length as ``images``).
        When provided, the GT image is pre-multiplied by alpha and the
        mask is attached to each ``Camera`` as ``gt_alpha_mask``; the
        training loop then re-composites against the sampled background
        per iteration (``gt = gt*mask + bg*(1-mask)``).
    use_depth_init : bool, default True
        If True **and** ``depths`` is provided, initialize the gaussians
        by unprojecting depth pixels to world space (see
        :func:`_pcd_from_depths`). Otherwise init from ``random_init_pts``
        points uniformly inside ``aabb``. Setting the flag True with
        ``depths=None`` triggers a warning and falls back to random init.
    use_depth_supervision : bool, default True
        If True **and** ``depths`` is provided, attach ``inv_depth =
        1/(depth + 1e-6)`` to each training camera and add an unmasked
        L1 inverse-depth loss with an exponentially decaying weight
        (init=1.0, final=0.01 over ``iterations``). Setting True with
        ``depths=None`` triggers a warning and falls back to no depth
        supervision.
    depth_init_pts_per_view : int
        Cap on points sampled per view when using depth init.
    depth_l1_weight_init, depth_l1_weight_final : float
        Bounds of the exponentially decaying inverse-depth loss weight.
    random_background : bool, default True
        If True, sample a random RGB background each iteration (standard
        3DGS behavior). If False, always use ``white_background``.
    use_isotropic_loss : bool, default True
        If True, add ``10 * mean |scaling - mean(scaling)|`` to the loss
        — an isotropic-scaling penalty on the gaussians.
    """
    aabb = np.asarray(aabb, dtype=np.float64)
    assert aabb.shape == (3, 2)
    assert len(images) == len(poses) and len(images) > 0
    if depths is not None:
        if len(depths) != len(images):
            raise ValueError(
                f"depths length ({len(depths)}) must match images length "
                f"({len(images)})."
            )
    if alpha_masks is not None:
        if len(alpha_masks) != len(images):
            raise ValueError(
                f"alpha_masks length ({len(alpha_masks)}) must match images length "
                f"({len(images)})."
            )

    # Resolve depth flags against data availability. Both flags are
    # "use if available"; if the caller asks for depth but didn't provide
    # any, warn loudly and fall back to the no-depth path instead of
    # failing.
    have_depth = depths is not None and len(depths) > 0
    have_alpha = alpha_masks is not None and len(alpha_masks) > 0
    do_depth_init = bool(use_depth_init and have_depth)
    do_depth_sup = bool(use_depth_supervision and have_depth)
    if use_depth_init and not have_depth:
        warnings.warn(
            "train_gaussian_model: use_depth_init=True but no depths were "
            "provided; falling back to random init inside the AABB.",
            stacklevel=2,
        )
    if use_depth_supervision and not have_depth:
        warnings.warn(
            "train_gaussian_model: use_depth_supervision=True but no depths "
            "were provided; falling back to no depth supervision.",
            stacklevel=2,
        )

    scene_size = float(np.max(aabb[:, 1] - aabb[:, 0]))
    # cameras_extent: 1.1 x max distance from the mean camera center. Used
    # by ``densify_and_prune`` as the spatial scale.
    cam_centers = np.stack(
        [np.asarray(p, dtype=np.float64)[:3, 3] for p in poses], axis=0
    )
    avg_cam_center = cam_centers.mean(axis=0)
    cameras_extent = float(
        1.1 * np.linalg.norm(cam_centers - avg_cam_center, axis=1).max()
    )

    # 1. Initial gaussians: from depth unprojection if available, else
    # random points uniformly inside the AABB.
    if do_depth_init:
        if verbose:
            print(
                f"[train] initializing gaussians from {len(depths)} depth "
                f"maps (up to {depth_init_pts_per_view} pts/view)"
            )
        pcd = _pcd_from_depths(
            images=images,
            poses=poses,
            depths=depths,
            fovx=fovx,
            fovy=fovy,
            H=H,
            W=W,
            pts_per_view=depth_init_pts_per_view,
        )
    else:
        pcd = _random_pcd_in_aabb(random_init_pts, aabb, seed=0)

    gaussians = GaussianModel(sh_degree=sh_degree, object_boundaries=aabb.tolist())
    gaussians.create_from_pcd(pcd, spatial_lr_scale=scene_size * 0.5)

    # 2. Optimizer + LR schedule.
    opt = _default_opt(
        iterations=iterations,
        lambda_dssim=lambda_dssim,
        densify_from=densify_from,
        densify_until=densify_until,
        densification_interval=densification_interval,
        opacity_reset_interval=opacity_reset_interval,
        densify_grad_threshold=densify_grad_threshold,
        random_background=random_background,
    )
    gaussians.training_setup(opt)

    # 3. Build training cameras.
    #
    # Per-camera GT layout:
    #   gt_image = rgb * alpha   (pre-multiplied, black-composited)
    #   gt_alpha_mask = alpha    (shape (1, H, W))
    #   inv_depth = 1.0 / (depth + 1e-6)  (no bg masking)
    # The per-iteration training loop then re-composites gt_image against
    # the sampled background via ``gt_image * mask + bg * (1 - mask)``.
    train_cameras: List[Camera] = []
    for i, (img, pose) in enumerate(zip(images, poses)):
        gt_alpha_mask_t: Optional[torch.Tensor] = None
        img_for_cam = np.asarray(img, dtype=np.float32)

        if have_alpha:
            a = np.asarray(alpha_masks[i], dtype=np.float32)
            if a.ndim == 3 and a.shape[-1] == 1:
                a = a[..., 0]
            if a.shape != (H, W):
                raise ValueError(
                    f"alpha mask shape {a.shape} does not match (H={H}, W={W})"
                )
            # Pre-multiply RGB by alpha:
            # ``gt_image = gt_image[:3, ...] * loaded_mask``.
            img_for_cam = img_for_cam * a[..., None]
            # (1, H, W) float alpha mask.
            gt_alpha_mask_t = torch.from_numpy(a).unsqueeze(0).float().to(device)

        inv_depth_np: Optional[np.ndarray] = None
        if do_depth_sup:
            d = np.asarray(depths[i], dtype=np.float32)
            if d.shape != (H, W):
                raise ValueError(
                    f"depth map shape {d.shape} does not match (H={H}, W={W})"
                )
            inv_depth_np = (1.0 / (d + 1e-6)).astype(np.float32)

        cam = _make_camera_with_image(
            img_for_cam, pose, fovx, fovy, uid=i, device=device,
            inv_depth=inv_depth_np, gt_alpha_mask=gt_alpha_mask_t,
        )
        train_cameras.append(cam)

    # Exponentially decaying inverse-depth loss weight (1.0 → 0.01 over
    # ``iterations``). Only built when depth supervision is active.
    depth_l1_weight_fn = None
    if do_depth_sup:
        depth_l1_weight_fn = get_expon_lr_func(
            lr_init=depth_l1_weight_init,
            lr_final=depth_l1_weight_final,
            max_steps=iterations,
        )

    # 4. Background.
    bg_color = [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )

    # 5. Training loop. Runs ``range(0, iterations)`` so the SH-degree bump
    # fires at iter 0, and the reset-opacity condition excludes iter 0.
    viewpoint_stack: List[Camera] = []
    ema_loss = 0.0
    for iteration in range(0, iterations):
        gaussians.update_learning_rate(iteration)

        # Bump SH degree every 1000 iters, up to max_sh_degree (incl. iter 0).
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
        cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        bg = torch.rand((3), device=device) if opt.random_background else background
        render_pkg = render(cam, gaussians, pipe, bg)
        image = render_pkg["render"]
        viewspace_points = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = cam.original_image.to(image.device)

        # Re-composite gt against the sampled bg using the alpha mask.
        if cam.gt_alpha_mask is not None:
            mask = cam.gt_alpha_mask.to(gt_image.device)
            gt_image = gt_image * mask + bg.view(3, 1, 1) * (1.0 - mask)

        Ll1 = l1_loss(image, gt_image)
        dssim = 1.0 - ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * dssim

        if use_isotropic_loss:
            scaling = gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss = loss + 10.0 * isotropic_loss.mean()

        # Inverse-depth L1 supervision (no pixel masking, matching
        # ``Ll1depth_pure = torch.abs(invDepth - inv_depth).mean()``).
        if (
            depth_l1_weight_fn is not None
            and cam.inv_depth is not None
            and depth_l1_weight_fn(iteration) > 0
        ):
            inv_depth_render = render_pkg["depth"]
            inv_depth_gt = cam.inv_depth.to(inv_depth_render.device)
            # Ignore bg pixels where d==0 → inv_depth_gt = 1/eps ≈ 1e6
            # would dominate the mean on object-on-bg scenes.
            valid = (inv_depth_gt >= 0) & (inv_depth_gt < 1e5)
            diff = torch.abs(inv_depth_render - inv_depth_gt) * valid
            Ll1depth_pure = diff.sum() / valid.sum().clamp_min(1.0)
            loss = loss + depth_l1_weight_fn(iteration) * Ll1depth_pure

        loss.backward()

        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            if verbose and iteration % 100 == 0:
                print(
                    f"[train] iter {iteration:5d}/{iterations} "
                    f"loss={ema_loss:.6f} N={gaussians.get_xyz.shape[0]}"
                )

            # Densification block.
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gaussians.add_densification_stats(viewspace_points, visibility_filter)

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20
                        if (iteration > opt.opacity_reset_interval or iteration == 0)
                        else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        cameras_extent,
                        size_threshold,
                    )

                if iteration != 0 and (
                    iteration % opt.opacity_reset_interval == 0
                    or iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step.
            if iteration < iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

    if out_ply is not None:
        os.makedirs(os.path.dirname(out_ply), exist_ok=True)
        gaussians.save_ply(out_ply)

    return gaussians

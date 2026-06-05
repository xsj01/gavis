
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import List

import imageio.v2 as imageio
import numpy as np
import torch

from gavis.backend.renderer import render
from gavis.backend.scene.gaussian_model import GaussianModel
from gavis.data_utils import fov2focal
from gavis.pose_utils import pose_to_camera


def render_trajectory_rgb(
    model: GaussianModel,
    trajectory: List[np.ndarray],
    H: int,
    W: int,
    fovx: float,
    fovy: float,
    white_background: bool = True,
    device: str = "cuda",
) -> List[np.ndarray]:
    """Render a list of COLMAP poses with the trained 3DGS model.

    Returns a list of ``(H, W, 3)`` uint8 RGB frames. Pure inference, no
    gradient tracking.
    """
    pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )
    bg_color = [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    frames: List[np.ndarray] = []
    with torch.no_grad():
        for i, pose in enumerate(trajectory):
            cam = pose_to_camera(
                pose, H, W, fovx, fovy, name=f"traj_{i}", uid=i, device=device
            )
            out = render(cam, model, pipe, background)
            img = out["render"].clamp(0.0, 1.0).cpu().numpy()
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            frames.append(img)
    return frames


def save_gif(
    frames: List[np.ndarray],
    out_path: str,
    duration: float = 0.05,
    loop: int = 0,
) -> None:
    """Save a list of RGB/RGBA frames as an animated GIF."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimsave(out_path, frames, duration=duration, loop=loop)


def load_depths(json_path: str) -> List[np.ndarray]:
    """Load the per-frame depth maps referenced by ``transforms.json``.

    Returns the depth maps in frame order (matching the order
    :func:`gavis.data_utils.load_dataset_from_transforms_json` returns
    images / poses). Each entry's ``depth_file_path`` is resolved
    relative to the JSON file's directory.
    """
    base = os.path.dirname(os.path.abspath(json_path))
    with open(json_path, "r") as f:
        meta = json.load(f)
    return [
        np.load(os.path.join(base, fr["depth_file_path"])).astype(np.float32)
        for fr in meta["frames"]
    ]


def load_eval_poses(json_path: str) -> List[np.ndarray]:
    """Parse an eval ``transforms.json``-shaped file into a list of poses.

    Schema matches the Blender ``transforms.json`` format
    (``frames[].transform_matrix``) but no images / depth are required
    since the eval set is rendered, not supervised.
    """
    with open(json_path, "r") as f:
        meta = json.load(f)
    return [
        np.asarray(fr["transform_matrix"], dtype=np.float64)
        for fr in meta["frames"]
    ]


def aabb_from_depths(
    poses: List[np.ndarray],
    depths: List[np.ndarray],
    fovx: float,
    fovy: float,
    H: int,
    W: int,
    margin: float = 0.0,
) -> np.ndarray:
    """Estimate the scene AABB from the depth point cloud.

    Every valid (``depth > 0``) pixel of every view is unprojected to
    world space (same camera-ray math as the depth-init path in
    :func:`gavis.train_utils._pcd_from_depths`) and the AABB is the
    running min/max of that cloud. Poses are COLMAP / OpenCV c2w.

    Parameters
    ----------
    margin : float, default 0.0
        Symmetric padding to apply to each axis as a fraction of that
        axis's extent. ``margin=0.1`` expands each side by 10% of the
        depth-cloud extent, i.e.
        ``[min - 0.1 * (max - min), max + 0.1 * (max - min)]``. A small
        margin gives the visibility-field construction and UQ rendering
        some headroom around the observed scene.
    """
    fx, fy = fov2focal(fovx, W), fov2focal(fovy, H)
    cx, cy = W / 2.0, H / 2.0
    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    xx_base = ((x_grid - cx) / fx).reshape(-1)
    yy_base = ((y_grid - cy) / fy).reshape(-1)

    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for pose, depth in zip(poses, depths):
        d = np.asarray(depth, dtype=np.float64).reshape(-1)
        mask = d > 0.0
        if not np.any(mask):
            continue
        dd = d[mask]
        pts_cam = np.stack([xx_base[mask] * dd, yy_base[mask] * dd, dd], axis=-1)
        pts_h = np.concatenate(
            [pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1
        )
        pts_world = (np.asarray(pose, dtype=np.float64) @ pts_h.T).T[:, :3]
        lo = np.minimum(lo, pts_world.min(axis=0))
        hi = np.maximum(hi, pts_world.max(axis=0))

    aabb = np.stack([lo, hi], axis=1)
    if margin != 0.0:
        extent = aabb[:, 1] - aabb[:, 0]
        aabb[:, 0] -= margin * extent
        aabb[:, 1] += margin * extent
    return aabb

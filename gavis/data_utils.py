"""Loader for Blender-style transforms.json datasets."""

import json
import math
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

from gavis.pose_utils import opengl_to_colmap_c2w


def load_dataset_from_transforms_json(
    json_path: str,
    white_background: bool = True,
    pose_convention: str = "colmap",
    return_alpha_masks: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], float, float, int, int]:
    """Parse a NeRF/Blender-style ``transforms.json`` file.

    Parameters
    ----------
    json_path : str
        Path to the ``transforms.json`` file.
    white_background : bool
        If the source image is RGBA, alpha-composite onto white (True) or
        black (False).
    pose_convention : {"colmap", "opengl"}
        Coordinate convention of ``transform_matrix`` in the JSON file.

        - ``"colmap"`` (default): poses are already in COLMAP / OpenCV
          convention (camera looks down +z). This matches files produced
          by the project's Blender renderer (which applies the
          opengl-to-colmap flip before writing) and is consumed verbatim
          downstream.
        - ``"opengl"``: poses are in OpenGL / Blender convention (camera
          looks down -z). This matches **vanilla NeRF synthetic** datasets
          (lego, ficus, hotdog from the original NeRF release). The flip
          ``c2w[:3, 1:3] *= -1`` is applied at load time so the returned
          poses are in COLMAP convention, ready for ``GAVIS.load_model``
          and the 3DGS rasterizer.

    Returns
    -------
    images : list of (H, W, 3) float32 arrays in [0, 1]
        RGB images, alpha-composited onto a white or black background if
        the source image is RGBA.
    poses : list of (4, 4) float64 numpy arrays
        Camera-to-world transforms, **always returned in COLMAP
        convention** regardless of the source convention.
    fovx, fovy : float
        Horizontal / vertical field of view in radians.
    H, W : int
        Image height and width.
    """
    if pose_convention not in ("colmap", "opengl"):
        raise ValueError(
            f"pose_convention must be 'colmap' or 'opengl', got {pose_convention!r}"
        )
    json_path = os.path.abspath(json_path)
    base = os.path.dirname(json_path)
    with open(json_path, "r") as f:
        meta = json.load(f)

    fovx = float(meta["camera_angle_x"])
    fovy = float(meta.get("camera_angle_y", fovx))

    # H, W: prefer the explicit fields if present, otherwise read from the
    # first image.
    H = int(meta["h"]) if "h" in meta else None
    W = int(meta["w"]) if "w" in meta else None

    images: List[np.ndarray] = []
    poses: List[np.ndarray] = []
    alpha_masks: List[np.ndarray] = []
    for frame in meta["frames"]:
        rel_path = frame["file_path"]
        # Blender datasets sometimes omit the extension; assume .png if so.
        img_path = os.path.join(base, rel_path)
        if not os.path.isfile(img_path):
            for ext in (".png", ".jpg", ".jpeg"):
                if os.path.isfile(img_path + ext):
                    img_path = img_path + ext
                    break
        pil_img = Image.open(img_path)
        if pil_img.mode == "RGBA":
            arr = np.array(pil_img, dtype=np.float32) / 255.0
            rgb, alpha = arr[..., :3], arr[..., 3:4]
            alpha_mask = alpha[..., 0].astype(np.float32)
            bg = np.ones_like(rgb) if white_background else np.zeros_like(rgb)
            arr = rgb * alpha + bg * (1.0 - alpha)
        else:
            arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
            alpha_mask = np.ones(arr.shape[:2], dtype=np.float32)

        if H is None:
            H = arr.shape[0]
        if W is None:
            W = arr.shape[1]

        images.append(arr.astype(np.float32))
        alpha_masks.append(alpha_mask)
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        if pose_convention == "opengl":
            c2w = opengl_to_colmap_c2w(c2w)
        poses.append(c2w)

    assert H is not None and W is not None
    if return_alpha_masks:
        return images, poses, fovx, fovy, H, W, alpha_masks
    return images, poses, fovx, fovy, H, W


def fov2focal(fov: float, pixels: int) -> float:
    return pixels / (2.0 * math.tan(fov / 2.0))

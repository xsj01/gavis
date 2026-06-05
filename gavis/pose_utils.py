"""Pose / camera helpers used by gavis.

Coordinate convention
---------------------
**Every public function in this module that consumes a c2w pose expects
COLMAP / OpenCV convention** (camera looks down its local ``+z``, ``+x``
right, ``+y`` down). This matches:

Functions that *generate* poses (``pose_point_to``, ``circular_trajectory``)
take a ``convention`` flag and default to ``"colmap"`` so they slot
straight into the rest of the pipeline. Use ``convention="opengl"`` only
if you specifically need a Blender-convention pose.

The 3DGS CUDA rasterizer additionally expects the rotation matrix in
column-major (GLM) order, so ``c2w_to_RT`` transposes ``w2c[:3, :3]``
before returning. Do not change this without changing the rasterizer
side.
"""

import numpy as np
import torch

from gavis.backend.scene.cameras import Camera


def c2w_to_RT(pose):
    """Convert a 4x4 camera-to-world pose to ``(R, T)`` for ``Camera()``.

    Expects ``pose`` in **COLMAP / OpenCV** convention (camera looks down
    local ``+z``). Feeding an OpenGL/Blender pose here will silently
    produce a camera that points away from the scene — convert it first
    with :func:`opengl_to_colmap_c2w`.

    The 3DGS CUDA rasterizer reads the rotation matrix in column-major
    (GLM) order, so we transpose ``w2c[:3, :3]`` here. **Do not change**
    this transpose without also changing the rasterizer side.
    """
    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()
    c2w = np.asarray(pose, dtype=np.float64)
    assert c2w.shape == (4, 4), f"expected (4,4) pose, got {c2w.shape}"
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3, :3])
    T = w2c[:3, 3]
    return R, T


def pose_to_camera(pose, H, W, fovx, fovy, name="query", uid=0, device="cuda"):
    """Build a 3DGS ``Camera`` object from a c2w pose + intrinsics.

    ``pose`` must be in **COLMAP / OpenCV** convention (see
    :func:`c2w_to_RT`).
    """
    R, T = c2w_to_RT(pose)
    gt_image = torch.zeros((3, H, W), dtype=torch.float32, device=device)
    return Camera(
        colmap_id=0,
        R=R,
        T=T,
        FoVx=float(fovx),
        FoVy=float(fovy),
        image=gt_image,
        gt_alpha_mask=None,
        image_name=name,
        uid=uid,
        data_device=device,
    )


def poses_to_cameras(poses, H, W, fovx, fovy, device="cuda"):
    """Vectorized helper: list of c2w poses → list of ``Camera`` objects.

    All poses must be in **COLMAP / OpenCV** convention.
    """
    return [
        pose_to_camera(p, H, W, fovx, fovy, name=f"train_{i}", uid=i, device=device)
        for i, p in enumerate(poses)
    ]


def opengl_to_colmap_c2w(pose):
    """Flip a c2w from Blender/OpenGL convention to COLMAP/OpenCV convention.

    Negates the second and third columns of the rotation; the translation
    is unchanged. Calling this twice is the identity.
    """
    pose = np.array(pose, dtype=np.float64, copy=True)
    pose[:3, 1:3] *= -1.0
    return pose


def pose_point_to(loc, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0), convention="colmap"):
    """Look-at pose builder.

    Builds a 4x4 c2w pose for a camera at ``loc`` looking at ``target``,
    with the world up vector ``up``.

    Parameters
    ----------
    loc : array-like (3,)
        Camera position in world coordinates.
    target : array-like (3,)
        Point the camera looks at.
    up : array-like (3,)
        World up vector.
    convention : {"colmap", "opengl"}
        Output coordinate convention for the returned c2w pose.

        - ``"colmap"`` (default): COLMAP / OpenCV — camera looks down its
          local ``+z``. This is the convention every downstream consumer
          in this package (``c2w_to_RT``, ``pose_to_camera``,
          ``GAVIS.load_model``, ``GAVIS.render_uncertainty_map``)
          expects, so this is almost always what you want.
        - ``"opengl"``: Blender / OpenGL — camera looks down its local
          ``-z``. This is the *raw* output of the underlying look-at math
          (the third column of the rotation, ``ez``, points from
          ``target`` to ``loc``, i.e. away from the scene). Choose this
          only if you need to feed the pose to a Blender-convention
          consumer; the project's Blender renderer flips this to COLMAP
          before writing to ``transforms.json``, which is why the on-disk
          datasets are COLMAP.

    Returns
    -------
    pose : np.ndarray (4, 4)
        c2w pose in the requested convention.
    """
    if convention not in ("colmap", "opengl"):
        raise ValueError(
            f"convention must be 'colmap' or 'opengl', got {convention!r}"
        )
    x0 = np.array(target, dtype=np.float64)
    p = np.array(loc, dtype=np.float64)
    up_v = np.array(up, dtype=np.float64)
    up_v = up_v / np.linalg.norm(up_v)
    ez = (p - x0) / np.linalg.norm(p - x0)
    ex = np.cross(up_v, ez)
    if np.linalg.norm(ex) < 1e-7:
        up_v = up_v + np.array([0.1, 0.0, 0.0])
        up_v = up_v / np.linalg.norm(up_v)
        ex = np.cross(up_v, ez)
    ex = ex / np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    ey = ey / np.linalg.norm(ey)
    rot = np.array([ex, ey, ez]).T
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = p
    if convention == "colmap":
        pose = opengl_to_colmap_c2w(pose)
    return pose


def circular_trajectory(
    center,
    radius,
    z=None,
    N=60,
    up=(0.0, 0.0, 1.0),
    convention="colmap",
    elev_deg=None,
):
    """Generate ``N`` c2w poses on a circle around ``center``.

    Two parameterizations are supported, via the mutually-exclusive
    ``z`` and ``elev_deg`` arguments:

    - **Cylinder mode** (``z`` given): cameras sit on a horizontal
      circle of XY-radius ``radius`` at absolute altitude ``z``. The
      camera-to-center distance is ``sqrt(radius**2 + (z - center[2])**2)``,
      which is *not* necessarily equal to ``radius`` — use this mode
      only when you want fine control over the altitude independent of
      the horizontal radius.
    - **Sphere mode** (``elev_deg`` given): cameras sit on a spherical
      shell of radius ``radius`` around ``center`` at elevation
      ``elev_deg`` (degrees above the horizontal XY plane). The
      camera-to-center distance is exactly ``radius`` for every frame,
      which is the right mode when you want the orbit to match a known
      training shell (e.g. the shell ``hemisphere_poses`` was built
      on). Internally: ``horiz_r = radius * cos(elev)``,
      ``altitude = center[2] + radius * sin(elev)``.

    Each camera looks toward ``center``, so with ``convention="colmap"``
    the target projects to the rendered image center (see
    :func:`pose_point_to`).

    Parameters
    ----------
    center : array-like (3,)
        Center of the circle (also the look-at target).
    radius : float
        Cylinder-mode: horizontal circle radius in the XY plane.
        Sphere-mode: spherical shell radius.
    z : float, optional
        Cylinder-mode camera altitude. Must be ``None`` when
        ``elev_deg`` is given.
    N : int
        Number of frames. Default 60.
    up : array-like (3,)
        World up vector. Default +Z.
    convention : {"colmap", "opengl"}
        Output coordinate convention. Default ``"colmap"``.
    elev_deg : float, optional
        Sphere-mode elevation angle above the XY plane, in degrees.
        Must be ``None`` when ``z`` is given.

    Returns
    -------
    poses : list of np.ndarray (4, 4)
        c2w poses in the requested convention.
    """
    if (z is None) == (elev_deg is None):
        raise ValueError(
            "circular_trajectory requires exactly one of `z` or `elev_deg`"
        )
    center = np.asarray(center, dtype=np.float64)
    if elev_deg is not None:
        elev = float(np.deg2rad(elev_deg))
        horiz_r = float(radius) * float(np.cos(elev))
        altitude = float(center[2] + float(radius) * float(np.sin(elev)))
    else:
        horiz_r = float(radius)
        altitude = float(z)
    poses = []
    for i in range(N):
        theta = 2.0 * np.pi * i / N
        loc = np.array(
            [
                center[0] + horiz_r * np.cos(theta),
                center[1] + horiz_r * np.sin(theta),
                altitude,
            ]
        )
        poses.append(
            pose_point_to(loc=loc, target=center, up=list(up), convention=convention)
        )
    return poses


def normalize_training_poses(training_pose):
    """Coerce a variety of pose containers into ``list[np.ndarray (4,4)]``."""
    if isinstance(training_pose, torch.Tensor):
        training_pose = training_pose.detach().cpu().numpy()
    if isinstance(training_pose, np.ndarray):
        if training_pose.ndim == 3:
            return [np.asarray(p, dtype=np.float64) for p in training_pose]
        if training_pose.ndim == 2 and training_pose.shape == (4, 4):
            return [np.asarray(training_pose, dtype=np.float64)]
        raise ValueError(f"unexpected ndarray shape for training poses: {training_pose.shape}")
    if isinstance(training_pose, (list, tuple)):
        out = []
        for p in training_pose:
            if isinstance(p, torch.Tensor):
                p = p.detach().cpu().numpy()
            p = np.asarray(p, dtype=np.float64)
            assert p.shape == (4, 4), f"expected each pose to be (4,4), got {p.shape}"
            out.append(p)
        return out
    raise TypeError(f"unsupported training_pose type: {type(training_pose)}")

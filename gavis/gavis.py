"""GAVIS uncertainty quantification.

Implements the visibility-guided splatting uncertainty estimator on top of a
trained 3DGS model, exposed as a standalone library without any trainer,
planning, or active-mapping machinery.
"""

from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from .base import BaseUQ
from .config_utils import load_default_cfg, merge_cfg
from .pose_utils import pose_to_camera, poses_to_cameras
from .backend.visibility_manager import VisibilityManager
from .backend.renderer import render_entropy


class GAVIS(BaseUQ):
    """Visibility-aware uncertainty estimator.

    Usage::

        cfg = OmegaConf.create({"H": 800, "W": 800, "fovx": ..., "fovy": ...,
                                "aabb": [[xmin,xmax],[ymin,ymax],[zmin,zmax]]})
        gavis = GAVIS(cfg)
        gavis.load_model(trained_gaussian_model, training_poses)
        umap = gavis.render_uncertainty_map(query_pose)   # → (H, W) tensor
    """

    def __init__(self, cfg):
        merged = merge_cfg(load_default_cfg("gavis_default"), cfg)
        super().__init__(merged)

        self._visibility_handler = None
        # The CUDA renderer accesses pipe.compute_cov3D_python and pipe.debug.
        # PipelineParams (the original argparse object) only carries these flags;
        # we duck-type with a SimpleNamespace.
        self._pipeline_params = SimpleNamespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False,
        )

    # ------------------------------------------------------------------ load
    def load_model(self, model, training_pose):
        """Build the visibility handler and run preprocessing.

        Note: ``VisibilityManager.set_pc`` deepcopies the input model, so
        the caller's ``model`` reference (also stored as ``self._model_raw``)
        remains untouched even after the augmented copy gets fixed gaussians
        prepended in :meth:`VisibilityManager.preprocess`.
        """
        super().load_model(model, training_pose)

        vis_cfg = self.cfg
        self._visibility_handler = VisibilityManager(
            pc=None,
            pipe=self._pipeline_params,
            use_angular_vis_learned=vis_cfg.vis.grid,
            kappa=vis_cfg.vis.kappa,
            num_fixed_gaussians=vis_cfg.vis.num_fixed_gaussians,
            fixed_gaussians=vis_cfg.vis.fixed_gaussian,
            surface_fixed=vis_cfg.vis.fixed_gaussian_surface,
            use_angular_vis_fixed=vis_cfg.vis.fixed_gaussian_use_angular_vis,
            background_kappa=vis_cfg.vis.fixed_gaussian_kappa,
            bounding_box=np.asarray(vis_cfg.aabb),
            cfg=vis_cfg,
        )
        self._visibility_handler.set_pc(model)

        camera_views = poses_to_cameras(
            self._training_poses,
            self.H,
            self.W,
            self.fovx,
            self.fovy,
            device=str(self.device),
        )
        self._visibility_handler.preprocess(camera_views, self.bg_color)

        # After preprocess, handler.pc may have fixed gaussians prepended.
        self._model = self._visibility_handler.pc

    # ------------------------------------------------------------------ render
    @torch.no_grad()
    def render_uncertainty_map(self, query_pose):
        assert self._visibility_handler is not None, (
            "GAVIS.load_model() must be called before render_uncertainty_map()"
        )

        view = pose_to_camera(
            query_pose,
            self.H,
            self.W,
            self.fovx,
            self.fovy,
            name="query",
            device=str(self.device),
        )

        # Zero out cfg.bn.d0 while rendering uncertainty maps (disables the
        # depth-decay factor).
        d0_orig = self.cfg.bn.d0
        try:
            self.cfg.bn.d0 = 0

            # Per-gaussian rgb variance. var_mode='none' uses the constant
            # 1e-4 baseline; FisherRF (TODO) would override with Fisher diag.
            raw_xyz = self._model_raw._xyz
            rgb_var = (
                torch.ones_like(raw_xyz, dtype=torch.float32, device=self.device)
                * 1e-4
            )
            if self.cfg.vis.fixed_gaussian:
                n_fixed = self._visibility_handler.num_fixed_gaussians
                rgb_var = torch.cat(
                    [
                        torch.ones(
                            n_fixed,
                            3,
                            dtype=torch.float32,
                            device=self.device,
                        )
                        * 1e-4,
                        rgb_var,
                    ],
                    dim=0,
                )

            rendering = render_entropy(
                view,
                self._visibility_handler.pc,
                rgb_var,
                self._pipeline_params,
                self.bg_color,
                cfg=self.cfg,
                visibility_handler=self._visibility_handler,
            )
        finally:
            self.cfg.bn.d0 = d0_orig

        return rendering.view(self.H, self.W).detach().cpu()

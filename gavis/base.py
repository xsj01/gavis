"""Base interface for uncertainty quantification on a trained 3DGS model."""

import numpy as np
import torch
from omegaconf import DictConfig

from .pose_utils import normalize_training_poses


class BaseUQ:
    """Generic UQ interface.

    Subclasses (``GAVIS``, ``FisherRF``, ...) implement the algorithm-specific
    behavior in :meth:`load_model` and :meth:`render_uncertainty_map`. The base
    class is responsible for storing the configuration, validating intrinsics,
    and normalizing the training pose container.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.H = int(cfg.H)
        self.W = int(cfg.W)
        self.fovx = float(cfg.fovx)
        self.fovy = float(cfg.fovy)
        self.aabb = np.asarray(cfg.aabb, dtype=np.float64)
        assert self.aabb.shape == (3, 2), (
            f"cfg.aabb must be (3, 2), got {self.aabb.shape}"
        )
        assert self.H > 0 and self.W > 0
        assert 0.0 < self.fovx < np.pi
        assert 0.0 < self.fovy < np.pi

        device_str = cfg.get("device", "cuda")
        self.device = torch.device(device_str)
        self.bg_color = torch.tensor(
            cfg.get("bg_color", [0.0, 0.0, 0.0]),
            dtype=torch.float32,
            device=self.device,
        )

        self._model_raw = None
        self._model = None
        self._training_poses = None

    def load_model(self, model, training_pose):
        """Store the trained 3DGS model and the training poses.

        ``training_pose`` may be a list of ``(4,4)`` numpy arrays, a
        ``(T,4,4)`` numpy array, or a ``(T,4,4)`` torch tensor. Subclasses
        should call ``super().load_model(...)`` first to populate
        ``self._model_raw`` and ``self._training_poses``.
        """
        self._model_raw = model
        self._model = model
        self._training_poses = normalize_training_poses(training_pose)

    def render_uncertainty_map(self, query_pose):
        """Return an ``(H, W)`` uncertainty tensor at ``query_pose`` (c2w)."""
        raise NotImplementedError

    @property
    def model_raw(self):
        """The original (un-augmented) 3DGS model handed in by the caller."""
        return self._model_raw

    @property
    def model(self):
        """The 3DGS model used for UQ rendering (may include fixed gaussians)."""
        return self._model

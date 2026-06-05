"""OmegaConf helpers for loading and merging the gavis default config."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

_CONFIG_DIR = Path(__file__).parent / "configs"


def load_default_cfg(name: str = "gavis_default") -> DictConfig:
    """Load a yaml config shipped under ``gavis/configs/<name>.yaml``."""
    path = _CONFIG_DIR / f"{name}.yaml"
    return OmegaConf.load(str(path))


def merge_cfg(default_cfg: DictConfig, user_cfg) -> DictConfig:
    """Merge a user-provided config (dict or DictConfig) onto a default."""
    if not isinstance(user_cfg, DictConfig):
        user_cfg = OmegaConf.create(user_cfg)
    return OmegaConf.merge(default_cfg, user_cfg)

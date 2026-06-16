"""Training device resolution with automatic GPU fallback."""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)

DevicePreference = Literal["auto", "cpu", "cuda"]
ResolvedTorchDevice = Literal["cpu", "cuda"]

__all__ = [
    "DevicePreference",
    "ResolvedTorchDevice",
    "cuda_available",
    "lightgbm_gpu_available",
    "resolve_lightgbm_device",
    "resolve_training_device",
]

_lightgbm_gpu_cache: bool | None = None


def cuda_available() -> bool:
    """Return whether PyTorch sees an available CUDA device."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def resolve_training_device(
    preference: DevicePreference = "auto",
) -> ResolvedTorchDevice:
    """Resolve a torch device string, falling back to CPU when needed."""
    if preference == "cpu":
        return "cpu"
    if cuda_available():
        return "cuda"
    if preference == "cuda":
        logger.warning("CUDA requested but unavailable; using CPU for training")
    return "cpu"


def lightgbm_gpu_available() -> bool:
    """Probe whether the installed LightGBM build can train on GPU."""
    global _lightgbm_gpu_cache
    if _lightgbm_gpu_cache is not None:
        return _lightgbm_gpu_cache
    if not cuda_available():
        _lightgbm_gpu_cache = False
        return False
    try:
        import lightgbm as lgb
        import numpy as np

        features = np.random.default_rng(0).random((16, 4), dtype=np.float64)
        labels = np.random.default_rng(1).random(16, dtype=np.float64)
        regressor = lgb.LGBMRegressor(
            device="gpu",
            n_estimators=1,
            num_leaves=4,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        regressor.fit(features, labels)
        _lightgbm_gpu_cache = True
    except Exception:  # noqa: BLE001 — probe must not break training setup
        _lightgbm_gpu_cache = False
    return _lightgbm_gpu_cache


def resolve_lightgbm_device(preference: DevicePreference = "auto") -> str:
    """Resolve LightGBM ``device`` param with automatic CPU fallback."""
    if resolve_training_device(preference) == "cpu":
        return "cpu"
    if lightgbm_gpu_available():
        return "gpu"
    logger.info("LightGBM GPU unavailable; using CPU for training")
    return "cpu"

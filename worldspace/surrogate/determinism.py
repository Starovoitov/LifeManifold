"""Deterministic training and inference settings for surrogate backends."""

from __future__ import annotations

DEFAULT_ENSEMBLE_SIZE = 4
DEFAULT_MLP_HIDDEN_DIMS: tuple[int, ...] = (32, 32)
LEGACY_MLP_HIDDEN_DIMS: tuple[int, ...] = (64, 64)
DEFAULT_RANDOM_STATE = 42

__all__ = [
    "DEFAULT_ENSEMBLE_SIZE",
    "DEFAULT_MLP_HIDDEN_DIMS",
    "LEGACY_MLP_HIDDEN_DIMS",
    "DEFAULT_RANDOM_STATE",
    "apply_mlp_determinism",
    "lightgbm_deterministic_params",
    "member_random_state",
]


def lightgbm_deterministic_params() -> dict[str, object]:
    """Keyword arguments for deterministic LightGBM estimators."""
    return {
        "random_state": DEFAULT_RANDOM_STATE,
        "deterministic": True,
        "force_col_wise": True,
    }


def member_random_state(member_index: int) -> int:
    """Derive a stable per-ensemble-member seed."""
    return DEFAULT_RANDOM_STATE + int(member_index)


def apply_mlp_determinism() -> None:
    """Configure PyTorch for reproducible MLP training and inference."""
    import torch

    torch.manual_seed(DEFAULT_RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(DEFAULT_RANDOM_STATE)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

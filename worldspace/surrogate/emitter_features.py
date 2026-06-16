"""Train-time emitter regime one-hot features from buffer rows."""

from __future__ import annotations

import numpy as np

EMITTER_ONEHOT_DIM = 3
EMITTER_ONEHOT_NAMES: tuple[str, ...] = (
    "emitter_random",
    "emitter_genetic",
    "emitter_llm",
)

__all__ = [
    "EMITTER_ONEHOT_DIM",
    "EMITTER_ONEHOT_NAMES",
    "augment_features_with_emitter",
    "emitter_onehot_vector",
    "training_feature_dim",
]


def emitter_onehot_vector(emitter_type: str) -> np.ndarray:
    """Return a length-3 one-hot vector for known MAP-Elites emitters."""
    vector = np.zeros(EMITTER_ONEHOT_DIM, dtype=np.float64)
    normalized = str(emitter_type or "").strip().lower()
    if normalized == "random":
        vector[0] = 1.0
    elif normalized == "genetic":
        vector[1] = 1.0
    elif normalized == "llm":
        vector[2] = 1.0
    return vector


def augment_features_with_emitter(
    feature_matrix: np.ndarray,
    emitter_types: np.ndarray,
) -> np.ndarray:
    """Concatenate emitter one-hot columns onto a feature matrix."""
    base = np.asarray(feature_matrix, dtype=float)
    if base.ndim != 2:
        msg = "feature_matrix must be two-dimensional"
        raise ValueError(msg)
    labels = np.asarray(emitter_types).reshape(-1)
    if int(labels.shape[0]) != int(base.shape[0]):
        msg = "emitter_types length must match feature row count"
        raise ValueError(msg)
    onehot = np.stack(
        [emitter_onehot_vector(str(label)) for label in labels],
        axis=0,
    )
    return np.concatenate([base, onehot], axis=1)


def training_feature_dim(*, base_dim: int, emitter_onehot: bool) -> int:
    """Return model input width for one training configuration."""
    width = int(base_dim)
    if emitter_onehot:
        width += EMITTER_ONEHOT_DIM
    return width

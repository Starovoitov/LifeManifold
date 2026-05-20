"""Surrogate component model API for MVP Strategy A."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from worldspace.surrogate.determinism import (
    DEFAULT_ENSEMBLE_SIZE,
    DEFAULT_RANDOM_STATE,
)

TARGET_KEYS: tuple[str, ...] = (
    "stability",
    "diversity",
    "oscillation_score",
    "topology_interface_index",
    "topology_window_heterogeneity",
    "final_density",
    "early_extinction_prob",
)

__all__ = [
    "TARGET_KEYS",
    "SurrogateModel",
]


@dataclass
class SurrogateModel:
    """Deterministic component regressor for Strategy A outputs."""

    model_type: str = "lightgbm"
    random_state: int = DEFAULT_RANDOM_STATE
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE
    _component_means: dict[str, float] = field(default_factory=dict)

    def fit(
        self,
        feature_matrix: np.ndarray,
        targets: dict[str, np.ndarray],
    ) -> None:
        """Fit deterministic MVP baseline from component targets.

        LightGBM/MLP backends should call ``lightgbm_deterministic_params`` or
        ``apply_mlp_determinism`` from ``determinism.py`` when implemented.
        """
        _ = feature_matrix
        self._component_means = {
            key: float(np.mean(_as_float_array(targets, key))) for key in TARGET_KEYS
        }

    def set_component_defaults(self, value: float) -> None:
        """Set all target means to one deterministic value."""
        self._component_means = {key: float(value) for key in TARGET_KEYS}

    def predict_components(self, features: np.ndarray) -> dict[str, float]:
        """Predict all Strategy A target components from extracted features."""
        _ = features
        if not self._component_means:
            self.set_component_defaults(0.5)
        return dict(self._component_means)

    def predict_uncertainty(self, features: np.ndarray) -> float:
        """Return deterministic MVP uncertainty proxy (ensemble std placeholder)."""
        _ = features
        return 0.0


def _as_float_array(targets: dict[str, np.ndarray], key: str) -> np.ndarray:
    values = targets.get(key)
    if values is None:
        msg = f"Missing target array for key={key!r}"
        raise ValueError(msg)
    return np.asarray(values, dtype=float)

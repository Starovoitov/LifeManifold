"""Surrogate component model API for MVP Strategy A."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any

import numpy as np

from worldspace.surrogate.determinism import (
    DEFAULT_ENSEMBLE_SIZE,
    DEFAULT_RANDOM_STATE,
    lightgbm_deterministic_params,
    member_random_state,
)
from worldspace.surrogate.types import SurrogatePrediction
from worldspace.surrogate.utils import compute_fitness_from_prediction

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
    _ensemble: dict[str, list[Any]] = field(default_factory=dict)
    _uses_lightgbm: bool = False

    def fit(
        self,
        feature_matrix: np.ndarray,
        targets: dict[str, np.ndarray],
    ) -> None:
        """Fit component regressors; LightGBM ensemble when available."""
        self._component_means = {
            key: float(np.mean(_as_float_array(targets, key))) for key in TARGET_KEYS
        }
        self._ensemble = {}
        self._uses_lightgbm = False
        if self.model_type != "lightgbm" or find_spec("lightgbm") is None:
            return
        self._fit_lightgbm_ensemble(feature_matrix, targets)

    def set_component_defaults(self, value: float) -> None:
        """Set all target means to one deterministic value."""
        self._component_means = {key: float(value) for key in TARGET_KEYS}
        self._ensemble = {}
        self._uses_lightgbm = False

    def predict_components(self, features: np.ndarray) -> dict[str, float]:
        """Predict all Strategy A target components from extracted features."""
        vector = np.asarray(features, dtype=float).reshape(-1)
        if self._uses_lightgbm:
            return {
                key: float(
                    np.mean(
                        [
                            float(estimator.predict(vector.reshape(1, -1))[0])
                            for estimator in self._ensemble[key]
                        ]
                    )
                )
                for key in TARGET_KEYS
            }
        if not self._component_means:
            self.set_component_defaults(0.5)
        return dict(self._component_means)

    def predict_uncertainty(self, features: np.ndarray) -> float:
        """Return ensemble standard deviation of predicted fitness."""
        vector = np.asarray(features, dtype=float).reshape(-1)
        if not self._uses_lightgbm:
            return 0.0
        fitness_values: list[float] = []
        for member_index in range(self.ensemble_size):
            components = {
                key: float(
                    self._ensemble[key][member_index].predict(vector.reshape(1, -1))[0]
                )
                for key in TARGET_KEYS
            }
            prediction = SurrogatePrediction(
                components=components,
                measures={
                    "stability": components["stability"],
                    "diversity": components["diversity"],
                },
                fitness=0.0,
                uncertainty=0.0,
            )
            fitness_values.append(compute_fitness_from_prediction(prediction))
        if len(fitness_values) < 2:
            return 0.0
        return float(np.std(np.asarray(fitness_values, dtype=float), ddof=0))

    def _fit_lightgbm_ensemble(
        self,
        feature_matrix: np.ndarray,
        targets: dict[str, np.ndarray],
    ) -> None:
        import lightgbm as lgb

        x_train = np.asarray(feature_matrix, dtype=float)
        base_params = {
            **lightgbm_deterministic_params(),
            "n_estimators": 64,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "verbosity": -1,
        }
        self._ensemble = {}
        for key in TARGET_KEYS:
            y_train = _as_float_array(targets, key)
            estimators: list[Any] = []
            for member_index in range(self.ensemble_size):
                member_params = dict(base_params)
                member_params["random_state"] = member_random_state(member_index)
                regressor = lgb.LGBMRegressor(**member_params)
                regressor.fit(x_train, y_train)
                estimators.append(regressor)
            self._ensemble[key] = estimators
        self._uses_lightgbm = True


def _as_float_array(targets: dict[str, np.ndarray], key: str) -> np.ndarray:
    values = targets.get(key)
    if values is None:
        msg = f"Missing target array for key={key!r}"
        raise ValueError(msg)
    return np.asarray(values, dtype=float)

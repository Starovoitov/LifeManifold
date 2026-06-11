"""Surrogate component model API for MVP Strategy A."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any

import numpy as np

from worldspace.surrogate.feature_extractor import FEATURE_NAMES
from worldspace.surrogate.determinism import (
    DEFAULT_ENSEMBLE_SIZE,
    DEFAULT_RANDOM_STATE,
    apply_mlp_determinism,
    lightgbm_deterministic_params,
    member_random_state,
)
from worldspace.surrogate.types import SurrogatePrediction
from worldspace.surrogate.utils import compute_fitness_from_prediction

_CONSISTENCY_REFINE_KEYS: tuple[str, ...] = ("stability", "diversity")

TARGET_KEYS: tuple[str, ...] = (
    "stability",
    "diversity",
    "oscillation_score",
    "topology_interface_index",
    "topology_window_heterogeneity",
    "final_density",
    "early_extinction_prob",
)
FITNESS_TARGET_KEY = "fitness"
MIN_FITNESS_HEAD_SAMPLES = 10

__all__ = [
    "EXPECTED_FEATURE_DIM",
    "FITNESS_TARGET_KEY",
    "MIN_FITNESS_HEAD_SAMPLES",
    "TARGET_KEYS",
    "SurrogateModel",
    "checkpoint_feature_dim",
    "checkpoint_matches_extractor",
    "consistency_mae_on_rows",
]

EXPECTED_FEATURE_DIM = len(FEATURE_NAMES)
_DEFAULT_MLP_HIDDEN_DIMS: tuple[int, ...] = (64, 64)


@dataclass
class SurrogateModel:
    """Deterministic component regressor for Strategy A outputs."""

    model_type: str = "lightgbm"
    random_state: int = DEFAULT_RANDOM_STATE
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE
    _component_means: dict[str, float] = field(default_factory=dict)
    _ensemble: dict[str, list[Any]] = field(default_factory=dict)
    _fitness_ensemble: list[Any] = field(default_factory=list)
    _uses_lightgbm: bool = False
    _uses_mlp: bool = False
    _mlp_members: list[dict[str, Any]] = field(default_factory=list)
    _mlp_hidden_dims: tuple[int, ...] = _DEFAULT_MLP_HIDDEN_DIMS
    _fitness_loss_weight: float = 1.0
    _has_fitness_head: bool = False

    def __setstate__(self, state: object) -> None:
        """Restore pickle state and backfill fields from pre-fitness-head checkpoints."""
        if not isinstance(state, dict):
            msg = f"unexpected SurrogateModel pickle state: {type(state)!r}"
            raise TypeError(msg)
        self.__dict__.update(state)
        self.ensure_legacy_checkpoint_fields()

    def ensure_legacy_checkpoint_fields(self) -> None:
        """Initialize fields missing from older pickled checkpoints."""
        if "_fitness_ensemble" not in self.__dict__:
            self.__dict__["_fitness_ensemble"] = []
        if "_uses_mlp" not in self.__dict__:
            self.__dict__["_uses_mlp"] = False
        if "_mlp_members" not in self.__dict__:
            self.__dict__["_mlp_members"] = []
        if "_mlp_hidden_dims" not in self.__dict__:
            self.__dict__["_mlp_hidden_dims"] = _DEFAULT_MLP_HIDDEN_DIMS
        if "_fitness_loss_weight" not in self.__dict__:
            self.__dict__["_fitness_loss_weight"] = 1.0
        if "_has_fitness_head" not in self.__dict__:
            self.__dict__["_has_fitness_head"] = False
        elif self.__dict__["_has_fitness_head"] and not self._fitness_head_is_trained():
            self.__dict__["_has_fitness_head"] = False

    def _fitness_head_is_trained(self) -> bool:
        if self.__dict__.get("_uses_mlp", False):
            return bool(self.__dict__.get("_mlp_members"))
        return bool(self.__dict__.get("_fitness_ensemble"))

    def fit(
        self,
        feature_matrix: np.ndarray,
        targets: dict[str, np.ndarray],
        *,
        fitness_loss_weight: float = 1.0,
        val_features: np.ndarray | None = None,
        val_targets: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Fit component regressors; LightGBM or MLP ensemble when available."""
        self._component_means = {
            key: float(np.mean(_as_float_array(targets, key))) for key in TARGET_KEYS
        }
        self._ensemble = {}
        self._fitness_ensemble = []
        self._uses_lightgbm = False
        self._uses_mlp = False
        self._mlp_members = []
        self._has_fitness_head = False
        self._fitness_loss_weight = float(fitness_loss_weight)

        if self.model_type == "mlp":
            if find_spec("torch") is None:
                return
            self._fit_mlp_ensemble(
                feature_matrix,
                targets,
                val_features=val_features,
                val_targets=val_targets,
            )
            return

        if self.model_type != "lightgbm" or find_spec("lightgbm") is None:
            return
        self._fit_lightgbm_ensemble(feature_matrix, targets)
        fitness = targets.get(FITNESS_TARGET_KEY)
        if fitness is not None:
            self._fit_fitness_ensemble(feature_matrix, fitness)

    def apply_consistency_refinement(
        self,
        feature_matrix: np.ndarray,
        targets: dict[str, np.ndarray],
        *,
        weight: float,
    ) -> float:
        """Refit stability/diversity after nudging targets by fitness residual."""
        if weight <= 0.0 or not self._uses_lightgbm:
            return consistency_mae_on_rows(self, feature_matrix, targets)
        from worldspace.surrogate.evaluation import fitness_from_target_row

        before = consistency_mae_on_rows(self, feature_matrix, targets)
        adjusted = {
            key: np.asarray(targets[key], dtype=float).copy() for key in TARGET_KEYS
        }
        n_rows = int(feature_matrix.shape[0])
        for row_index in range(n_rows):
            row_features = feature_matrix[row_index]
            components = self.predict_components(row_features)
            prediction = SurrogatePrediction(
                components=components,
                measures={
                    "stability": float(components["stability"]),
                    "diversity": float(components["diversity"]),
                },
                fitness=0.0,
                uncertainty=0.0,
            )
            pred_fitness = compute_fitness_from_prediction(prediction)
            actual_fitness = fitness_from_target_row(
                {key: float(targets[key][row_index]) for key in TARGET_KEYS}
            )
            delta = float(weight) * (actual_fitness - pred_fitness) * 0.5
            for key in _CONSISTENCY_REFINE_KEYS:
                adjusted[key][row_index] = float(
                    np.clip(adjusted[key][row_index] + delta, 0.0, 1.0)
                )
        self._refit_lightgbm_components(
            feature_matrix, adjusted, _CONSISTENCY_REFINE_KEYS
        )
        return before

    def set_component_defaults(self, value: float) -> None:
        """Set all target means to one deterministic value."""
        self._component_means = {key: float(value) for key in TARGET_KEYS}
        self._ensemble = {}
        self._fitness_ensemble = []
        self._uses_lightgbm = False
        self._uses_mlp = False
        self._mlp_members = []
        self._has_fitness_head = False

    def predict_fitness(self, features: np.ndarray) -> float | None:
        """Predict illuminator fitness directly when the fitness head is trained."""
        if not self._has_fitness_head:
            return None
        vector = np.asarray(features, dtype=float).reshape(-1)
        if self._uses_mlp:
            return self._predict_mlp_fitness(vector)
        row = _lightgbm_feature_row(vector)
        values = [
            float(estimator.predict(row)[0]) for estimator in self._fitness_ensemble
        ]
        return float(np.mean(values))

    def predict_components(self, features: np.ndarray) -> dict[str, float]:
        """Predict all Strategy A target components from extracted features."""
        vector = np.asarray(features, dtype=float).reshape(-1)
        if self._uses_mlp:
            return self._predict_mlp_components(vector)
        if self._uses_lightgbm:
            row = _lightgbm_feature_row(vector)
            return {
                key: float(
                    np.mean(
                        [
                            float(estimator.predict(row)[0])
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
        if self._uses_mlp:
            return self._predict_mlp_uncertainty(vector)
        if not self._uses_lightgbm:
            return 0.0
        fitness_values: list[float] = []
        row = _lightgbm_feature_row(vector)
        for member_index in range(self.ensemble_size):
            components = {
                key: float(self._ensemble[key][member_index].predict(row)[0])
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

    def _fit_mlp_ensemble(
        self,
        feature_matrix: np.ndarray,
        targets: dict[str, np.ndarray],
        *,
        val_features: np.ndarray | None = None,
        val_targets: dict[str, np.ndarray] | None = None,
    ) -> None:
        from worldspace.surrogate.mlp_model import (
            MlpTrainConfig,
            ensemble_member_seed,
            train_mlp_member,
        )

        apply_mlp_determinism()
        fitness = targets.get(FITNESS_TARGET_KEY)
        train_fitness_head = False
        if fitness is not None:
            labels = np.asarray(fitness, dtype=float).reshape(-1)
            train_fitness_head = (
                int(np.isfinite(labels).sum()) >= MIN_FITNESS_HEAD_SAMPLES
            )

        config = MlpTrainConfig(
            hidden_dims=self._mlp_hidden_dims,
            fitness_loss_weight=self._fitness_loss_weight,
        )
        members: list[dict[str, Any]] = []
        for member_index in range(self.ensemble_size):
            seed = ensemble_member_seed(self.random_state, member_index)
            state_dict = train_mlp_member(
                feature_matrix,
                targets,
                seed=seed,
                config=config,
                val_features=val_features,
                val_targets=val_targets,
            )
            members.append(state_dict)
        self._mlp_members = members
        self._uses_mlp = True
        self._has_fitness_head = train_fitness_head

    def _predict_mlp_outputs(self, vector: np.ndarray) -> np.ndarray:
        from worldspace.surrogate.mlp_model import predict_mlp_state_dict

        matrix = np.asarray(vector, dtype=float).reshape(1, -1)
        member_preds = [
            predict_mlp_state_dict(
                state_dict,
                matrix,
                hidden_dims=self._mlp_hidden_dims,
            )[0]
            for state_dict in self._mlp_members
        ]
        return np.mean(np.stack(member_preds, axis=0), axis=0)

    def _predict_mlp_member_outputs(self, vector: np.ndarray) -> list[np.ndarray]:
        from worldspace.surrogate.mlp_model import predict_mlp_state_dict

        matrix = np.asarray(vector, dtype=float).reshape(1, -1)
        return [
            predict_mlp_state_dict(
                state_dict,
                matrix,
                hidden_dims=self._mlp_hidden_dims,
            )[0]
            for state_dict in self._mlp_members
        ]

    def _predict_mlp_components(self, vector: np.ndarray) -> dict[str, float]:
        outputs = self._predict_mlp_outputs(vector)
        return {key: float(outputs[index]) for index, key in enumerate(TARGET_KEYS)}

    def _predict_mlp_fitness(self, vector: np.ndarray) -> float:
        from worldspace.surrogate.mlp_model import FITNESS_OUTPUT_INDEX

        member_outputs = self._predict_mlp_member_outputs(vector)
        values = [float(row[FITNESS_OUTPUT_INDEX]) for row in member_outputs]
        return float(np.mean(values))

    def _predict_mlp_uncertainty(self, vector: np.ndarray) -> float:
        from worldspace.surrogate.mlp_model import FITNESS_OUTPUT_INDEX

        fitness_values: list[float] = []
        for row in self._predict_mlp_member_outputs(vector):
            components = {
                key: float(row[index]) for index, key in enumerate(TARGET_KEYS)
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
            if self._has_fitness_head:
                fitness_values.append(float(row[FITNESS_OUTPUT_INDEX]))
            else:
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

        x_train = _lightgbm_feature_matrix(feature_matrix)
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

    def _fit_fitness_ensemble(
        self,
        feature_matrix: np.ndarray,
        fitness: np.ndarray,
    ) -> None:
        import lightgbm as lgb

        labels = np.asarray(fitness, dtype=float).reshape(-1)
        mask = np.isfinite(labels)
        if int(mask.sum()) < MIN_FITNESS_HEAD_SAMPLES:
            return
        x_train = _lightgbm_feature_matrix(feature_matrix[mask])
        y_train = labels[mask]
        base_params = {
            **lightgbm_deterministic_params(),
            "n_estimators": 64,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "verbosity": -1,
        }
        estimators: list[Any] = []
        for member_index in range(self.ensemble_size):
            member_params = dict(base_params)
            member_params["random_state"] = member_random_state(member_index)
            regressor = lgb.LGBMRegressor(**member_params)
            regressor.fit(x_train, y_train)
            estimators.append(regressor)
        self._fitness_ensemble = estimators
        self._has_fitness_head = True

    def _refit_lightgbm_components(
        self,
        feature_matrix: np.ndarray,
        targets: dict[str, np.ndarray],
        keys: tuple[str, ...],
    ) -> None:
        import lightgbm as lgb

        if not self._uses_lightgbm:
            return
        x_train = _lightgbm_feature_matrix(feature_matrix)
        base_params = {
            **lightgbm_deterministic_params(),
            "n_estimators": 32,
            "num_leaves": 31,
            "learning_rate": 0.03,
            "verbosity": -1,
        }
        for key in keys:
            y_train = _as_float_array(targets, key)
            estimators: list[Any] = []
            for member_index in range(self.ensemble_size):
                member_params = dict(base_params)
                member_params["random_state"] = member_random_state(member_index)
                regressor = lgb.LGBMRegressor(**member_params)
                regressor.fit(x_train, y_train)
                estimators.append(regressor)
            self._ensemble[key] = estimators


def consistency_mae_on_rows(
    model: SurrogateModel,
    feature_matrix: np.ndarray,
    targets: dict[str, np.ndarray],
) -> float:
    """Mean |predicted fitness - target fitness| on matrix rows."""
    from worldspace.surrogate.evaluation import fitness_from_target_row

    n_rows = int(feature_matrix.shape[0])
    if n_rows == 0:
        return float("nan")
    errors = np.empty(n_rows, dtype=float)
    for row_index in range(n_rows):
        components = model.predict_components(feature_matrix[row_index])
        prediction = SurrogatePrediction(
            components=components,
            measures={
                "stability": float(components["stability"]),
                "diversity": float(components["diversity"]),
            },
            fitness=0.0,
            uncertainty=0.0,
        )
        pred_fitness = compute_fitness_from_prediction(prediction)
        actual_fitness = fitness_from_target_row(
            {key: float(targets[key][row_index]) for key in TARGET_KEYS}
        )
        errors[row_index] = abs(pred_fitness - actual_fitness)
    return float(np.mean(errors))


def checkpoint_feature_dim(model: SurrogateModel) -> int | None:
    """Return trained input width for LightGBM or MLP checkpoints.

    Default-only models (no fitted backend) return ``None``.
    """
    if model._uses_mlp:
        return EXPECTED_FEATURE_DIM if model._mlp_members else None
    if not model._uses_lightgbm:
        return None
    dims: set[int] = set()
    for key in TARGET_KEYS:
        estimators = model._ensemble.get(key)
        if not estimators:
            continue
        for estimator in estimators:
            n_features = getattr(estimator, "n_features_in_", None)
            if n_features is None:
                booster = getattr(estimator, "booster_", None)
                if booster is not None:
                    n_features = booster.num_feature()
            if n_features is not None:
                dims.add(int(n_features))
    if not dims:
        return None
    if len(dims) > 1:
        msg = f"Inconsistent checkpoint feature dimensions: {sorted(dims)}"
        raise ValueError(msg)
    return next(iter(dims))


def checkpoint_matches_extractor(model: SurrogateModel) -> bool:
    """Return whether a checkpoint was trained for the current feature extractor."""
    dim = checkpoint_feature_dim(model)
    if dim is None:
        return True
    return dim == EXPECTED_FEATURE_DIM


def _lightgbm_feature_matrix(feature_matrix: np.ndarray) -> Any:
    """Wrap feature rows with stable column names for sklearn LightGBM."""
    import pandas as pd

    matrix = np.asarray(feature_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        msg = (
            f"feature matrix must be (N, {len(FEATURE_NAMES)}), "
            f"got shape={matrix.shape!r}"
        )
        raise ValueError(msg)
    return pd.DataFrame(matrix, columns=list(FEATURE_NAMES))


def _lightgbm_feature_row(vector: np.ndarray) -> Any:
    """Single-row feature frame for LightGBM predict."""
    return _lightgbm_feature_matrix(np.asarray(vector, dtype=float).reshape(1, -1))


def _as_float_array(targets: dict[str, np.ndarray], key: str) -> np.ndarray:
    values = targets.get(key)
    if values is None:
        msg = f"Missing target array for key={key!r}"
        raise ValueError(msg)
    return np.asarray(values, dtype=float)

"""Surrogate component model API for MVP Strategy A."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any

import numpy as np

from worldspace.surrogate.genome_features import FEATURE_DIM
from worldspace.surrogate.feature_extractor import FEATURE_NAMES, feature_names_for_dim
from worldspace.surrogate.determinism import (
    DEFAULT_ENSEMBLE_SIZE,
    DEFAULT_MLP_HIDDEN_DIMS,
    DEFAULT_RANDOM_STATE,
    apply_mlp_determinism,
    lightgbm_deterministic_params,
    member_random_state,
)
from worldspace.surrogate.device import DevicePreference, resolve_lightgbm_device
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


@dataclass
class SurrogateModel:
    """Deterministic component regressor for Strategy A outputs."""

    model_type: str = "mlp"
    random_state: int = DEFAULT_RANDOM_STATE
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE
    _component_means: dict[str, float] = field(default_factory=dict)
    _ensemble: dict[str, list[Any]] = field(default_factory=dict)
    _fitness_ensemble: list[Any] = field(default_factory=list)
    _uses_lightgbm: bool = False
    _uses_mlp: bool = False
    _mlp_members: list[dict[str, Any]] = field(default_factory=list)
    _mlp_hidden_dims: tuple[int, ...] = DEFAULT_MLP_HIDDEN_DIMS
    _fitness_loss_weight: float = 1.0
    _has_fitness_head: bool = False
    _training_device_preference: DevicePreference = "auto"
    _resolved_lightgbm_device: str = "cpu"
    _mlp_dropout_p: float = 0.0
    _mlp_mc_samples: int = 16
    _mlp_uncertainty_method: str = "ensemble"

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
            members = self.__dict__.get("_mlp_members") or []
            if members:
                from worldspace.surrogate.mlp_model import hidden_dims_from_state_dict

                self.__dict__["_mlp_hidden_dims"] = hidden_dims_from_state_dict(
                    members[0]
                )
            else:
                from worldspace.surrogate.determinism import LEGACY_MLP_HIDDEN_DIMS

                self.__dict__["_mlp_hidden_dims"] = LEGACY_MLP_HIDDEN_DIMS
        if "_fitness_loss_weight" not in self.__dict__:
            self.__dict__["_fitness_loss_weight"] = 1.0
        if "_has_fitness_head" not in self.__dict__:
            self.__dict__["_has_fitness_head"] = False
        if "_training_device_preference" not in self.__dict__:
            self.__dict__["_training_device_preference"] = "auto"
        if "_resolved_lightgbm_device" not in self.__dict__:
            self.__dict__["_resolved_lightgbm_device"] = "cpu"
        if "_mlp_dropout_p" not in self.__dict__:
            self.__dict__["_mlp_dropout_p"] = 0.0
        if "_mlp_mc_samples" not in self.__dict__:
            self.__dict__["_mlp_mc_samples"] = 16
        if "_mlp_uncertainty_method" not in self.__dict__:
            self.__dict__["_mlp_uncertainty_method"] = "ensemble"
        if "_trained_input_dim" not in self.__dict__ and self.__dict__.get(
            "_mlp_members"
        ):
            from worldspace.surrogate.mlp_model import input_dim_from_state_dict

            self.__dict__["_trained_input_dim"] = input_dim_from_state_dict(
                self.__dict__["_mlp_members"][0]
            )
        if (
            self.__dict__.get("_has_fitness_head")
            and not self._fitness_head_is_trained()
        ):
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
        sample_weight: np.ndarray | None = None,
        device: DevicePreference = "auto",
    ) -> None:
        """Fit component regressors; LightGBM or MLP ensemble when available."""
        self._training_device_preference = device
        self._resolved_lightgbm_device = resolve_lightgbm_device(device)
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
        self._trained_input_dim = int(feature_matrix.shape[1])

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
        self._fit_lightgbm_ensemble(
            feature_matrix, targets, sample_weight=sample_weight
        )
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
        return self.predict_components_batch(
            np.asarray(features, dtype=float).reshape(1, -1)
        )[0]

    def predict_components_batch(
        self, feature_matrix: np.ndarray
    ) -> list[dict[str, float]]:
        """Predict component dicts for each row of ``feature_matrix``."""
        matrix = np.asarray(feature_matrix, dtype=float)
        if matrix.ndim != 2:
            msg = f"feature matrix must be 2-D, got shape={matrix.shape!r}"
            raise ValueError(msg)
        n_rows = int(matrix.shape[0])
        if n_rows == 0:
            return []
        if self._uses_mlp:
            return self._predict_mlp_components_batch(matrix)
        if self._uses_lightgbm:
            return self._predict_lightgbm_components_batch(matrix)
        if not self._component_means:
            self.set_component_defaults(0.5)
        defaults = dict(self._component_means)
        return [dict(defaults) for _ in range(n_rows)]

    def predict_uncertainty(self, features: np.ndarray) -> float:
        """Return ensemble standard deviation of predicted fitness."""
        return float(
            self.predict_uncertainty_batch(
                np.asarray(features, dtype=float).reshape(1, -1)
            )[0]
        )

    def predict_uncertainty_batch(self, feature_matrix: np.ndarray) -> list[float]:
        """Return ensemble uncertainty for each row of ``feature_matrix``."""
        matrix = np.asarray(feature_matrix, dtype=float)
        if matrix.ndim != 2:
            msg = f"feature matrix must be 2-D, got shape={matrix.shape!r}"
            raise ValueError(msg)
        n_rows = int(matrix.shape[0])
        if n_rows == 0:
            return []
        if self._uses_mlp:
            return [
                self._predict_mlp_uncertainty(matrix[row_index])
                for row_index in range(n_rows)
            ]
        if not self._uses_lightgbm:
            return [0.0] * n_rows
        return [
            self._predict_lightgbm_uncertainty_row(matrix[row_index])
            for row_index in range(n_rows)
        ]

    def _predict_lightgbm_uncertainty_row(self, vector: np.ndarray) -> float:
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

    def _predict_lightgbm_components_batch(
        self, feature_matrix: np.ndarray
    ) -> list[dict[str, float]]:
        x_train = _lightgbm_feature_matrix(feature_matrix)
        n_rows = int(feature_matrix.shape[0])
        means_by_key = {
            key: np.mean(
                np.stack(
                    [
                        np.asarray(estimator.predict(x_train), dtype=float)
                        for estimator in self._ensemble[key]
                    ],
                    axis=0,
                ),
                axis=0,
            )
            for key in TARGET_KEYS
        }
        return [
            {key: float(means_by_key[key][row_index]) for key in TARGET_KEYS}
            for row_index in range(n_rows)
        ]

    def _predict_mlp_components_batch(
        self, feature_matrix: np.ndarray
    ) -> list[dict[str, float]]:
        from worldspace.surrogate.mlp_model import predict_mlp_state_dict

        member_preds = [
            predict_mlp_state_dict(
                state_dict,
                feature_matrix,
                hidden_dims=self._mlp_hidden_dims,
                input_dim=self._trained_input_dim,
            )
            for state_dict in self._mlp_members
        ]
        mean_preds = np.mean(np.stack(member_preds, axis=0), axis=0)
        return [
            {
                key: float(mean_preds[row_index, index])
                for index, key in enumerate(TARGET_KEYS)
            }
            for row_index in range(mean_preds.shape[0])
        ]

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
        self._mlp_dropout_p = float(config.dropout_p)
        self._mlp_mc_samples = int(config.mc_samples)
        self._mlp_uncertainty_method = str(config.uncertainty_method)
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
                device=self._training_device_preference,
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
                input_dim=self._trained_input_dim,
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
                input_dim=self._trained_input_dim,
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

    def _fitness_from_mlp_output_row(self, row: np.ndarray) -> float:
        from worldspace.surrogate.mlp_model import FITNESS_OUTPUT_INDEX

        if self._has_fitness_head:
            return float(row[FITNESS_OUTPUT_INDEX])
        components = {key: float(row[index]) for index, key in enumerate(TARGET_KEYS)}
        prediction = SurrogatePrediction(
            components=components,
            measures={
                "stability": components["stability"],
                "diversity": components["diversity"],
            },
            fitness=0.0,
            uncertainty=0.0,
        )
        return compute_fitness_from_prediction(prediction)

    def _ensemble_member_fitness_values(self, vector: np.ndarray) -> list[float]:
        fitness_values: list[float] = []
        for row in self._predict_mlp_member_outputs(vector):
            fitness_values.append(self._fitness_from_mlp_output_row(row))
        return fitness_values

    def _predict_mlp_uncertainty(self, vector: np.ndarray) -> float:
        from worldspace.surrogate.mlp_model import (
            ensemble_member_seed,
            mlp_state_dict_uses_dropout,
            sample_mlp_member_outputs_mc,
        )

        use_mc = (
            self._mlp_uncertainty_method == "ensemble_mc"
            and self._mlp_members
            and mlp_state_dict_uses_dropout(self._mlp_members[0])
            and self._mlp_mc_samples > 0
        )
        if use_mc:
            fitness_values: list[float] = []
            for member_index, state_dict in enumerate(self._mlp_members):
                seed = ensemble_member_seed(
                    self.random_state,
                    member_index * max(1, self._mlp_mc_samples),
                )
                for row in sample_mlp_member_outputs_mc(
                    state_dict,
                    vector,
                    hidden_dims=self._mlp_hidden_dims,
                    input_dim=self._trained_input_dim,
                    n_samples=self._mlp_mc_samples,
                    seed=seed,
                ):
                    fitness_values.append(self._fitness_from_mlp_output_row(row))
            if len(fitness_values) >= 2:
                return float(np.std(np.asarray(fitness_values, dtype=float), ddof=0))

        fitness_values = self._ensemble_member_fitness_values(vector)
        if len(fitness_values) < 2:
            return 0.0
        return float(np.std(np.asarray(fitness_values, dtype=float), ddof=0))

    def _fit_lightgbm_ensemble(
        self,
        feature_matrix: np.ndarray,
        targets: dict[str, np.ndarray],
        *,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        import lightgbm as lgb

        x_train = _lightgbm_feature_matrix(feature_matrix)
        weights = None
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float).reshape(-1)
        base_params = {
            **lightgbm_deterministic_params(),
            "device": self._resolved_lightgbm_device,
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
                regressor.fit(x_train, y_train, sample_weight=weights)
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
            "device": self._resolved_lightgbm_device,
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
            "device": self._resolved_lightgbm_device,
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
    trained_dim = getattr(model, "_trained_input_dim", None)
    if trained_dim is not None:
        return int(trained_dim)
    if model._uses_mlp:
        if model._mlp_members:
            from worldspace.surrogate.mlp_model import input_dim_from_state_dict

            return input_dim_from_state_dict(model._mlp_members[0])
        return None
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
    return dim in (FEATURE_DIM, EXPECTED_FEATURE_DIM)


def _lightgbm_feature_matrix(feature_matrix: np.ndarray) -> Any:
    """Wrap feature rows with stable column names for sklearn LightGBM."""
    import pandas as pd

    matrix = np.asarray(feature_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        msg = f"feature matrix must be (N, D) with N >= 1, got shape={matrix.shape!r}"
        raise ValueError(msg)
    n_features = int(matrix.shape[1])
    columns = feature_names_for_dim(n_features)
    if len(columns) != n_features:
        columns = tuple(f"feature_{index}" for index in range(n_features))
    return pd.DataFrame(matrix, columns=list(columns))


def _lightgbm_feature_row(vector: np.ndarray) -> Any:
    """Single-row feature frame for LightGBM predict."""
    return _lightgbm_feature_matrix(np.asarray(vector, dtype=float).reshape(1, -1))


def _as_float_array(targets: dict[str, np.ndarray], key: str) -> np.ndarray:
    values = targets.get(key)
    if values is None:
        msg = f"Missing target array for key={key!r}"
        raise ValueError(msg)
    return np.asarray(values, dtype=float)

"""Maze-specific deterministic MLP ensemble and checkpoint utilities."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from worldspace.mazes.evaluation import MazeEvaluation
from worldspace.mazes.features import FEATURE_NAMES, extract_features
from worldspace.mazes.spec import MazeSpec


@dataclass(frozen=True)
class MazePrediction:
    components: dict[str, float]
    measures: dict[str, float]
    fitness: float
    uncertainty: float


@dataclass
class MazeSurrogateCheckpoint:
    feature_names: tuple[str, ...]
    scaler: StandardScaler
    models: list[MLPRegressor]
    uncertainty_scale: float
    fitness_threshold: float
    uncertainty_threshold: float


class MazeSurrogate:
    def __init__(self, checkpoint: MazeSurrogateCheckpoint) -> None:
        if checkpoint.feature_names != FEATURE_NAMES:
            raise ValueError("maze surrogate feature schema mismatch")
        self.checkpoint = checkpoint

    @classmethod
    def load(cls, path: Path) -> MazeSurrogate:
        with path.open("rb") as handle:
            checkpoint = pickle.load(handle)  # noqa: S301
        if not isinstance(checkpoint, MazeSurrogateCheckpoint):
            raise TypeError("invalid maze surrogate checkpoint")
        return cls(checkpoint)

    def predict(self, spec: MazeSpec) -> MazePrediction:
        features = extract_features(spec)[np.newaxis, :]
        transformed = self.checkpoint.scaler.transform(features)
        members = np.stack(
            [
                np.asarray(model.predict(transformed))[0]
                for model in self.checkpoint.models
            ]
        )
        mean = np.mean(members, axis=0)
        uncertainty = float(
            np.std(members[:, 0], ddof=0) * self.checkpoint.uncertainty_scale
        )
        return MazePrediction(
            components={},
            measures={
                "path_length": float(np.clip(mean[1], 0.0, 1.0)),
                "branching": float(np.clip(mean[2], 0.0, 1.0)),
            },
            fitness=float(np.clip(mean[0], 0.0, 1.0)),
            uncertainty=max(0.0, uncertainty),
        )


def buffer_row(
    spec: MazeSpec,
    evaluation: MazeEvaluation,
    *,
    design_seed: int,
) -> dict[str, object]:
    return {
        "schema_version": "maze-surrogate-1.0",
        "candidate_hash": spec.candidate_hash(),
        "design_seed": design_seed,
        "features": extract_features(spec).tolist(),
        "targets": {
            "fitness": evaluation.fitness,
            "path_length": evaluation.measures[0],
            "branching": evaluation.measures[1],
        },
    }


def load_buffer(
    path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    features: list[list[float]] = []
    targets: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        features.append([float(value) for value in row["features"]])
        target = row["targets"]
        targets.append(
            [
                float(target["fitness"]),
                float(target["path_length"]),
                float(target["branching"]),
            ]
        )
    if len(features) < 20:
        raise ValueError("maze surrogate buffer requires at least 20 rows")
    return np.asarray(features), np.asarray(targets)


def train_checkpoint(
    features: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    random_state: int = 1729,
    ensemble_size: int = 5,
    max_iter: int = 300,
) -> tuple[MazeSurrogateCheckpoint, dict[str, object]]:
    """Fit an ensemble and return checkpoint plus frozen hold-out diagnostics."""
    rng = np.random.default_rng(random_state)
    order = rng.permutation(features.shape[0])
    split = max(1, int(features.shape[0] * 0.8))
    train_idx, holdout_idx = order[:split], order[split:]
    if holdout_idx.size == 0:
        raise ValueError("training data must leave a hold-out split")
    scaler = StandardScaler().fit(features[train_idx])
    train_x = scaler.transform(features[train_idx])
    holdout_x = scaler.transform(features[holdout_idx])
    models: list[MLPRegressor] = []
    for member in range(ensemble_size):
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            random_state=random_state + member,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )
        model.fit(train_x, targets[train_idx])
        models.append(model)
    member_predictions = np.stack([model.predict(holdout_x) for model in models])
    prediction = np.mean(member_predictions, axis=0)
    raw_uncertainty = np.std(member_predictions[:, :, 0], axis=0)
    absolute_error = np.abs(prediction[:, 0] - targets[holdout_idx, 0])
    positive = raw_uncertainty > 1e-9
    uncertainty_scale = (
        float(np.median(absolute_error[positive] / raw_uncertainty[positive]))
        if np.any(positive)
        else 1.0
    )
    predicted_fitness = np.clip(prediction[:, 0], 0.0, 1.0)
    fitness_threshold = float(np.quantile(predicted_fitness, 0.40))
    calibrated_uncertainty = raw_uncertainty * uncertainty_scale
    uncertainty_threshold = float(np.quantile(calibrated_uncertainty, 0.75))
    mae = float(np.mean(absolute_error))
    baseline = float(
        np.mean(np.abs(targets[holdout_idx, 0] - float(np.mean(targets[train_idx, 0]))))
    )
    spearman_result = cast(
        Any,
        stats.spearmanr(predicted_fitness, targets[holdout_idx, 0]),
    )
    spearman = float(spearman_result.statistic)
    shadow_skip = float(
        np.mean(
            (predicted_fitness < fitness_threshold)
            & (calibrated_uncertainty <= uncertainty_threshold)
        )
    )
    checkpoint = MazeSurrogateCheckpoint(
        feature_names=FEATURE_NAMES,
        scaler=scaler,
        models=models,
        uncertainty_scale=uncertainty_scale,
        fitness_threshold=fitness_threshold,
        uncertainty_threshold=uncertainty_threshold,
    )
    report: dict[str, object] = {
        "schema_version": "maze-surrogate-1.0",
        "rows": int(features.shape[0]),
        "train_rows": int(train_idx.size),
        "holdout_rows": int(holdout_idx.size),
        "fitness_mae": mae,
        "mean_baseline_mae": baseline,
        "spearman_fitness": spearman,
        "quality_gate_pass": mae < baseline and spearman > 0.0,
        "uncertainty_scale": uncertainty_scale,
        "fitness_threshold": fitness_threshold,
        "uncertainty_threshold": uncertainty_threshold,
        "shadow_skip_rate": shadow_skip,
        "shadow_skip_gate_pass": 0.25 <= shadow_skip <= 0.45,
    }
    return checkpoint, report


def save_checkpoint(checkpoint: MazeSurrogateCheckpoint, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(checkpoint, handle)

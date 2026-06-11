"""Hold-out quality metrics for surrogate training."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from worldspace.surrogate.model import FITNESS_TARGET_KEY, TARGET_KEYS, SurrogateModel
from worldspace.surrogate.types import SurrogatePrediction
from worldspace.surrogate.utils import compute_fitness_from_prediction

MIN_TRAIN_SAMPLES_FULL = 2000
MIN_TRAIN_SAMPLES_MICRO = 100
QUALITY_R2_FITNESS_MIN = 0.72
QUALITY_MAE_FITNESS_MAX = 0.085
QUALITY_MAE_STABILITY_MAX = 0.06

__all__ = [
    "MIN_TRAIN_SAMPLES_FULL",
    "MIN_TRAIN_SAMPLES_MICRO",
    "QUALITY_MAE_FITNESS_MAX",
    "QUALITY_MAE_STABILITY_MAX",
    "QUALITY_R2_FITNESS_MIN",
    "evaluate_holdout",
    "fitness_from_target_row",
    "quality_thresholds_met",
]


def fitness_from_target_row(
    targets: dict[str, float],
    *,
    prefer_stored: bool = False,
) -> float:
    """Derive illuminator fitness from one Strategy A target dict."""
    if prefer_stored and FITNESS_TARGET_KEY in targets:
        stored = float(targets[FITNESS_TARGET_KEY])
        if np.isfinite(stored):
            return stored
    components = {key: float(targets[key]) for key in TARGET_KEYS}
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


def evaluate_holdout(
    model: SurrogateModel,
    feature_matrix: np.ndarray,
    targets: dict[str, np.ndarray],
) -> dict[str, float]:
    """Score ``model`` on hold-out rows; return R²/MAE for fitness and stability."""
    n_rows = int(feature_matrix.shape[0])
    if n_rows < 1:
        msg = "hold-out set must contain at least one row"
        raise ValueError(msg)

    true_fitness = np.asarray(
        [
            fitness_from_target_row({k: float(targets[k][i]) for k in TARGET_KEYS})
            for i in range(n_rows)
        ],
        dtype=float,
    )
    true_stability = np.asarray(targets["stability"], dtype=float)
    pred_fitness = np.empty(n_rows, dtype=float)
    pred_stability = np.empty(n_rows, dtype=float)

    for row_index in range(n_rows):
        components = model.predict_components(feature_matrix[row_index])
        pred_stability[row_index] = float(components["stability"])
        prediction = SurrogatePrediction(
            components=components,
            measures={
                "stability": float(components["stability"]),
                "diversity": float(components["diversity"]),
            },
            fitness=0.0,
            uncertainty=float(model.predict_uncertainty(feature_matrix[row_index])),
        )
        pred_fitness[row_index] = compute_fitness_from_prediction(prediction)

    metrics = {
        "r2_fitness": float(r2_score(true_fitness, pred_fitness)),
        "mae_fitness": float(mean_absolute_error(true_fitness, pred_fitness)),
        "mae_stability": float(mean_absolute_error(true_stability, pred_stability)),
    }
    fitness_labels = targets.get(FITNESS_TARGET_KEY)
    if fitness_labels is not None and model._has_fitness_head:
        label_array = np.asarray(fitness_labels, dtype=float)
        valid_mask = np.isfinite(label_array)
        if int(valid_mask.sum()) >= 2:
            true_direct = label_array[valid_mask]
            pred_direct = np.asarray(
                [
                    float(model.predict_fitness(feature_matrix[row_index]))
                    for row_index in np.where(valid_mask)[0]
                ],
                dtype=float,
            )
            metrics["r2_fitness_direct"] = float(r2_score(true_direct, pred_direct))
            metrics["mae_fitness_direct"] = float(
                mean_absolute_error(true_direct, pred_direct)
            )
    return metrics


def quality_thresholds_met(metrics: dict[str, float]) -> bool:
    """Return whether hold-out metrics satisfy MVP DoD thresholds."""
    return (
        metrics["r2_fitness"] > QUALITY_R2_FITNESS_MIN
        and metrics["mae_fitness"] < QUALITY_MAE_FITNESS_MAX
        and metrics["mae_stability"] < QUALITY_MAE_STABILITY_MAX
    )

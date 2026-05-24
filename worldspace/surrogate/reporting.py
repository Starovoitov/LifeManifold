"""Offline Surrogate Acquisition quality metrics from buffer replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from worldspace.illuminators.archive import GridArchive
from worldspace.illuminators.scheduler import TargetBin
from worldspace.surrogate.acquisition import decide
from worldspace.surrogate.acquisition_config import AcquisitionConfig
from worldspace.surrogate.calibration import (
    UncertaintyCalibrator,
    apply_calibrated_uncertainty,
    expected_calibration_error,
    load_uncertainty_calibration,
)
from worldspace.surrogate.evaluation import fitness_from_target_row
from worldspace.surrogate.model import TARGET_KEYS, SurrogateModel
from worldspace.surrogate.types import SurrogatePrediction
from worldspace.surrogate.utils import compute_fitness_from_prediction

FALSE_SKIP_FITNESS_MARGIN = 0.01

__all__ = [
    "FALSE_SKIP_FITNESS_MARGIN",
    "AcquisitionReplayMetrics",
    "consistency_mae",
    "evaluate_acquisition_replay",
    "estimate_false_skip_rate",
    "merge_acquisition_into_summary",
    "recommended_skip_rate_at_policy",
    "load_calibration_for_report",
]


@dataclass(frozen=True)
class AcquisitionReplayMetrics:
    """Offline acquisition replay statistics for one feature matrix."""

    row_count: int
    policy_skip_count: int
    recommended_skip_rate: float
    false_skip_count: int
    false_skip_rate_estimate: float
    consistency_mae: float
    calibration_ece: float

    def as_dict(self) -> dict[str, Any]:
        """Serialize metrics for JSON summaries."""
        return {
            "row_count": self.row_count,
            "policy_skip_count": self.policy_skip_count,
            "recommended_skip_rate": self.recommended_skip_rate,
            "false_skip_count": self.false_skip_count,
            "false_skip_rate_estimate": self.false_skip_rate_estimate,
            "consistency_mae": self.consistency_mae,
            "calibration_ece": self.calibration_ece,
        }


def evaluate_acquisition_replay(
    model: SurrogateModel,
    feature_matrix: np.ndarray,
    targets: dict[str, np.ndarray],
    policy: AcquisitionConfig,
    *,
    calibrator: UncertaintyCalibrator | None = None,
    grid_resolution: int = 10,
    never_skip_empty_bin: bool | None = None,
) -> AcquisitionReplayMetrics:
    """Replay buffer rows through predict + policy (offline proxy metrics)."""
    effective_policy = policy
    if never_skip_empty_bin is not None:
        from dataclasses import replace

        effective_policy = replace(policy, never_skip_empty_bin=never_skip_empty_bin)
    archive = GridArchive(grid_resolution)
    n_rows = int(feature_matrix.shape[0])
    policy_skips = 0
    false_skips = 0
    pred_uncertainties: list[float] = []
    abs_errors: list[float] = []
    for row_index in range(n_rows):
        row_features = feature_matrix[row_index]
        prediction = _predict_row(model, row_features, calibrator=calibrator)
        actual_fitness = fitness_from_target_row(
            {key: float(targets[key][row_index]) for key in TARGET_KEYS}
        )
        abs_errors.append(abs(prediction.fitness - actual_fitness))
        pred_uncertainties.append(prediction.uncertainty)
        i = row_index % grid_resolution
        j = (row_index // grid_resolution) % grid_resolution
        target = TargetBin(
            bin=(i, j),
            target_stability=0.5,
            target_diversity=0.5,
        )
        decision = decide(effective_policy, prediction, target, archive)
        if decision.action == "skip":
            policy_skips += 1
            if _is_false_skip_proxy(
                prediction,
                actual_fitness,
                effective_policy,
            ):
                false_skips += 1
    skip_rate = float(policy_skips) / float(n_rows) if n_rows else 0.0
    false_rate = float(false_skips) / float(policy_skips) if policy_skips > 0 else 0.0
    ece = expected_calibration_error(
        np.asarray(pred_uncertainties, dtype=float),
        np.asarray(abs_errors, dtype=float),
    )
    return AcquisitionReplayMetrics(
        row_count=n_rows,
        policy_skip_count=policy_skips,
        recommended_skip_rate=skip_rate,
        false_skip_count=false_skips,
        false_skip_rate_estimate=false_rate,
        consistency_mae=float(np.mean(abs_errors)) if abs_errors else float("nan"),
        calibration_ece=float(ece),
    )


def estimate_false_skip_rate(
    model: SurrogateModel,
    feature_matrix: np.ndarray,
    targets: dict[str, np.ndarray],
    policy: AcquisitionConfig,
    *,
    calibrator: UncertaintyCalibrator | None = None,
    grid_resolution: int = 10,
) -> float:
    """Fraction of policy skips that look like false positives on buffer replay."""
    metrics = evaluate_acquisition_replay(
        model,
        feature_matrix,
        targets,
        policy,
        calibrator=calibrator,
        grid_resolution=grid_resolution,
        never_skip_empty_bin=False,
    )
    return metrics.false_skip_rate_estimate


def recommended_skip_rate_at_policy(
    model: SurrogateModel,
    feature_matrix: np.ndarray,
    targets: dict[str, np.ndarray],
    policy: AcquisitionConfig,
    *,
    calibrator: UncertaintyCalibrator | None = None,
    grid_resolution: int = 10,
) -> float:
    """Fraction of replay rows where the policy recommends skip."""
    metrics = evaluate_acquisition_replay(
        model,
        feature_matrix,
        targets,
        policy,
        calibrator=calibrator,
        grid_resolution=grid_resolution,
        never_skip_empty_bin=False,
    )
    return metrics.recommended_skip_rate


def consistency_mae(
    model: SurrogateModel,
    feature_matrix: np.ndarray,
    targets: dict[str, np.ndarray],
) -> float:
    """Mean absolute error between predicted and target-derived fitness."""
    n_rows = int(feature_matrix.shape[0])
    if n_rows == 0:
        return float("nan")
    errors = np.empty(n_rows, dtype=float)
    for row_index in range(n_rows):
        prediction = _predict_row(model, feature_matrix[row_index])
        actual_fitness = fitness_from_target_row(
            {key: float(targets[key][row_index]) for key in TARGET_KEYS}
        )
        errors[row_index] = abs(prediction.fitness - actual_fitness)
    return float(np.mean(errors))


def merge_acquisition_into_summary(
    summary_path: Path | str,
    acquisition: dict[str, Any],
) -> None:
    """Merge fields into the acquisition block of an existing training summary JSON."""
    path = Path(summary_path).expanduser()
    payload: dict[str, Any] = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    existing = payload.get("acquisition")
    if isinstance(existing, dict):
        merged: dict[str, Any] = {**existing, **acquisition}
    else:
        merged = dict(acquisition)
    payload["acquisition"] = merged
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def load_calibration_for_report(
    calibration_path: Path | str | None,
) -> UncertaintyCalibrator | None:
    """Load a calibrator when a path is provided for replay metrics."""
    if calibration_path is None:
        return None
    stripped = str(calibration_path).strip()
    if not stripped:
        return None
    return load_uncertainty_calibration(stripped)


def _predict_row(
    model: SurrogateModel,
    features: np.ndarray,
    *,
    calibrator: UncertaintyCalibrator | None = None,
) -> SurrogatePrediction:
    components = model.predict_components(features)
    raw_uncertainty = float(model.predict_uncertainty(features))
    uncertainty = apply_calibrated_uncertainty(
        calibrator,
        raw_uncertainty,
        calibration_configured=calibrator is not None,
    )
    prediction = SurrogatePrediction(
        components=components,
        measures={
            "stability": float(components["stability"]),
            "diversity": float(components["diversity"]),
        },
        fitness=0.0,
        uncertainty=uncertainty,
    )
    return SurrogatePrediction(
        components=prediction.components,
        measures=prediction.measures,
        fitness=compute_fitness_from_prediction(prediction),
        uncertainty=prediction.uncertainty,
    )


def _is_false_skip_proxy(
    prediction: SurrogatePrediction,
    actual_fitness: float,
    policy: AcquisitionConfig,
) -> bool:
    """Heuristic false skip: policy skip but realized fitness clears the bar."""
    return (
        actual_fitness >= policy.min_predicted_fitness
        and actual_fitness > prediction.fitness + FALSE_SKIP_FITNESS_MARGIN
    )

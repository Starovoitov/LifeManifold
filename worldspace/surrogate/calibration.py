"""Load, fit, and apply surrogate uncertainty calibration (isotonic on hold-out)."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from worldspace.surrogate.checkpoint_io import (
    CHECKPOINT_LOAD_ERRORS,
    load_surrogate_checkpoint,
)
from worldspace.surrogate.evaluation import fitness_from_target_row
from worldspace.surrogate.model import TARGET_KEYS, SurrogateModel
from worldspace.surrogate.training import holdout_split, load_buffer
from worldspace.surrogate.types import SurrogatePrediction
from worldspace.surrogate.utils import compute_fitness_from_prediction

CALIBRATION_SCHEMA_VERSION = "1.0"
CALIBRATION_METHOD_ISOTONIC = "isotonic_v1"
DEFAULT_MAX_ECE = 0.12
MIN_CALIBRATION_HOLDOUT_SAMPLES = 20
DEFAULT_CALIBRATION_BINS = 10

logger = logging.getLogger(__name__)

_missing_calibration_warned = False

__all__ = [
    "CALIBRATION_METHOD_ISOTONIC",
    "CALIBRATION_SCHEMA_VERSION",
    "CalibrationResult",
    "DEFAULT_MAX_ECE",
    "UncertaintyCalibrator",
    "apply_calibrated_uncertainty",
    "collect_holdout_calibration_pairs",
    "expected_calibration_error",
    "fit_calibration_from_buffer",
    "fit_uncertainty_calibrator",
    "load_uncertainty_calibration",
    "save_uncertainty_calibration",
]


@dataclass(frozen=True)
class UncertaintyCalibrator:
    """Monotonic map from raw ensemble spread to expected absolute fitness error."""

    schema_version: str
    method: str
    x_thresholds: tuple[float, ...]
    y_thresholds: tuple[float, ...]

    def apply(self, raw_uncertainty: float) -> float:
        """Return calibrated uncertainty for one raw ensemble spread value."""
        value = float(raw_uncertainty)
        if not self.x_thresholds:
            return max(0.0, value)
        if len(self.x_thresholds) == 1:
            return max(0.0, float(self.y_thresholds[0]))
        calibrated = float(
            np.interp(
                value,
                np.asarray(self.x_thresholds, dtype=float),
                np.asarray(self.y_thresholds, dtype=float),
            )
        )
        return max(0.0, calibrated)


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of one offline calibration fit."""

    success: bool
    holdout_samples: int
    calibration_path: Path
    summary_path: Path
    ece: float
    raw_min: float
    raw_max: float
    calibrated_min: float
    calibrated_max: float
    ece_passed: bool
    error_message: str | None = None


def load_uncertainty_calibration(path: Path | str) -> UncertaintyCalibrator | None:
    """Load a calibration artifact; return None when missing or unreadable."""
    target = Path(path).expanduser()
    if not target.is_file():
        return None
    try:
        with target.open("rb") as handle:
            loaded = pickle.load(handle)
    except CHECKPOINT_LOAD_ERRORS as exc:
        logger.warning("Failed to load uncertainty calibration %s: %s", target, exc)
        return None
    if isinstance(loaded, UncertaintyCalibrator):
        return loaded
    logger.warning(
        "Uncertainty calibration %s has unexpected type %r",
        target,
        type(loaded),
    )
    return None


def save_uncertainty_calibration(
    calibrator: UncertaintyCalibrator,
    path: Path | str,
) -> None:
    """Persist a fitted uncertainty calibrator."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(calibrator, handle)


def fit_uncertainty_calibrator(
    raw_uncertainty: np.ndarray,
    abs_error: np.ndarray,
) -> UncertaintyCalibrator:
    """Fit isotonic regression mapping raw spread to hold-out absolute error."""
    raw = np.asarray(raw_uncertainty, dtype=float).reshape(-1)
    errors = np.asarray(abs_error, dtype=float).reshape(-1)
    if raw.size < 2:
        msg = "need at least two hold-out samples to fit uncertainty calibration"
        raise ValueError(msg)
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0)
    isotonic.fit(raw, errors)
    x_thresholds = tuple(float(v) for v in isotonic.X_thresholds_)
    y_thresholds = tuple(float(v) for v in isotonic.y_thresholds_)
    return UncertaintyCalibrator(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        method=CALIBRATION_METHOD_ISOTONIC,
        x_thresholds=x_thresholds,
        y_thresholds=y_thresholds,
    )


def collect_holdout_calibration_pairs(
    model: SurrogateModel,
    feature_matrix: np.ndarray,
    targets: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect (raw_uncertainty, absolute_fitness_error) pairs on hold-out rows."""
    n_rows = int(feature_matrix.shape[0])
    raw_values = np.empty(n_rows, dtype=float)
    abs_errors = np.empty(n_rows, dtype=float)
    for row_index in range(n_rows):
        row_features = feature_matrix[row_index]
        components = model.predict_components(row_features)
        raw_values[row_index] = float(model.predict_uncertainty(row_features))
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
        abs_errors[row_index] = abs(pred_fitness - actual_fitness)
    return raw_values, abs_errors


def expected_calibration_error(
    predicted_uncertainty: np.ndarray,
    actual_abs_error: np.ndarray,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> float:
    """Weighted mean |bin_mean(predicted_u) - bin_mean(actual_error)| across uncertainty bins."""
    predicted = np.asarray(predicted_uncertainty, dtype=float).reshape(-1)
    actual = np.asarray(actual_abs_error, dtype=float).reshape(-1)
    if predicted.size == 0:
        return float("nan")
    if predicted.size < n_bins:
        return float(np.mean(np.abs(predicted - actual)))
    order = np.argsort(predicted)
    sorted_pred = predicted[order]
    sorted_actual = actual[order]
    chunks = np.array_split(np.arange(sorted_pred.size), n_bins)
    total_weight = 0.0
    weighted_gap = 0.0
    for indices in chunks:
        if indices.size == 0:
            continue
        bin_pred = sorted_pred[indices]
        bin_actual = sorted_actual[indices]
        gap = abs(float(np.mean(bin_pred)) - float(np.mean(bin_actual)))
        weight = float(indices.size) / float(predicted.size)
        weighted_gap += gap * weight
        total_weight += weight
    if total_weight <= 0.0:
        return float("nan")
    return weighted_gap / total_weight


def apply_calibrated_uncertainty(
    calibrator: UncertaintyCalibrator | None,
    raw_uncertainty: float,
    *,
    calibration_configured: bool,
) -> float:
    """Map raw spread to calibrated uncertainty; warn once when path configured but missing."""
    raw = max(0.0, float(raw_uncertainty))
    if calibrator is not None:
        return calibrator.apply(raw)
    if calibration_configured:
        _warn_missing_calibration_once()
    return raw


def fit_calibration_from_buffer(
    *,
    buffer_path: Path,
    checkpoint_path: Path,
    calibration_path: Path,
    summary_path: Path | None = None,
    test_fraction: float = 0.2,
    random_state: int = 42,
    max_ece: float = DEFAULT_MAX_ECE,
    require_ece_gate: bool = True,
) -> CalibrationResult:
    """Fit isotonic calibration on the same hold-out split used for training reports."""
    resolved_summary = summary_path or calibration_path.with_name(
        f"{calibration_path.stem}.summary.json"
    )
    try:
        feature_matrix, targets = load_buffer(buffer_path.expanduser())
        _train_x, _train_y, holdout_x, holdout_y = holdout_split(
            feature_matrix,
            targets,
            test_fraction=test_fraction,
            random_state=random_state,
        )
        holdout_count = int(holdout_x.shape[0])
        if holdout_count < MIN_CALIBRATION_HOLDOUT_SAMPLES:
            return CalibrationResult(
                success=False,
                holdout_samples=holdout_count,
                calibration_path=calibration_path,
                summary_path=resolved_summary,
                ece=float("nan"),
                raw_min=0.0,
                raw_max=0.0,
                calibrated_min=0.0,
                calibrated_max=0.0,
                ece_passed=False,
                error_message=(
                    f"Need at least {MIN_CALIBRATION_HOLDOUT_SAMPLES} hold-out rows, "
                    f"got {holdout_count}"
                ),
            )
        model = load_surrogate_checkpoint(checkpoint_path.expanduser())
        raw_values, abs_errors = collect_holdout_calibration_pairs(
            model,
            holdout_x,
            holdout_y,
        )
        calibrator = fit_uncertainty_calibrator(raw_values, abs_errors)
        calibrated_values = np.asarray(
            [calibrator.apply(float(value)) for value in raw_values],
            dtype=float,
        )
        ece = expected_calibration_error(calibrated_values, abs_errors)
        ece_passed = bool(np.isfinite(ece) and ece < max_ece)
        save_uncertainty_calibration(calibrator, calibration_path)
        _save_calibration_summary(
            resolved_summary,
            holdout_samples=holdout_count,
            ece=ece,
            max_ece=max_ece,
            ece_passed=ece_passed,
            raw_min=float(np.min(raw_values)),
            raw_max=float(np.max(raw_values)),
            calibrated_min=float(np.min(calibrated_values)),
            calibrated_max=float(np.max(calibrated_values)),
        )
        if require_ece_gate and not ece_passed:
            return CalibrationResult(
                success=False,
                holdout_samples=holdout_count,
                calibration_path=calibration_path,
                summary_path=resolved_summary,
                ece=ece,
                raw_min=float(np.min(raw_values)),
                raw_max=float(np.max(raw_values)),
                calibrated_min=float(np.min(calibrated_values)),
                calibrated_max=float(np.max(calibrated_values)),
                ece_passed=False,
                error_message=f"ECE {ece:.4f} >= target {max_ece}",
            )
    except (OSError, ValueError, TypeError) as exc:
        return CalibrationResult(
            success=False,
            holdout_samples=0,
            calibration_path=calibration_path,
            summary_path=resolved_summary,
            ece=float("nan"),
            raw_min=0.0,
            raw_max=0.0,
            calibrated_min=0.0,
            calibrated_max=0.0,
            ece_passed=False,
            error_message=str(exc),
        )

    return CalibrationResult(
        success=True,
        holdout_samples=holdout_count,
        calibration_path=calibration_path,
        summary_path=resolved_summary,
        ece=ece,
        raw_min=float(np.min(raw_values)),
        raw_max=float(np.max(raw_values)),
        calibrated_min=float(np.min(calibrated_values)),
        calibrated_max=float(np.max(calibrated_values)),
        ece_passed=ece_passed,
    )


def _warn_missing_calibration_once() -> None:
    global _missing_calibration_warned
    if _missing_calibration_warned:
        return
    _missing_calibration_warned = True
    logger.warning(
        "Uncertainty calibration path configured but artifact missing or invalid; "
        "using raw ensemble spread"
    )


def _save_calibration_summary(
    path: Path,
    *,
    holdout_samples: int,
    ece: float,
    max_ece: float,
    ece_passed: bool,
    raw_min: float,
    raw_max: float,
    calibrated_min: float,
    calibrated_max: float,
) -> None:
    payload = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "method": CALIBRATION_METHOD_ISOTONIC,
        "holdout_samples": holdout_samples,
        "ece": ece,
        "max_ece": max_ece,
        "ece_passed": ece_passed,
        "raw_uncertainty_min": raw_min,
        "raw_uncertainty_max": raw_max,
        "calibrated_uncertainty_min": calibrated_min,
        "calibrated_uncertainty_max": calibrated_max,
    }
    target = path.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )

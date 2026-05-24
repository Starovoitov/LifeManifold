"""Programmatic surrogate training from an append-only buffer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

from worldspace.surrogate.checkpoint_io import save_surrogate_checkpoint
from worldspace.surrogate.evaluation import (
    MIN_TRAIN_SAMPLES_FULL,
    MIN_TRAIN_SAMPLES_MICRO,
    evaluate_holdout,
    quality_thresholds_met,
)
from worldspace.surrogate.model import TARGET_KEYS, SurrogateModel
from worldspace.surrogate.training import holdout_split, load_buffer

ModelType = Literal["lightgbm", "mlp"]

__all__ = [
    "ModelType",
    "TrainError",
    "TrainResult",
    "default_summary_path",
    "train_from_buffer",
    "validate_model_dependencies",
]


@dataclass(frozen=True)
class TrainResult:
    """Outcome of one in-process training attempt."""

    success: bool
    sample_count: int
    holdout_metrics: dict[str, float]
    quality_passed: bool
    checkpoint_path: Path
    summary_path: Path
    error_message: str | None = None


class TrainError(Exception):
    """Raised when training cannot complete successfully."""


def default_summary_path(checkpoint_path: Path) -> Path:
    """Return the default JSON summary path beside a checkpoint file."""
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.summary.json")


def validate_model_dependencies(model_type: ModelType) -> None:
    """Fail fast when the requested model backend is unavailable."""
    if model_type == "lightgbm" and find_spec("lightgbm") is None:
        msg = (
            "Model type 'lightgbm' requested, but dependency is missing. "
            "Install project dependencies or use model_type='mlp'."
        )
        raise TrainError(msg)


def train_from_buffer(
    *,
    buffer_path: Path,
    checkpoint_path: Path,
    summary_path: Path | None = None,
    model_type: ModelType = "lightgbm",
    micro: bool = False,
    min_samples: int | None = None,
    require_quality_gate: bool = True,
) -> TrainResult:
    """Train a surrogate from buffer JSONL and write checkpoint + summary."""
    resolved_summary = summary_path or default_summary_path(checkpoint_path)
    try:
        validate_model_dependencies(model_type)
    except TrainError as exc:
        return TrainResult(
            success=False,
            sample_count=0,
            holdout_metrics={},
            quality_passed=False,
            checkpoint_path=checkpoint_path,
            summary_path=resolved_summary,
            error_message=str(exc),
        )
    effective_min = min_samples
    if effective_min is None:
        effective_min = MIN_TRAIN_SAMPLES_MICRO if micro else MIN_TRAIN_SAMPLES_FULL

    try:
        feature_matrix, targets = load_buffer(buffer_path.expanduser())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return TrainResult(
            success=False,
            sample_count=0,
            holdout_metrics={},
            quality_passed=False,
            checkpoint_path=checkpoint_path,
            summary_path=resolved_summary,
            error_message=str(exc),
        )

    sample_count = int(feature_matrix.shape[0])
    if sample_count < effective_min:
        return TrainResult(
            success=False,
            sample_count=sample_count,
            holdout_metrics={},
            quality_passed=False,
            checkpoint_path=checkpoint_path,
            summary_path=resolved_summary,
            error_message=(
                f"Need at least {effective_min} buffer rows, got {sample_count}"
            ),
        )

    try:
        x_train, y_train, x_holdout, y_holdout = holdout_split(
            feature_matrix,
            targets,
        )
        model = SurrogateModel(
            model_type=model_type,
            random_state=42,
            ensemble_size=8,
        )
        model.fit(x_train, y_train)
        holdout_metrics = evaluate_holdout(model, x_holdout, y_holdout)
        quality_passed = quality_thresholds_met(holdout_metrics)
        if not micro and require_quality_gate and not quality_passed:
            return TrainResult(
                success=False,
                sample_count=sample_count,
                holdout_metrics=holdout_metrics,
                quality_passed=False,
                checkpoint_path=checkpoint_path,
                summary_path=resolved_summary,
                error_message="Hold-out quality thresholds were not met",
            )
        save_surrogate_checkpoint(model, checkpoint_path)
        _save_summary(
            resolved_summary,
            model_type=model_type,
            sample_count=sample_count,
            train_count=int(x_train.shape[0]),
            holdout_count=int(x_holdout.shape[0]),
            feature_dim=int(feature_matrix.shape[1]),
            holdout_metrics=holdout_metrics,
            micro=micro,
        )
    except (
        Exception
    ) as exc:  # noqa: BLE001 — training failures must not crash illuminator
        return TrainResult(
            success=False,
            sample_count=sample_count,
            holdout_metrics={},
            quality_passed=False,
            checkpoint_path=checkpoint_path,
            summary_path=resolved_summary,
            error_message=str(exc),
        )

    return TrainResult(
        success=True,
        sample_count=sample_count,
        holdout_metrics=holdout_metrics,
        quality_passed=quality_passed,
        checkpoint_path=checkpoint_path,
        summary_path=resolved_summary,
    )


def _save_summary(
    path: Path,
    *,
    model_type: ModelType,
    sample_count: int,
    train_count: int,
    holdout_count: int,
    feature_dim: int,
    holdout_metrics: dict[str, float],
    micro: bool,
) -> None:
    payload = {
        "model_type": model_type,
        "sample_count": sample_count,
        "train_count": train_count,
        "holdout_count": holdout_count,
        "feature_dim": feature_dim,
        "target_keys": list(TARGET_KEYS),
        "holdout_metrics": holdout_metrics,
        "quality_passed": quality_thresholds_met(holdout_metrics),
        "micro": micro,
    }
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

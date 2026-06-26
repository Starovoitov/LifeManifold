"""Programmatic surrogate training from an append-only buffer."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

from worldspace.surrogate.checkpoint_io import save_surrogate_checkpoint
from worldspace.surrogate.evaluation import (
    MIN_TRAIN_SAMPLES_FULL,
    MIN_TRAIN_SAMPLES_MICRO,
    evaluate_holdout,
    hints_thresholds_met,
    per_target_holdout,
    quality_thresholds_met,
)
from worldspace.surrogate.device import (
    DevicePreference,
    resolve_lightgbm_device,
    resolve_training_device,
)
from worldspace.surrogate.acquisition_config import AcquisitionConfig
import numpy as np

from worldspace.surrogate.model import (
    FITNESS_TARGET_KEY,
    TARGET_KEYS,
    SurrogateModel,
    consistency_mae_on_rows,
)
from worldspace.surrogate.reporting import (
    evaluate_acquisition_replay,
    load_calibration_for_report,
    merge_acquisition_into_summary,
)
from worldspace.surrogate.training import (
    BUFFER_SCHEMA_VERSION,
    detect_buffer_schema_version,
    holdout_split,
    load_buffer,
    load_buffer_emitter_types,
    training_sample_weights,
)
from worldspace.surrogate.feature_extractor import feature_dim_for_schema

ModelType = Literal["lightgbm", "mlp"]

logger = logging.getLogger(__name__)

__all__ = [
    "DevicePreference",
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
    consistency_mae_before: float | None = None
    consistency_mae_after: float | None = None
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
    if model_type == "mlp" and find_spec("torch") is None:
        msg = (
            "Model type 'mlp' requested, but dependency is missing. "
            "Install project dependencies (pyproject.toml includes torch>=2.2) "
            "or use model_type='lightgbm'."
        )
        raise TrainError(msg)


def train_from_buffer(
    *,
    buffer_path: Path,
    checkpoint_path: Path,
    summary_path: Path | None = None,
    model_type: ModelType = "mlp",
    micro: bool = False,
    min_samples: int | None = None,
    require_quality_gate: bool = True,
    consistency_weight: float = 0.0,
    fitness_loss_weight: float = 1.0,
    emitter_onehot: bool = False,
    stratify_emitter: bool = False,
    low_stability_weight: float = 1.0,
    acquisition_report: bool = False,
    calibration_path: Path | str | None = None,
    acquisition_policy: AcquisitionConfig | None = None,
    device: DevicePreference = "auto",
    mlp_dropout_p: float | None = None,
    mlp_mc_samples: int | None = None,
    mlp_uncertainty_method: str | None = None,
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
        feature_matrix, targets = load_buffer(
            buffer_path.expanduser(),
            emitter_onehot=emitter_onehot,
        )
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
    feature_dim = int(feature_matrix.shape[1])
    schema_version = detect_buffer_schema_version(buffer_path.expanduser())
    expected_dim = feature_dim_for_schema(schema_version)
    if feature_dim != expected_dim:
        logger.warning(
            "Buffer %s: feature_schema_version=%s expects %d dims but loaded width is %d; "
            "training uses actual row width. Re-featurize with "
            "scripts/migrate_surrogate_buffer.py --re-featurize for schema 2.1 rows.",
            buffer_path,
            schema_version,
            expected_dim,
            feature_dim,
        )
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
        stratify_labels = None
        if stratify_emitter:
            stratify_labels = load_buffer_emitter_types(buffer_path.expanduser())
        x_train, y_train, x_holdout, y_holdout = holdout_split(
            feature_matrix,
            targets,
            stratify_labels=stratify_labels,
        )
        sample_weight = training_sample_weights(
            y_train,
            low_stability_weight=low_stability_weight,
        )
        model = SurrogateModel(
            model_type=model_type,
            random_state=42,
        )
        model.fit(
            x_train,
            y_train,
            fitness_loss_weight=fitness_loss_weight,
            val_features=x_holdout,
            val_targets=y_holdout,
            sample_weight=sample_weight,
            device=device,
            mlp_dropout_p=mlp_dropout_p,
            mlp_mc_samples=mlp_mc_samples,
            mlp_uncertainty_method=mlp_uncertainty_method,
        )
        consistency_before: float | None = None
        consistency_after: float | None = None
        if consistency_weight > 0.0:
            consistency_before = model.apply_consistency_refinement(
                x_train,
                y_train,
                weight=consistency_weight,
            )
            consistency_after = consistency_mae_on_rows(model, x_train, y_train)
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
                consistency_mae_before=consistency_before,
                consistency_mae_after=consistency_after,
                error_message="Hold-out quality thresholds were not met",
            )
        save_surrogate_checkpoint(model, checkpoint_path)
        fitness_rows_with_label: int | None = None
        if FITNESS_TARGET_KEY in targets:
            fitness_rows_with_label = int(
                np.isfinite(targets[FITNESS_TARGET_KEY]).sum()
            )
        resolved_torch_device = resolve_training_device(device)
        resolved_lightgbm_device = resolve_lightgbm_device(device)
        _save_summary(
            resolved_summary,
            model_type=model_type,
            sample_count=sample_count,
            train_count=int(x_train.shape[0]),
            holdout_count=int(x_holdout.shape[0]),
            feature_dim=int(feature_matrix.shape[1]),
            holdout_metrics=holdout_metrics,
            per_target_holdout_rows=per_target_holdout(model, x_holdout, y_holdout),
            micro=micro,
            consistency_mae_before=consistency_before,
            consistency_mae_after=consistency_after,
            fitness_rows_with_label=fitness_rows_with_label,
            fitness_loss_weight=fitness_loss_weight if model_type == "mlp" else None,
            mlp_dropout_p=(
                float(model._mlp_dropout_p) if model_type == "mlp" else None
            ),
            mlp_mc_samples=(
                int(model._mlp_mc_samples) if model_type == "mlp" else None
            ),
            mlp_uncertainty_method=(
                str(model._mlp_uncertainty_method) if model_type == "mlp" else None
            ),
            mlp_hidden_dims=(
                list(model._mlp_hidden_dims) if model_type == "mlp" else None
            ),
            ensemble_size=int(model.ensemble_size),
            emitter_onehot=emitter_onehot,
            stratify_emitter=stratify_emitter,
            low_stability_weight=low_stability_weight,
            consistency_weight=consistency_weight,
            training_device_requested=device,
            training_device_resolved=(
                resolved_torch_device
                if model_type == "mlp"
                else resolved_lightgbm_device
            ),
        )
        if acquisition_report:
            policy = acquisition_policy or AcquisitionConfig(mode="filter")
            calibrator = load_calibration_for_report(calibration_path)
            replay = evaluate_acquisition_replay(
                model,
                x_holdout,
                y_holdout,
                policy,
                calibrator=calibrator,
            )
            acquisition_block = replay.as_dict()
            acquisition_block["policy_mode"] = policy.mode
            acquisition_block["policy_min_predicted_fitness"] = (
                policy.min_predicted_fitness
            )
            acquisition_block["policy_max_uncertainty_to_skip"] = (
                policy.max_uncertainty_to_skip
            )
            if consistency_before is not None:
                acquisition_block["consistency_mae_train_before"] = consistency_before
            if consistency_after is not None:
                acquisition_block["consistency_mae_train_after"] = consistency_after
            merge_acquisition_into_summary(resolved_summary, acquisition_block)
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
        consistency_mae_before=consistency_before,
        consistency_mae_after=consistency_after,
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
    per_target_holdout_rows: list[dict[str, float | str]],
    micro: bool,
    consistency_mae_before: float | None = None,
    consistency_mae_after: float | None = None,
    fitness_rows_with_label: int | None = None,
    fitness_loss_weight: float | None = None,
    mlp_dropout_p: float | None = None,
    mlp_mc_samples: int | None = None,
    mlp_uncertainty_method: str | None = None,
    mlp_hidden_dims: list[int] | None = None,
    ensemble_size: int | None = None,
    emitter_onehot: bool = False,
    stratify_emitter: bool = False,
    low_stability_weight: float = 1.0,
    consistency_weight: float = 0.0,
    training_device_requested: DevicePreference = "auto",
    training_device_resolved: str = "cpu",
) -> None:
    payload: dict[str, object] = {
        "model_type": model_type,
        "sample_count": sample_count,
        "train_count": train_count,
        "holdout_count": holdout_count,
        "feature_dim": feature_dim,
        "feature_schema_version": BUFFER_SCHEMA_VERSION,
        "target_keys": list(TARGET_KEYS),
        "holdout_metrics": holdout_metrics,
        "per_target_holdout": per_target_holdout_rows,
        "quality_passed": quality_thresholds_met(holdout_metrics),
        "hints_ok": hints_thresholds_met(holdout_metrics),
        "micro": micro,
    }
    if consistency_mae_before is not None:
        payload["consistency_mae_train_before"] = consistency_mae_before
    if consistency_mae_after is not None:
        payload["consistency_mae_train_after"] = consistency_mae_after
    if fitness_rows_with_label is not None:
        payload["fitness_rows_with_label"] = fitness_rows_with_label
    if fitness_loss_weight is not None:
        payload["fitness_loss_weight"] = fitness_loss_weight
    if mlp_dropout_p is not None and mlp_dropout_p > 0.0:
        payload["mlp_dropout_p"] = float(mlp_dropout_p)
    if mlp_mc_samples is not None:
        payload["mlp_mc_samples"] = int(mlp_mc_samples)
    if mlp_uncertainty_method is not None:
        payload["mlp_uncertainty_method"] = str(mlp_uncertainty_method)
    if mlp_hidden_dims is not None:
        payload["mlp_hidden_dims"] = list(mlp_hidden_dims)
    if ensemble_size is not None:
        payload["ensemble_size"] = int(ensemble_size)
    if emitter_onehot:
        payload["emitter_onehot"] = True
    if stratify_emitter:
        payload["stratify_emitter"] = True
    if low_stability_weight > 1.0:
        payload["low_stability_weight"] = float(low_stability_weight)
    if consistency_weight > 0.0:
        payload["consistency_weight"] = float(consistency_weight)
    payload["training_device_requested"] = training_device_requested
    payload["training_device_resolved"] = training_device_resolved
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

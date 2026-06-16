"""Shared training data loading and hold-out split."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from worldspace.surrogate.emitter_features import augment_features_with_emitter
from worldspace.surrogate.feature_extractor import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    SUPPORTED_FEATURE_SCHEMA_VERSIONS,
    feature_dim_for_schema,
)
from worldspace.surrogate.model import FITNESS_TARGET_KEY, TARGET_KEYS

logger = logging.getLogger(__name__)

__all__ = [
    "BUFFER_FEATURE_DIM",
    "BUFFER_SCHEMA_VERSION",
    "LOW_STABILITY_BAND_MAX",
    "detect_buffer_schema_version",
    "holdout_split",
    "load_buffer",
    "load_buffer_emitter_types",
    "scan_buffer_rows",
    "training_sample_weights",
]

BUFFER_SCHEMA_VERSION = FEATURE_SCHEMA_VERSION
BUFFER_FEATURE_DIM = len(FEATURE_NAMES)
LOW_STABILITY_BAND_MAX = 0.3


def detect_buffer_schema_version(path: Path) -> str:
    """Return the single schema version used by all non-empty rows in a buffer."""
    versions: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                msg = f"Invalid row format at {path}:{line_no}: expected JSON object"
                raise ValueError(msg)
            schema = row.get("feature_schema_version")
            if schema not in SUPPORTED_FEATURE_SCHEMA_VERSIONS:
                msg = (
                    f"Unsupported feature_schema_version at {path}:{line_no}: "
                    f"got {schema!r}"
                )
                raise ValueError(msg)
            versions.add(str(schema))
    if not versions:
        msg = f"No training samples found in {path}"
        raise ValueError(msg)
    if len(versions) > 1:
        msg = (
            f"Mixed feature_schema_version values in {path}: "
            f"{sorted(versions)}; migrate or split before training"
        )
        raise ValueError(msg)
    return next(iter(versions))


def load_buffer(
    path: Path,
    *,
    emitter_onehot: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load a schema 2.0/2.1 training matrix and per-target arrays from JSONL buffer."""
    if not path.is_file():
        msg = f"Buffer JSONL not found: {path}"
        raise FileNotFoundError(msg)
    schema_version = detect_buffer_schema_version(path)
    expected_dim = feature_dim_for_schema(schema_version)
    features: list[list[float]] = []
    emitter_types: list[str] = []
    target_rows: dict[str, list[float]] = {key: [] for key in TARGET_KEYS}
    fitness_rows: list[float] = []
    rows_without_fitness = 0
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"Invalid JSON at {path}:{line_no}: {exc}"
                raise ValueError(msg) from exc
            _validate_buffer_row(
                row,
                path=path,
                line_no=line_no,
                schema_version=schema_version,
                expected_dim=expected_dim,
            )
            row_features = row["features"]
            row_targets = row["targets"]
            features.append([float(value) for value in row_features])
            emitter_types.append(str(row.get("emitter_type") or "unknown"))
            for key in TARGET_KEYS:
                target_rows[key].append(float(row_targets[key]))
            if FITNESS_TARGET_KEY in row_targets:
                fitness_rows.append(float(row_targets[FITNESS_TARGET_KEY]))
            else:
                fitness_rows.append(float("nan"))
                rows_without_fitness += 1
    if not features:
        msg = f"No training samples found in {path}"
        raise ValueError(msg)
    if rows_without_fitness:
        logger.warning(
            "Buffer %s: %d/%d rows missing optional target %r",
            path,
            rows_without_fitness,
            len(features),
            FITNESS_TARGET_KEY,
        )
    feature_matrix = np.asarray(features, dtype=float)
    if emitter_onehot:
        feature_matrix = augment_features_with_emitter(
            feature_matrix,
            np.asarray(emitter_types, dtype=object),
        )
    targets = {
        key: np.asarray(values, dtype=float) for key, values in target_rows.items()
    }
    targets[FITNESS_TARGET_KEY] = np.asarray(fitness_rows, dtype=float)
    return feature_matrix, targets


def load_buffer_emitter_types(path: Path) -> np.ndarray:
    """Return per-row emitter_type labels in the same order as ``load_buffer``."""
    if not path.is_file():
        msg = f"Buffer JSONL not found: {path}"
        raise FileNotFoundError(msg)
    labels: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                msg = f"Invalid row format in {path}: expected JSON object"
                raise ValueError(msg)
            labels.append(str(row.get("emitter_type") or "unknown"))
    if not labels:
        msg = f"No training samples found in {path}"
        raise ValueError(msg)
    return np.asarray(labels, dtype=object)


def scan_buffer_rows(path: Path) -> dict[str, Any]:
    """Summarize buffer JSONL rows without building training matrices."""
    if not path.is_file():
        msg = f"Buffer JSONL not found: {path}"
        raise FileNotFoundError(msg)
    valid_rows = 0
    invalid_rows = 0
    rows_with_fitness = 0
    feature_dims: dict[int, int] = {}
    schema_versions: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                continue
            try:
                schema = str(row.get("feature_schema_version"))
                expected_dim = feature_dim_for_schema(schema)
                _validate_buffer_row(
                    row,
                    path=path,
                    line_no=line_no,
                    schema_version=schema,
                    expected_dim=expected_dim,
                )
            except ValueError:
                invalid_rows += 1
                continue
            valid_rows += 1
            row_targets = row.get("targets")
            if isinstance(row_targets, dict) and FITNESS_TARGET_KEY in row_targets:
                rows_with_fitness += 1
            schema_versions[schema] = schema_versions.get(schema, 0) + 1
            dim = len(row["features"])
            feature_dims[dim] = feature_dims.get(dim, 0) + 1
    return {
        "path": str(path.resolve()),
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "rows_with_fitness": rows_with_fitness,
        "feature_schema_version": BUFFER_SCHEMA_VERSION,
        "feature_dim": BUFFER_FEATURE_DIM,
        "schema_versions": schema_versions,
        "feature_dims": feature_dims,
    }


def holdout_split(
    feature_matrix: np.ndarray,
    targets: dict[str, np.ndarray],
    *,
    test_fraction: float = 0.2,
    random_state: int = 42,
    stratify_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Split features and targets into train and hold-out sets."""
    n_rows = int(feature_matrix.shape[0])
    if n_rows < 2:
        msg = "need at least two samples for hold-out split"
        raise ValueError(msg)
    if stratify_labels is not None:
        train_indices, test_indices = _stratified_holdout_indices(
            np.asarray(stratify_labels).reshape(-1),
            test_fraction=test_fraction,
            random_state=random_state,
        )
    else:
        rng = np.random.default_rng(random_state)
        indices = np.arange(n_rows)
        rng.shuffle(indices)
        test_size = max(1, int(round(n_rows * test_fraction)))
        test_size = min(test_size, n_rows - 1)
        test_indices = np.sort(indices[:test_size])
        train_indices = np.sort(indices[test_size:])
    target_keys = list(TARGET_KEYS)
    if FITNESS_TARGET_KEY in targets:
        target_keys.append(FITNESS_TARGET_KEY)
    train_targets = {key: targets[key][train_indices] for key in target_keys}
    test_targets = {key: targets[key][test_indices] for key in target_keys}
    return (
        feature_matrix[train_indices],
        train_targets,
        feature_matrix[test_indices],
        test_targets,
    )


def training_sample_weights(
    targets: dict[str, np.ndarray],
    *,
    low_stability_weight: float,
) -> np.ndarray | None:
    """Return per-row LightGBM weights emphasizing low-stability training rows."""
    if low_stability_weight <= 1.0:
        return None
    stability = np.asarray(targets["stability"], dtype=float)
    weights = np.ones(int(stability.shape[0]), dtype=float)
    weights[stability < LOW_STABILITY_BAND_MAX] = float(low_stability_weight)
    return weights


def _validate_buffer_row(
    row: object,
    *,
    path: Path,
    line_no: int,
    schema_version: str,
    expected_dim: int,
) -> None:
    if not isinstance(row, dict):
        msg = f"Invalid row format at {path}:{line_no}: expected JSON object"
        raise ValueError(msg)
    schema = row.get("feature_schema_version")
    if schema != schema_version:
        msg = (
            f"Inconsistent feature_schema_version at {path}:{line_no}: "
            f"expected {schema_version!r}, got {schema!r}"
        )
        raise ValueError(msg)
    if schema not in SUPPORTED_FEATURE_SCHEMA_VERSIONS:
        msg = (
            f"Unsupported feature_schema_version at {path}:{line_no}: "
            f"got {schema!r}"
        )
        raise ValueError(msg)
    row_features = row.get("features")
    if not isinstance(row_features, list) or not row_features:
        msg = f"Invalid features at {path}:{line_no}"
        raise ValueError(msg)
    if len(row_features) != expected_dim:
        msg = (
            f"Invalid feature dimension at {path}:{line_no}: "
            f"expected {expected_dim}, got {len(row_features)}"
        )
        raise ValueError(msg)
    row_targets = row.get("targets")
    if not isinstance(row_targets, dict):
        msg = f"Invalid targets at {path}:{line_no}"
        raise ValueError(msg)
    for key in TARGET_KEYS:
        if key not in row_targets:
            msg = f"Missing target {key!r} at {path}:{line_no}"
            raise ValueError(msg)
    world_spec = row.get("world_spec")
    if not isinstance(world_spec, dict) or not world_spec:
        msg = f"Missing world_spec at {path}:{line_no}"
        raise ValueError(msg)


def _stratified_holdout_indices(
    labels: np.ndarray,
    *,
    test_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = int(labels.shape[0])
    if n_rows < 2:
        msg = "need at least two samples for hold-out split"
        raise ValueError(msg)
    rng = np.random.default_rng(random_state)
    test_parts: list[np.ndarray] = []
    train_parts: list[np.ndarray] = []
    for label in np.unique(labels):
        group = np.flatnonzero(labels == label)
        if int(group.shape[0]) < 1:
            continue
        shuffled = group.copy()
        rng.shuffle(shuffled)
        group_test = max(0, int(round(int(shuffled.shape[0]) * test_fraction)))
        if group_test == 0 and int(shuffled.shape[0]) > 1:
            group_test = 1
        group_test = min(group_test, int(shuffled.shape[0]) - 1)
        test_parts.append(np.sort(shuffled[:group_test]))
        train_parts.append(np.sort(shuffled[group_test:]))
    test_indices = (
        np.sort(np.concatenate(test_parts)) if test_parts else np.array([], dtype=int)
    )
    train_indices = (
        np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int)
    )
    if int(train_indices.shape[0]) < 1 or int(test_indices.shape[0]) < 1:
        msg = "stratified hold-out split produced an empty train or test partition"
        raise ValueError(msg)
    return train_indices, test_indices

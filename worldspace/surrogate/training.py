"""Shared training data loading and hold-out split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from worldspace.surrogate.feature_extractor import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from worldspace.surrogate.model import TARGET_KEYS

__all__ = [
    "BUFFER_FEATURE_DIM",
    "BUFFER_SCHEMA_VERSION",
    "holdout_split",
    "load_buffer",
    "scan_buffer_rows",
]

BUFFER_SCHEMA_VERSION = FEATURE_SCHEMA_VERSION
BUFFER_FEATURE_DIM = len(FEATURE_NAMES)


def load_buffer(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load a schema 2.0 training matrix and per-target arrays from JSONL buffer."""
    if not path.is_file():
        msg = f"Buffer JSONL not found: {path}"
        raise FileNotFoundError(msg)
    features: list[list[float]] = []
    target_rows: dict[str, list[float]] = {key: [] for key in TARGET_KEYS}
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
            _validate_buffer_row(row, path=path, line_no=line_no)
            row_features = row["features"]
            row_targets = row["targets"]
            features.append([float(value) for value in row_features])
            for key in TARGET_KEYS:
                target_rows[key].append(float(row_targets[key]))
    if not features:
        msg = f"No training samples found in {path}"
        raise ValueError(msg)
    feature_matrix = np.asarray(features, dtype=float)
    targets = {
        key: np.asarray(values, dtype=float) for key, values in target_rows.items()
    }
    return feature_matrix, targets


def scan_buffer_rows(path: Path) -> dict[str, Any]:
    """Summarize buffer JSONL rows without building training matrices."""
    if not path.is_file():
        msg = f"Buffer JSONL not found: {path}"
        raise FileNotFoundError(msg)
    valid_rows = 0
    invalid_rows = 0
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
                _validate_buffer_row(row, path=path, line_no=line_no)
            except ValueError:
                invalid_rows += 1
                continue
            valid_rows += 1
            schema = str(row["feature_schema_version"])
            schema_versions[schema] = schema_versions.get(schema, 0) + 1
            dim = len(row["features"])
            feature_dims[dim] = feature_dims.get(dim, 0) + 1
    return {
        "path": str(path.resolve()),
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
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
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Split features and targets into train and hold-out sets."""
    n_rows = int(feature_matrix.shape[0])
    if n_rows < 2:
        msg = "need at least two samples for hold-out split"
        raise ValueError(msg)
    rng = np.random.default_rng(random_state)
    indices = np.arange(n_rows)
    rng.shuffle(indices)
    test_size = max(1, int(round(n_rows * test_fraction)))
    test_size = min(test_size, n_rows - 1)
    test_indices = np.sort(indices[:test_size])
    train_indices = np.sort(indices[test_size:])
    train_targets = {key: targets[key][train_indices] for key in TARGET_KEYS}
    test_targets = {key: targets[key][test_indices] for key in TARGET_KEYS}
    return (
        feature_matrix[train_indices],
        train_targets,
        feature_matrix[test_indices],
        test_targets,
    )


def _validate_buffer_row(row: object, *, path: Path, line_no: int) -> None:
    if not isinstance(row, dict):
        msg = f"Invalid row format at {path}:{line_no}: expected JSON object"
        raise ValueError(msg)
    schema = row.get("feature_schema_version")
    if schema != BUFFER_SCHEMA_VERSION:
        msg = (
            f"Unsupported feature_schema_version at {path}:{line_no}: "
            f"expected {BUFFER_SCHEMA_VERSION!r}, got {schema!r}"
        )
        raise ValueError(msg)
    row_features = row.get("features")
    if not isinstance(row_features, list) or not row_features:
        msg = f"Invalid features at {path}:{line_no}"
        raise ValueError(msg)
    if len(row_features) != BUFFER_FEATURE_DIM:
        msg = (
            f"Invalid feature dimension at {path}:{line_no}: "
            f"expected {BUFFER_FEATURE_DIM}, got {len(row_features)}"
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

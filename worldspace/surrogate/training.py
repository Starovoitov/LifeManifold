"""Shared training data loading and hold-out split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldspace.surrogate.model import TARGET_KEYS

__all__ = ["holdout_split", "load_buffer"]


def load_buffer(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load training matrix and per-target arrays from JSONL buffer."""
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
            row_features = row.get("features")
            row_targets = row.get("targets")
            if not isinstance(row_features, list) or not isinstance(row_targets, dict):
                msg = f"Invalid row format at {path}:{line_no}"
                raise ValueError(msg)
            features.append([float(v) for v in row_features])
            for key in TARGET_KEYS:
                if key not in row_targets:
                    msg = f"Missing target {key!r} at {path}:{line_no}"
                    raise ValueError(msg)
                target_rows[key].append(float(row_targets[key]))
    if not features:
        msg = f"No training samples found in {path}"
        raise ValueError(msg)
    feature_matrix = np.asarray(features, dtype=float)
    targets = {k: np.asarray(v, dtype=float) for k, v in target_rows.items()}
    return feature_matrix, targets


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

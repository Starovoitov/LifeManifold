"""Synthetic JSONL buffer rows for surrogate training tests and smoke."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldspace.illuminators.evaluation import extinction_probability
from worldspace.surrogate.buffer import buffer_record
from worldspace.surrogate.feature_extractor import FEATURE_SCHEMA_VERSION

__all__ = ["write_synthetic_buffer"]


def write_synthetic_buffer(
    path: Path | str,
    *,
    n_samples: int,
    seed: int = 42,
) -> None:
    """Write deterministic learnable (features, targets) rows for hold-out tests."""
    if n_samples < 1:
        msg = "n_samples must be >= 1"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for _ in range(n_samples):
        features = _random_features(rng)
        targets = _targets_from_features(features)
        rows.append(
            buffer_record(
                features=features,
                targets=targets,
                emitter_type="synthetic",
                feature_schema_version=FEATURE_SCHEMA_VERSION,
            )
        )
    with target.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _random_features(rng: np.random.Generator) -> np.ndarray:
    return np.array(
        [
            float(rng.uniform(0.05, 0.95)),
            float(rng.uniform(0.05, 0.95)),
            float(rng.uniform(0.0, 0.2)),
            float(rng.uniform(0.0, 0.15)),
            float(rng.uniform(0.0, 0.25)),
            float(rng.integers(8, 64)),
            float(rng.integers(200, 400)),
            float(rng.integers(0, 2**31)),
        ],
        dtype=float,
    )


def _targets_from_features(features: np.ndarray) -> dict[str, float]:
    (
        birth_density,
        survival_density,
        noise,
        regen,
        predation,
        grid_size,
        steps,
        _seed,
    ) = features.tolist()
    stability = float(
        np.clip(0.35 * birth_density + 0.25 * survival_density + 0.1 * noise, 0, 1)
    )
    diversity = float(
        np.clip(0.30 * survival_density + 0.20 * regen + 0.15 * predation, 0, 1)
    )
    oscillation_score = float(np.clip(0.40 * noise + 0.20 * predation, 0, 1))
    topology_interface_index = float(
        np.clip(0.25 * birth_density + 0.35 * grid_size / 64.0, 0, 1)
    )
    topology_window_heterogeneity = float(
        np.clip(0.30 * survival_density + 0.10 * steps / 400.0, 0, 1)
    )
    final_density = float(np.clip(0.40 * regen + 0.30 * (1.0 - predation), 0, 1))
    early_extinction_prob = extinction_probability(final_density)
    return {
        "stability": stability,
        "diversity": diversity,
        "oscillation_score": oscillation_score,
        "topology_interface_index": topology_interface_index,
        "topology_window_heterogeneity": topology_window_heterogeneity,
        "final_density": final_density,
        "early_extinction_prob": early_extinction_prob,
    }

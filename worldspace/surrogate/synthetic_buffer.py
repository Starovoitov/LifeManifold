"""Synthetic JSONL buffer rows for surrogate training tests and smoke."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldspace.illuminators.evaluation import extinction_probability
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec
from worldspace.surrogate.buffer import buffer_record, world_spec_dict_for_buffer
from worldspace.surrogate.feature_extractor import FEATURE_SCHEMA_VERSION
from worldspace.surrogate.genome_features import FEATURE_DIM
from worldspace.specs.world_param_bounds import (
    NOISE_MAX,
    PREDATION_MAX,
    RESOURCE_REGEN_MAX,
)

# v1 synthetic caps; coefficients below assume these effective magnitudes.
_SYNTH_REGEN_MAX = 0.15
_SYNTH_PREDATION_MAX = 0.25

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
        world_spec = world_spec_dict_for_buffer(_world_spec_from_features(features))
        rows.append(
            buffer_record(
                features=features,
                targets=targets,
                emitter_type="synthetic",
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                world_spec=world_spec,
            )
        )
    with target.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _random_features(rng: np.random.Generator) -> np.ndarray:
    birth_bits = rng.integers(0, 2, size=9, dtype=np.int64)
    survival_bits = rng.integers(0, 2, size=9, dtype=np.int64)
    noise = float(rng.uniform(0.0, NOISE_MAX))
    resource_regen = float(rng.uniform(0.0, RESOURCE_REGEN_MAX))
    predation = float(rng.uniform(0.0, PREDATION_MAX))
    return np.array(
        [
            *[float(value) for value in birth_bits],
            *[float(value) for value in survival_bits],
            noise,
            resource_regen,
            predation,
        ],
        dtype=float,
    )


def _targets_from_features(features: np.ndarray) -> dict[str, float]:
    if features.shape != (FEATURE_DIM,):
        msg = f"expected feature vector shape ({FEATURE_DIM},), got {features.shape!r}"
        raise ValueError(msg)
    birth = features[0:9]
    survival = features[9:18]
    noise = float(features[18])
    resource_regen = float(features[19])
    predation = float(features[20])
    birth_density = float(np.mean(birth))
    survival_density = float(np.mean(survival))
    regen = _v1_scaled_regen(resource_regen)
    pred = _v1_scaled_predation(predation)
    stability = float(
        np.clip(
            0.35 * birth_density + 0.25 * survival_density + 0.10 * noise,
            0.0,
            1.0,
        )
    )
    diversity = float(
        np.clip(
            0.30 * survival_density + 0.20 * regen + 0.15 * pred,
            0.0,
            1.0,
        )
    )
    oscillation_score = float(np.clip(0.40 * noise + 0.20 * pred, 0.0, 1.0))
    topology_interface_index = float(
        np.clip(
            0.20 * birth[3] + 0.25 * birth[5] + 0.15 * survival[2] + 0.10 * regen,
            0.0,
            1.0,
        )
    )
    topology_window_heterogeneity = float(
        np.clip(
            0.25 * survival[4] + 0.20 * survival[7] + 0.15 * birth[1] + 0.10 * pred,
            0.0,
            1.0,
        )
    )
    final_density = float(np.clip(0.40 * regen + 0.30 * (1.0 - pred), 0.0, 1.0))
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


def _v1_scaled_regen(resource_regen: float) -> float:
    """Map full-range regen features to v1 synthetic magnitude for target formulas."""
    return float(resource_regen * (_SYNTH_REGEN_MAX / RESOURCE_REGEN_MAX))


def _v1_scaled_predation(predation: float) -> float:
    """Map full-range predation features to v1 synthetic magnitude for target formulas."""
    return float(predation * (_SYNTH_PREDATION_MAX / PREDATION_MAX))


def _world_spec_from_features(features: np.ndarray) -> WorldSpec:
    """Rebuild a ``WorldSpec`` from one genome-aligned feature vector."""
    birth = [index for index, value in enumerate(features[0:9]) if value >= 0.5]
    survival = [index for index, value in enumerate(features[9:18]) if value >= 0.5]
    if not birth:
        birth = [0]
    if not survival:
        survival = [0]
    return WorldSpec(
        birth=birth,
        survival=survival,
        noise=float(features[18]),
        resource_regen=float(features[19]),
        predation=float(features[20]),
        cell_types=list(CANONICAL_CELL_TYPES),
        grid_size=30,
        steps=220,
        seed=0,
    )

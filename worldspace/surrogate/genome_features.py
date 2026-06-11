"""Genome-aligned feature encoding shared with MAP-Elites genetics."""

from __future__ import annotations

import numpy as np

from worldspace.specs.spec import WorldSpec
from worldspace.specs.world_param_bounds import RULE_INDEX_COUNT

FEATURE_DIM = 21
FEATURE_DIM_V21 = 24

__all__ = [
    "FEATURE_DIM",
    "FEATURE_DIM_V21",
    "encode_world_spec_features",
    "encode_world_spec_features_v21",
    "rule_count_overlap_features",
]


def encode_world_spec_features(spec: WorldSpec) -> np.ndarray:
    """Return genome-aligned feature vector with shape ``(21,)``."""
    birth_mask = [
        1.0 if index in set(spec.birth) else 0.0 for index in range(RULE_INDEX_COUNT)
    ]
    survival_mask = [
        1.0 if index in set(spec.survival) else 0.0 for index in range(RULE_INDEX_COUNT)
    ]
    tail = [float(spec.noise), float(spec.resource_regen), float(spec.predation)]
    return np.asarray(birth_mask + survival_mask + tail, dtype=np.float64)


def rule_count_overlap_features(spec: WorldSpec) -> tuple[float, float, float]:
    """Normalized birth count, survival count, and birth∩survival overlap."""
    birth = set(spec.birth)
    survival = set(spec.survival)
    scale = float(RULE_INDEX_COUNT)
    return (
        len(birth) / scale,
        len(survival) / scale,
        len(birth & survival) / scale,
    )


def encode_world_spec_features_v21(spec: WorldSpec) -> np.ndarray:
    """Return schema 2.1 feature vector with shape ``(24,)``."""
    base = encode_world_spec_features(spec)
    counts = rule_count_overlap_features(spec)
    return np.concatenate([base, np.asarray(counts, dtype=np.float64)])

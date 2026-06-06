"""Genome-aligned feature encoding shared with MAP-Elites genetics."""

from __future__ import annotations

import numpy as np

from worldspace.specs.spec import WorldSpec
from worldspace.specs.world_param_bounds import RULE_INDEX_COUNT

FEATURE_DIM = 21

__all__ = ["FEATURE_DIM", "encode_world_spec_features"]


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

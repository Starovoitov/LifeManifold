"""Deterministic feature extraction from canonicalized world specs."""

from __future__ import annotations

import numpy as np

from worldspace.specs.spec import WorldSpec
from worldspace.surrogate.genome_features import encode_world_spec_features

FEATURE_SCHEMA_VERSION = "2.0"
FEATURE_NAMES: tuple[str, ...] = (
    tuple(f"birth_{index}" for index in range(9))
    + tuple(f"survival_{index}" for index in range(9))
    + (
        "noise",
        "resource_regen",
        "predation",
    )
)

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "extract",
]


def extract(spec: WorldSpec) -> np.ndarray:
    """Return deterministic numeric features from a canonicalized ``WorldSpec``."""
    _require_canonical_seed(spec)
    return encode_world_spec_features(spec)


def _require_canonical_seed(spec: WorldSpec) -> None:
    from worldspace.illuminators.evaluation import canonical_seed

    expected = canonical_seed(spec)
    if spec.seed != expected:
        msg = (
            "feature_extractor.extract requires canonicalized WorldSpec: "
            f"seed={spec.seed}, expected={expected}"
        )
        raise ValueError(msg)

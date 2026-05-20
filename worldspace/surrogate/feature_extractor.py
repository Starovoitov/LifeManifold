"""Deterministic feature extraction from canonicalized world specs."""

from __future__ import annotations

import numpy as np

from worldspace.illuminators.evaluation import canonical_seed
from worldspace.specs.spec import WorldSpec

FEATURE_SCHEMA_VERSION = "1.0"

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "extract",
]


def extract(spec: WorldSpec) -> np.ndarray:
    """Return deterministic numeric features from a canonicalized ``WorldSpec``."""
    _require_canonical_seed(spec)
    birth_density = _rule_density(spec.birth)
    survival_density = _rule_density(spec.survival)
    return np.array(
        [
            birth_density,
            survival_density,
            float(spec.noise),
            float(spec.resource_regen),
            float(spec.predation),
            float(spec.grid_size),
            float(spec.steps),
            float(spec.seed),
        ],
        dtype=float,
    )


def _rule_density(rule: list[int]) -> float:
    if not rule:
        return 0.0
    return float(sum(rule) / (8.0 * len(rule)))


def _require_canonical_seed(spec: WorldSpec) -> None:
    expected = canonical_seed(spec)
    if spec.seed != expected:
        msg = (
            "feature_extractor.extract requires canonicalized WorldSpec: "
            f"seed={spec.seed}, expected={expected}"
        )
        raise ValueError(msg)

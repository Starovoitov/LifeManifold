"""Shared ``WorldSpec`` scalar bounds (genetic genome, LLM patch, random-walk clamps)."""

from __future__ import annotations

import numpy as np

RULE_INDEX_COUNT = 9
RULE_BIT_MIN = 0.0
RULE_BIT_MAX = 1.0

NOISE_MIN = 0.0
NOISE_MAX = 0.2
RESOURCE_REGEN_MIN = 0.0
RESOURCE_REGEN_MAX = 0.5
PREDATION_MIN = 0.0
PREDATION_MAX = 1.0

# Genome tail order: noise, resource_regen, predation (see ``GeneticWorldGenerator``).
FLOAT_PARAM_BOUNDS: tuple[
    tuple[float, float], tuple[float, float], tuple[float, float]
] = (
    (NOISE_MIN, NOISE_MAX),
    (RESOURCE_REGEN_MIN, RESOURCE_REGEN_MAX),
    (PREDATION_MIN, PREDATION_MAX),
)

__all__ = [
    "FLOAT_PARAM_BOUNDS",
    "NOISE_MAX",
    "NOISE_MIN",
    "PREDATION_MAX",
    "PREDATION_MIN",
    "RESOURCE_REGEN_MAX",
    "RESOURCE_REGEN_MIN",
    "RULE_BIT_MAX",
    "RULE_BIT_MIN",
    "RULE_INDEX_COUNT",
    "clip_genome_float_params",
    "clip_scalar",
]


def clip_scalar(value: float, low: float, high: float) -> float:
    """Clip one float to ``[low, high]``."""
    return float(np.clip(value, low, high))


def clip_genome_float_params(
    genes: np.ndarray,
    *,
    start_index: int = 18,
) -> np.ndarray:
    """Clip the three float genes in place and return ``genes``."""
    for offset, (low, high) in enumerate(FLOAT_PARAM_BOUNDS):
        index = start_index + offset
        genes[index] = np.clip(genes[index], low, high)
    return genes

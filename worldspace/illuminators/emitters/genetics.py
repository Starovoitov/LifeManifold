"""Genome encode/decode and genetic operators for MAP-Elites emitters."""

from __future__ import annotations

import numpy as np

from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec
from worldspace.specs.world_param_bounds import (
    FLOAT_PARAM_BOUNDS,
    NOISE_MAX,
    NOISE_MIN,
    PREDATION_MAX,
    PREDATION_MIN,
    RESOURCE_REGEN_MAX,
    RESOURCE_REGEN_MIN,
    RULE_BIT_MAX,
    RULE_BIT_MIN,
    RULE_INDEX_COUNT,
    clip_genome_float_params,
    clip_scalar,
)

GENOME_SIZE = 21
_BIT_FLIP_SCALE = 5.0
_FLOAT_GENE_START = 18

__all__ = [
    "GENOME_SIZE",
    "decode_genome",
    "encode_world",
    "gaussian_mutate",
    "uniform_crossover",
]


def encode_world(world: WorldSpec) -> np.ndarray:
    """Encode a world as 9 birth bits, 9 survival bits, and 3 floats."""
    birth_mask = [1 if i in set(world.birth) else 0 for i in range(RULE_INDEX_COUNT)]
    survival_mask = [
        1 if i in set(world.survival) else 0 for i in range(RULE_INDEX_COUNT)
    ]
    tail = [float(world.noise), float(world.resource_regen), float(world.predation)]
    return np.asarray(birth_mask + survival_mask + tail, dtype=np.float64)


def decode_genome(
    genes: np.ndarray,
    *,
    grid_size: int,
    steps: int,
) -> WorldSpec:
    """Decode a genome vector into a ``WorldSpec`` (``seed`` left at 0)."""
    vals = np.asarray(genes, dtype=np.float64)
    birth_mask = np.rint(np.clip(vals[:9], RULE_BIT_MIN, RULE_BIT_MAX)).astype(np.int8)
    survival_mask = np.rint(np.clip(vals[9:18], RULE_BIT_MIN, RULE_BIT_MAX)).astype(
        np.int8
    )
    birth = [i for i in range(RULE_INDEX_COUNT) if int(birth_mask[i]) == 1]
    survival = [i for i in range(RULE_INDEX_COUNT) if int(survival_mask[i]) == 1]
    if not birth:
        birth = [int(np.argmax(vals[:9]))]
    if not survival:
        survival = [int(np.argmax(vals[9:18]))]
    return WorldSpec(
        birth=sorted(set(birth)),
        survival=sorted(set(survival)),
        noise=clip_scalar(vals[18], NOISE_MIN, NOISE_MAX),
        resource_regen=clip_scalar(vals[19], RESOURCE_REGEN_MIN, RESOURCE_REGEN_MAX),
        predation=clip_scalar(vals[20], PREDATION_MIN, PREDATION_MAX),
        cell_types=CANONICAL_CELL_TYPES.copy(),
        neighborhood="moore",
        grid_size=grid_size,
        steps=steps,
        seed=0,
    )


def uniform_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform crossover over two parent genomes."""
    a = np.asarray(parent_a, dtype=np.float64)
    b = np.asarray(parent_b, dtype=np.float64)
    mask = rng.random(GENOME_SIZE) < 0.5
    return np.where(mask, a, b).astype(np.float64)


def gaussian_mutate(
    genes: np.ndarray,
    mutation_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mutate rule bits and float genes using ``mutation_scale``."""
    child = np.asarray(genes, dtype=np.float64).copy()
    flip_prob = float(
        np.clip(mutation_scale * _BIT_FLIP_SCALE, RULE_BIT_MIN, RULE_BIT_MAX)
    )
    for index in range(_FLOAT_GENE_START):
        if rng.random() < flip_prob:
            child[index] = RULE_BIT_MAX - child[index]
    child[_FLOAT_GENE_START:] += rng.normal(
        0.0, mutation_scale, size=len(FLOAT_PARAM_BOUNDS)
    )
    return clip_genome_float_params(child, start_index=_FLOAT_GENE_START)

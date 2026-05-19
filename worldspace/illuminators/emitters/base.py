"""Emitter protocol for MAP-Elites candidate generation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from worldspace.illuminators.archive import EliteMetadata, GridArchive
from worldspace.illuminators.scheduler import EmitterKind, TargetBin
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec

__all__ = [
    "CandidateEmitter",
    "EmitterOutput",
    "MapElitesEmitter",
    "strip_seed",
]


@dataclass(frozen=True)
class EmitterOutput:
    """Candidate world and lineage metadata before canonical seed assignment."""

    world_spec: WorldSpec
    metadata: EliteMetadata


def strip_seed(world_spec: WorldSpec) -> WorldSpec:
    """Return a copy with canonical ``cell_types`` and ``seed`` cleared for evaluation."""
    return replace(
        world_spec,
        cell_types=CANONICAL_CELL_TYPES.copy(),
        seed=0,
    )


class CandidateEmitter(Protocol):
    """Generate a candidate ``WorldSpec`` (without canonical seed) for one batch slot."""

    def emit(
        self,
        *,
        emitter_kind: EmitterKind,
        target: TargetBin,
        archive: GridArchive,
        rng: np.random.Generator,
        grid_size: int,
        steps: int,
    ) -> EmitterOutput:
        """Return a world spec and metadata for evaluation and archive insert."""
        ...


class MapElitesEmitter:
    """Dispatch batch slots to random or genetic emitters."""

    def __init__(self, *, mutation_scale: float = 0.02) -> None:
        from worldspace.illuminators.emitters.genetic_emitter import GeneticEmitter
        from worldspace.illuminators.emitters.random_emitter import RandomEmitter

        self._random = RandomEmitter()
        self._genetic = GeneticEmitter(
            mutation_scale=mutation_scale,
            random_emitter=self._random,
        )

    def emit(
        self,
        *,
        emitter_kind: EmitterKind,
        target: TargetBin,
        archive: GridArchive,
        rng: np.random.Generator,
        grid_size: int,
        steps: int,
    ) -> EmitterOutput:
        if emitter_kind == "genetic":
            return self._genetic.emit(
                target=target,
                archive=archive,
                rng=rng,
                grid_size=grid_size,
                steps=steps,
            )
        return self._random.emit(
            target=target,
            archive=archive,
            rng=rng,
            grid_size=grid_size,
            steps=steps,
        )

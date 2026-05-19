"""Emitter protocol for MAP-Elites candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from worldspace.illuminators.archive import EliteMetadata, GridArchive
from worldspace.illuminators.scheduler import EmitterKind, TargetBin
from worldspace.specs.spec import WorldSpec

__all__ = ["CandidateEmitter", "EmitterOutput"]


@dataclass(frozen=True)
class EmitterOutput:
    """Candidate world and lineage metadata before canonical seed assignment."""

    world_spec: WorldSpec
    metadata: EliteMetadata


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

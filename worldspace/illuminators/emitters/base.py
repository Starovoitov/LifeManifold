"""Emitter protocol for MAP-Elites candidate generation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from worldspace.illuminators.emitters.llm_emitter import LlmEmitter

import numpy as np

from worldspace.illuminators.archive import EliteMetadata, GridArchive
from worldspace.illuminators.scheduler import (
    EmitterKind,
    SchedulerConfig,
    TargetBin,
    surrogate_config_from_scheduler,
)
from worldspace.surrogate.types import SurrogateProtocol
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
    """Dispatch batch slots to random, genetic, or LLM emitters."""

    def __init__(
        self,
        *,
        mutation_scale: float = 0.02,
        scheduler: SchedulerConfig | None = None,
        llm_emitter: LlmEmitter | None = None,
        surrogate: SurrogateProtocol | None = None,
    ) -> None:
        from worldspace.generators.llm_config import load_llm_config
        from worldspace.illuminators.emitters.genetic_emitter import GeneticEmitter
        from worldspace.illuminators.emitters.llm_emitter import LlmEmitter
        from worldspace.illuminators.emitters.random_emitter import RandomEmitter

        self._random = RandomEmitter()
        self._genetic = GeneticEmitter(
            mutation_scale=mutation_scale,
            random_emitter=self._random,
        )
        llm_cfg = load_llm_config()
        if llm_emitter is not None:
            self._llm = llm_emitter
        elif scheduler is not None:
            if surrogate is None:
                from worldspace.surrogate import get_surrogate

                surrogate = get_surrogate(surrogate_config_from_scheduler(scheduler))
            effective_surrogate = surrogate
            self._llm = LlmEmitter(
                grid_resolution=scheduler.grid_resolution,
                scheduler=scheduler,
                surrogate=effective_surrogate,
                fallback_scale=llm_cfg.fallback_scale,
            )
        else:
            self._llm = LlmEmitter(
                grid_resolution=50,
                surrogate_mean=0.5,
                surrogate_uncertainty=1.0,
                fallback_scale=llm_cfg.fallback_scale,
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
        if emitter_kind == "llm":
            return self._llm.emit(
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

"""Stub emitters until full random / genetic / llm implementations (E4)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from worldspace.generators import RandomWorldGenerator
from worldspace.illuminators.archive import GridArchive, new_elite_metadata
from worldspace.illuminators.emitters.base import EmitterOutput
from worldspace.illuminators.scheduler import EmitterKind, TargetBin
from worldspace.specs.spec import CANONICAL_CELL_TYPES

__all__ = ["StubCandidateEmitter"]


class StubCandidateEmitter:
    """Random-world stub for all emitter kinds (genetic / llm defer to E4)."""

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
        del target, archive
        generator = RandomWorldGenerator(grid_size=grid_size, steps=steps)
        draw_seed = int(rng.integers(0, 2**31))
        spec = generator._make_world(seed=draw_seed)
        spec = replace(
            spec,
            cell_types=CANONICAL_CELL_TYPES.copy(),
            seed=0,
        )
        metadata = new_elite_metadata(
            generated_by=emitter_kind,
            emitter_type=emitter_kind,
            parent_id=None,
        )
        return EmitterOutput(world_spec=spec, metadata=metadata)

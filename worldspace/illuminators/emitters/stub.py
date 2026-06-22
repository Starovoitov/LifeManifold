"""Stub emitters until full random / genetic / llm implementations (E4)."""

from __future__ import annotations

import numpy as np

from worldspace.generators import RandomWorldGenerator
from worldspace.illuminators.archive import new_elite_metadata
from worldspace.illuminators.archive_protocol import ArchiveProtocol
from worldspace.illuminators.emitters.base import EmitterOutput, strip_seed
from worldspace.illuminators.scheduler import EmitterKind, TargetCell

__all__ = ["StubCandidateEmitter"]


class StubCandidateEmitter:
    """Random-world stub for all emitter kinds (genetic / llm defer to E4)."""

    def emit(
        self,
        *,
        emitter_kind: EmitterKind,
        target: TargetCell,
        archive: ArchiveProtocol,
        rng: np.random.Generator,
        grid_size: int,
        steps: int,
    ) -> EmitterOutput:
        del target, archive
        generator = RandomWorldGenerator(grid_size=grid_size, steps=steps)
        draw_seed = int(rng.integers(0, 2**31))
        spec = strip_seed(generator._make_world(seed=draw_seed))
        metadata = new_elite_metadata(
            generated_by=emitter_kind,
            emitter_type=emitter_kind,
            parent_id=None,
        )
        return EmitterOutput(world_spec=spec, metadata=metadata)

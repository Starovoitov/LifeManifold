"""Random MAP-Elites emitter (independent ``WorldSpec`` samples)."""

from __future__ import annotations

import numpy as np

from worldspace.generators import RandomWorldGenerator
from worldspace.illuminators.archive import new_elite_metadata
from worldspace.illuminators.archive_protocol import ArchiveProtocol
from worldspace.illuminators.emitters.base import EmitterOutput, strip_seed
from worldspace.illuminators.scheduler import TargetCell

__all__ = ["RandomEmitter"]


class RandomEmitter:
    """Generate a new random world using ``RandomWorldGenerator`` bounds."""

    def emit(
        self,
        *,
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
            generated_by="random",
            emitter_type="random",
            parent_id=None,
        )
        return EmitterOutput(world_spec=spec, metadata=metadata)

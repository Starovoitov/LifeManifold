"""Genetic MAP-Elites emitter: crossover, mutation, and parent selection."""

from __future__ import annotations

import numpy as np

from worldspace.illuminators.archive import ArchiveElite, new_elite_metadata
from worldspace.illuminators.archive_protocol import ArchiveProtocol
from worldspace.illuminators.emitters.archive_neighbors import (
    min_fitness_elite,
    occupied_neighbors,
)
from worldspace.illuminators.emitters.base import EmitterOutput, strip_seed
from worldspace.illuminators.emitters.genetics import (
    decode_genome,
    encode_world,
    gaussian_mutate,
    uniform_crossover,
)
from worldspace.illuminators.emitters.random_emitter import RandomEmitter
from worldspace.illuminators.scheduler import TargetCell

DEFAULT_MUTATION_SCALE = 0.02

__all__ = ["DEFAULT_MUTATION_SCALE", "GeneticEmitter"]


class GeneticEmitter:
    """Produce offspring from archive elites via uniform crossover and Gaussian mutation."""

    def __init__(
        self,
        *,
        mutation_scale: float = DEFAULT_MUTATION_SCALE,
        random_emitter: RandomEmitter | None = None,
    ) -> None:
        self._mutation_scale = float(mutation_scale)
        self._random = random_emitter or RandomEmitter()

    def emit(
        self,
        *,
        target: TargetCell,
        archive: ArchiveProtocol,
        rng: np.random.Generator,
        grid_size: int,
        steps: int,
    ) -> EmitterOutput:
        if archive.filled_count() == 0:
            return self._random_fallback(
                target=target,
                archive=archive,
                rng=rng,
                grid_size=grid_size,
                steps=steps,
            )
        parent1 = archive.get_cell(target.cell_id)
        if parent1 is None or parent1.world_spec is None:
            return self._random_fallback(
                target=target,
                archive=archive,
                rng=rng,
                grid_size=grid_size,
                steps=steps,
            )
        parent2 = _select_parent_two(target.cell_id, archive, rng)
        if parent2 is None or parent2.world_spec is None:
            return self._random_fallback(
                target=target,
                archive=archive,
                rng=rng,
                grid_size=grid_size,
                steps=steps,
            )
        genes1 = encode_world(parent1.world_spec)
        genes2 = encode_world(parent2.world_spec)
        child_genes = gaussian_mutate(
            uniform_crossover(genes1, genes2, rng),
            self._mutation_scale,
            rng,
        )
        spec = strip_seed(decode_genome(child_genes, grid_size=grid_size, steps=steps))
        parent_id = parent1.metadata.id if parent1.metadata is not None else None
        metadata = new_elite_metadata(
            generated_by="genetic",
            emitter_type="genetic",
            parent_id=parent_id,
        )
        return EmitterOutput(world_spec=spec, metadata=metadata)

    def _random_fallback(
        self,
        *,
        target: TargetCell,
        archive: ArchiveProtocol,
        rng: np.random.Generator,
        grid_size: int,
        steps: int,
    ) -> EmitterOutput:
        output = self._random.emit(
            target=target,
            archive=archive,
            rng=rng,
            grid_size=grid_size,
            steps=steps,
        )
        metadata = new_elite_metadata(
            generated_by="genetic",
            emitter_type="genetic",
            parent_id=None,
        )
        return EmitterOutput(world_spec=output.world_spec, metadata=metadata)


def _select_parent_two(
    cell_id: int,
    archive: ArchiveProtocol,
    rng: np.random.Generator,
) -> ArchiveElite | None:
    neighbors = occupied_neighbors(archive, cell_id)
    if neighbors:
        index = int(rng.integers(0, len(neighbors)))
        return neighbors[index]
    return min_fitness_elite(archive)

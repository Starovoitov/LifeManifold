"""Genetic MAP-Elites emitter: crossover, mutation, and parent selection."""

from __future__ import annotations

import numpy as np

from worldspace.illuminators.archive import (
    ArchiveElite,
    GridArchive,
    new_elite_metadata,
)
from worldspace.illuminators.emitters.base import EmitterOutput, strip_seed
from worldspace.illuminators.emitters.genetics import (
    decode_genome,
    encode_world,
    gaussian_mutate,
    uniform_crossover,
)
from worldspace.illuminators.emitters.random_emitter import RandomEmitter
from worldspace.illuminators.grid_neighbors import cardinal_neighbors_bounded
from worldspace.illuminators.scheduler import TargetBin

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
        target: TargetBin,
        archive: GridArchive,
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
        parent1 = archive.get(*target.bin)
        if parent1 is None or parent1.world_spec is None:
            return self._random_fallback(
                target=target,
                archive=archive,
                rng=rng,
                grid_size=grid_size,
                steps=steps,
            )
        parent2 = _select_parent_two(target.bin, archive, rng)
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
        target: TargetBin,
        archive: GridArchive,
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
    target_bin: tuple[int, int],
    archive: GridArchive,
    rng: np.random.Generator,
) -> ArchiveElite | None:
    neighbors = _occupied_cardinal_neighbors(target_bin, archive)
    if neighbors:
        index = int(rng.integers(0, len(neighbors)))
        return neighbors[index]
    return _min_fitness_elite(archive)


def _occupied_cardinal_neighbors(
    target_bin: tuple[int, int],
    archive: GridArchive,
) -> list[ArchiveElite]:
    i, j = target_bin
    resolution = archive.resolution
    occupied: list[ArchiveElite] = []
    for ni, nj in cardinal_neighbors_bounded(i, j, resolution):
        elite = archive.get(ni, nj)
        if elite is not None:
            occupied.append(elite)
    return occupied


def _min_fitness_elite(archive: GridArchive) -> ArchiveElite | None:
    best: ArchiveElite | None = None
    best_bin: tuple[int, int] | None = None
    best_fitness = float("inf")
    resolution = archive.resolution
    for i in range(resolution):
        for j in range(resolution):
            elite = archive.get(i, j)
            if elite is None:
                continue
            if elite.fitness < best_fitness or (
                elite.fitness == best_fitness
                and (best_bin is None or (i, j) < best_bin)
            ):
                best = elite
                best_bin = (i, j)
                best_fitness = elite.fitness
    return best

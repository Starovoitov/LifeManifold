"""Random and genetic emitters for dungeon MAP-Elites."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from worldspace.dungeons.archive import DungeonArchive, DungeonElite
from worldspace.dungeons.genetics import (
    crossover_dungeons,
    mutate_dungeon,
    random_dungeon,
)
from worldspace.dungeons.spec import DungeonSpec


@dataclass(frozen=True)
class DungeonTarget:
    cell_id: int
    bin: tuple[int, int]
    center: tuple[float, float]
    parent: DungeonElite | None


@dataclass(frozen=True)
class DungeonEmitterResult:
    spec: DungeonSpec
    parent_id: str | None
    emitter_type: str


def select_uniform_frontier(
    archive: DungeonArchive,
    rng: np.random.Generator,
) -> DungeonTarget:
    """Select an occupied frontier cell uniformly, or a random empty target."""
    occupied = [
        cell_id
        for cell_id in range(archive.n_cells)
        if not archive.is_empty_cell(cell_id)
    ]
    frontier = [
        cell_id
        for cell_id in occupied
        if any(
            archive.is_empty_cell(neighbor) for neighbor in archive.neighbors(cell_id)
        )
    ]
    choices = frontier or occupied
    if choices:
        cell_id = choices[int(rng.integers(0, len(choices)))]
        parent = archive.get_cell(cell_id)
    else:
        cell_id = int(rng.integers(0, archive.n_cells))
        parent = None
    return DungeonTarget(
        cell_id=cell_id,
        bin=archive.bin_from_cell_id(cell_id),
        center=archive.cell_center(cell_id),
        parent=parent,
    )


def emit_random(rng: np.random.Generator) -> DungeonEmitterResult:
    return DungeonEmitterResult(
        spec=random_dungeon(rng),
        parent_id=None,
        emitter_type="random",
    )


def emit_genetic(
    target: DungeonTarget,
    archive: DungeonArchive,
    rng: np.random.Generator,
) -> DungeonEmitterResult:
    if target.parent is None:
        return emit_random(rng)
    parent = target.parent
    elites = archive.elites()
    base = parent.spec
    if len(elites) > 1 and rng.random() < 0.5:
        mate = elites[int(rng.integers(0, len(elites)))]
        base = crossover_dungeons(parent.spec, mate.spec, rng)
    return DungeonEmitterResult(
        spec=mutate_dungeon(base, rng),
        parent_id=parent.candidate_id,
        emitter_type="genetic",
    )

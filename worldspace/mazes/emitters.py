"""Random and genetic emitters for maze MAP-Elites."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from worldspace.mazes.archive import MazeArchive, MazeElite
from worldspace.mazes.genetics import crossover_mazes, mutate_maze, random_maze
from worldspace.mazes.spec import MazeSpec


@dataclass(frozen=True)
class MazeTarget:
    cell_id: int
    bin: tuple[int, int]
    center: tuple[float, float]
    parent: MazeElite | None


@dataclass(frozen=True)
class MazeEmitterResult:
    spec: MazeSpec
    parent_id: str | None
    emitter_type: str


def select_uniform_frontier(
    archive: MazeArchive,
    rng: np.random.Generator,
) -> MazeTarget:
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
    return MazeTarget(
        cell_id=cell_id,
        bin=archive.bin_from_cell_id(cell_id),
        center=archive.cell_center(cell_id),
        parent=parent,
    )


def emit_random(rng: np.random.Generator) -> MazeEmitterResult:
    return MazeEmitterResult(
        spec=random_maze(rng),
        parent_id=None,
        emitter_type="random",
    )


def emit_genetic(
    target: MazeTarget,
    archive: MazeArchive,
    rng: np.random.Generator,
) -> MazeEmitterResult:
    if target.parent is None:
        return emit_random(rng)
    parent = target.parent
    elites = archive.elites()
    base = parent.spec
    if len(elites) > 1 and rng.random() < 0.5:
        mate = elites[int(rng.integers(0, len(elites)))]
        base = crossover_mazes(parent.spec, mate.spec, rng)
    return MazeEmitterResult(
        spec=mutate_maze(base, rng),
        parent_id=parent.candidate_id,
        emitter_type="genetic",
    )

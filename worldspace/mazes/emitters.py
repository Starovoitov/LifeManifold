"""Random and genetic emitters for maze MAP-Elites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from worldspace.mazes.archive import MazeArchive, MazeElite
from worldspace.mazes.genetics import crossover_mazes, mutate_maze, random_maze
from worldspace.mazes.spec import MazeSpec

MazeTargetSelection = Literal[
    "min_fitness_frontier",
    "uniform_frontier",
    "max_fitness_frontier",
]
DEFAULT_MAZE_TARGET_SELECTION: MazeTargetSelection = "uniform_frontier"


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
    return select_target_cell(
        archive,
        rng,
        target_selection="uniform_frontier",
    )


def select_target_cell(
    archive: MazeArchive,
    rng: np.random.Generator,
    *,
    target_selection: MazeTargetSelection = DEFAULT_MAZE_TARGET_SELECTION,
) -> MazeTarget:
    """Select a maze archive target under min / uniform / max frontier policy."""
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
    pool = frontier or occupied
    if not pool:
        cell_id = int(rng.integers(0, archive.n_cells))
        parent = None
    elif target_selection == "uniform_frontier":
        cell_id = pool[int(rng.integers(0, len(pool)))]
        parent = archive.get_cell(cell_id)
    elif target_selection == "min_fitness_frontier":
        cell_id = _extremal_fitness_cell(pool, archive, maximize=False)
        parent = archive.get_cell(cell_id)
    elif target_selection == "max_fitness_frontier":
        cell_id = _extremal_fitness_cell(pool, archive, maximize=True)
        parent = archive.get_cell(cell_id)
    else:
        msg = f"unknown target_selection {target_selection!r}"
        raise ValueError(msg)
    return MazeTarget(
        cell_id=cell_id,
        bin=archive.bin_from_cell_id(cell_id),
        center=archive.cell_center(cell_id),
        parent=parent,
    )


def _extremal_fitness_cell(
    cell_ids: list[int],
    archive: MazeArchive,
    *,
    maximize: bool,
) -> int:
    best: int | None = None
    best_fitness = float("-inf") if maximize else float("inf")
    for cell_id in cell_ids:
        elite = archive.get_cell(cell_id)
        if elite is None:
            continue
        better = (
            elite.fitness > best_fitness if maximize else elite.fitness < best_fitness
        )
        tied = elite.fitness == best_fitness and (best is None or cell_id < best)
        if better or tied:
            best = cell_id
            best_fitness = elite.fitness
    if best is None:
        msg = "frontier cells must contain elites"
        raise RuntimeError(msg)
    return best


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

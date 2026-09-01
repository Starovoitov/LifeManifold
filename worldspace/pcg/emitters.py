"""Random and one-tile genetic emitters plus frontier selectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from worldspace.pcg.archive import PcgArchive, PcgElite
from worldspace.pcg.spec import PcgSpec, PcgTask

PcgTargetSelection = Literal[
    "min_fitness_frontier",
    "uniform_frontier",
    "max_fitness_frontier",
]
DEFAULT_PCG_TARGET_SELECTION: PcgTargetSelection = "uniform_frontier"


@dataclass(frozen=True)
class PcgTarget:
    cell_id: int
    bin: tuple[int, int]
    center: tuple[float, float]
    parent: PcgElite | None


@dataclass(frozen=True)
class PcgEmitterResult:
    spec: PcgSpec
    parent_id: str | None
    emitter_type: str


def select_target_cell(
    archive: PcgArchive,
    rng: np.random.Generator,
    *,
    target_selection: PcgTargetSelection = DEFAULT_PCG_TARGET_SELECTION,
) -> PcgTarget:
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
        raise ValueError(f"unknown target_selection {target_selection!r}")
    return PcgTarget(
        cell_id=cell_id,
        bin=archive.bin_from_cell_id(cell_id),
        center=archive.cell_center(cell_id),
        parent=parent,
    )


def _extremal_fitness_cell(
    cell_ids: list[int],
    archive: PcgArchive,
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
        raise RuntimeError("frontier cells must contain elites")
    return best


def random_spec(task: PcgTask, rng: np.random.Generator) -> PcgSpec:
    grid = tuple(
        tuple(int(rng.integers(0, task.n_tiles)) for _ in range(task.cols))
        for _ in range(task.rows)
    )
    return PcgSpec.from_task_grid(task, grid)


def spec_from_sampled(task: PcgTask, content: object) -> PcgSpec:
    return PcgSpec.from_task_grid(task, content)


def mutate_one_tile(parent: PcgSpec, rng: np.random.Generator) -> PcgSpec:
    """Change exactly one cell to a different tile. No solvability repair."""
    row = int(rng.integers(0, parent.rows))
    col = int(rng.integers(0, parent.cols))
    current = parent.grid[row][col]
    choices = [tile for tile in range(parent.n_tiles) if tile != current]
    replacement = choices[int(rng.integers(0, len(choices)))]
    grid = [list(item) for item in parent.grid]
    grid[row][col] = replacement
    return PcgSpec(
        problem_name=parent.problem_name,
        rows=parent.rows,
        cols=parent.cols,
        n_tiles=parent.n_tiles,
        grid=grid,  # type: ignore[arg-type]
    )


def emit_random(
    task: PcgTask,
    rng: np.random.Generator,
    *,
    sampled: object | None = None,
) -> PcgEmitterResult:
    spec = (
        spec_from_sampled(task, sampled)
        if sampled is not None
        else random_spec(task, rng)
    )
    return PcgEmitterResult(spec=spec, parent_id=None, emitter_type="random")


def emit_genetic(
    target: PcgTarget,
    rng: np.random.Generator,
    task: PcgTask,
) -> PcgEmitterResult:
    if target.parent is None:
        return emit_random(task, rng)
    return PcgEmitterResult(
        spec=mutate_one_tile(target.parent.spec, rng),
        parent_id=target.parent.candidate_id,
        emitter_type="genetic",
    )

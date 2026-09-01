"""Random and genetic emitters plus frontier selectors for NAS-Bench-201."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from worldspace.nas201.archive import Nas201Archive, Nas201Elite
from worldspace.nas201.spec import OPERATIONS, N_EDGES, Nas201Spec, OpName

Nas201TargetSelection = Literal[
    "min_fitness_frontier",
    "uniform_frontier",
    "max_fitness_frontier",
]
DEFAULT_NAS201_TARGET_SELECTION: Nas201TargetSelection = "uniform_frontier"


@dataclass(frozen=True)
class Nas201Target:
    cell_id: int
    bin: tuple[int, int]
    center: tuple[float, float]
    parent: Nas201Elite | None


@dataclass(frozen=True)
class Nas201EmitterResult:
    spec: Nas201Spec
    parent_id: str | None
    emitter_type: str


def select_target_cell(
    archive: Nas201Archive,
    rng: np.random.Generator,
    *,
    target_selection: Nas201TargetSelection = DEFAULT_NAS201_TARGET_SELECTION,
) -> Nas201Target:
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
    return Nas201Target(
        cell_id=cell_id,
        bin=archive.bin_from_cell_id(cell_id),
        center=archive.cell_center(cell_id),
        parent=parent,
    )


def _extremal_fitness_cell(
    cell_ids: list[int],
    archive: Nas201Archive,
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


def random_spec(rng: np.random.Generator) -> Nas201Spec:
    ops = tuple(
        OPERATIONS[int(rng.integers(0, len(OPERATIONS)))] for _ in range(N_EDGES)
    )
    return Nas201Spec(ops=ops)  # type: ignore[arg-type]


def mutate_one_edge(parent: Nas201Spec, rng: np.random.Generator) -> Nas201Spec:
    """Change exactly one edge to a different operation (no graph repair)."""
    edge = int(rng.integers(0, N_EDGES))
    current = parent.ops[edge]
    choices = [index for index, op in enumerate(OPERATIONS) if op != current]
    replacement = OPERATIONS[choices[int(rng.integers(0, len(choices)))]]
    ops: list[OpName] = list(parent.ops)
    ops[edge] = replacement
    return Nas201Spec(ops=tuple(ops))  # type: ignore[arg-type]


def emit_random(rng: np.random.Generator) -> Nas201EmitterResult:
    return Nas201EmitterResult(
        spec=random_spec(rng),
        parent_id=None,
        emitter_type="random",
    )


def emit_genetic(
    target: Nas201Target,
    rng: np.random.Generator,
) -> Nas201EmitterResult:
    if target.parent is None:
        return emit_random(rng)
    return Nas201EmitterResult(
        spec=mutate_one_edge(target.parent.spec, rng),
        parent_id=target.parent.candidate_id,
        emitter_type="genetic",
    )

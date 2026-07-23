"""Deterministic planning and objective/measure evaluation for mazes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from worldspace.mazes.spec import GOAL, START, WALL, MazeSpec, Position


@dataclass(frozen=True)
class MazeEvaluation:
    fitness: float
    measures: tuple[float, float]
    solvable: bool
    shortest_path: int | None
    reachable_ratio: float
    branching_density: float


def shortest_path_length(spec: MazeSpec) -> int | None:
    """BFS shortest path length from S to G (4-connected)."""
    start = spec.position(START)
    goal = spec.position(GOAL)
    assert start is not None and goal is not None
    queue: deque[tuple[Position, int]] = deque([(start, 0)])
    seen = {start}
    while queue:
        position, distance = queue.popleft()
        if position == goal:
            return distance
        for neighbor in _neighbors(position):
            if neighbor in seen:
                continue
            if spec.tile_at(neighbor) == WALL:
                continue
            seen.add(neighbor)
            queue.append((neighbor, distance + 1))
    return None


def evaluate_maze(spec: MazeSpec) -> MazeEvaluation:
    """Return deterministic fitness and two normalized behavior descriptors."""
    path = shortest_path_length(spec)
    reachable, branching = _reachable_structure(spec)
    traversable = sum(tile != WALL for row in spec.rows for tile in row)
    reachable_ratio = float(len(reachable)) / float(traversable) if traversable else 0.0
    branching_density = float(branching) / float(len(reachable)) if reachable else 0.0
    path_measure = float(np.clip((float(path or 0) - 8.0) / 36.0, 0.0, 1.0))
    branching_measure = float(np.clip((branching_density - 0.15) / 0.55, 0.0, 1.0))
    if path is None:
        return MazeEvaluation(
            fitness=0.0,
            measures=(0.0, branching_measure),
            solvable=False,
            shortest_path=None,
            reachable_ratio=reachable_ratio,
            branching_density=branching_density,
        )
    # Prefer longer interesting paths with moderate branching (illumination).
    length_score = float(np.clip((float(path) - 8.0) / 28.0, 0.0, 1.0))
    fitness = float(
        np.clip(
            0.55 * length_score + 0.30 * branching_measure + 0.15 * reachable_ratio,
            0.0,
            1.0,
        )
    )
    return MazeEvaluation(
        fitness=fitness,
        measures=(path_measure, branching_measure),
        solvable=True,
        shortest_path=path,
        reachable_ratio=reachable_ratio,
        branching_density=branching_density,
    )


def _neighbors(position: Position) -> tuple[Position, ...]:
    row, column = position
    return (
        (row - 1, column),
        (row + 1, column),
        (row, column - 1),
        (row, column + 1),
    )


def _reachable_structure(spec: MazeSpec) -> tuple[set[Position], int]:
    start = spec.position(START)
    assert start is not None
    queue: deque[Position] = deque([start])
    seen = {start}
    branching = 0
    while queue:
        position = queue.popleft()
        open_neighbors = [
            neighbor
            for neighbor in _neighbors(position)
            if 0 <= neighbor[0] < spec.SIZE
            and 0 <= neighbor[1] < spec.SIZE
            and spec.tile_at(neighbor) != WALL
        ]
        if len(open_neighbors) >= 3:
            branching += 1
        for neighbor in open_neighbors:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
    return seen, branching

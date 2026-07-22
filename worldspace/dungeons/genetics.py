"""Random generation and validity-preserving dungeon variation."""

from __future__ import annotations

import numpy as np
from pydantic import ValidationError

from worldspace.dungeons.evaluation import shortest_path_length
from worldspace.dungeons.spec import (
    DOOR,
    FLOOR,
    GOAL,
    HAZARD,
    KEY,
    START,
    WALL,
    DungeonSpec,
)


def random_dungeon(rng: np.random.Generator) -> DungeonSpec:
    """Generate a solvable dungeon with a carved start-goal backbone."""
    size = DungeonSpec.SIZE
    grid = [[WALL] * size for _ in range(size)]
    for row in range(1, size - 1):
        for column in range(1, size - 1):
            grid[row][column] = FLOOR if rng.random() < 0.62 else WALL
    start = (int(rng.integers(1, 5)), int(rng.integers(1, 5)))
    goal = (
        int(rng.integers(size - 5, size - 1)),
        int(rng.integers(size - 5, size - 1)),
    )
    path = _carve_monotonic_path(grid, start, goal, rng)
    grid[start[0]][start[1]] = START
    grid[goal[0]][goal[1]] = GOAL
    for row in range(1, size - 1):
        for column in range(1, size - 1):
            if grid[row][column] == FLOOR and rng.random() < 0.06:
                grid[row][column] = HAZARD
    if len(path) >= 8 and rng.random() < 0.5:
        door_index = max(4, len(path) * 2 // 3)
        key_index = max(2, door_index // 2)
        key_position = path[key_index]
        door_position = path[door_index]
        if key_position not in (start, goal) and door_position not in (start, goal):
            grid[key_position[0]][key_position[1]] = KEY
            grid[door_position[0]][door_position[1]] = DOOR
    spec = DungeonSpec(rows=tuple("".join(row) for row in grid))
    assert shortest_path_length(spec) is not None
    return spec


def mutate_dungeon(
    parent: DungeonSpec,
    rng: np.random.Generator,
    *,
    edits: int = 4,
) -> DungeonSpec:
    """Apply local edits, returning the parent if validity/solvability is lost."""
    grid = [list(row) for row in parent.rows]
    mutable = [
        (row, column)
        for row in range(1, parent.SIZE - 1)
        for column in range(1, parent.SIZE - 1)
        if grid[row][column] not in (START, GOAL, KEY, DOOR)
    ]
    for _ in range(max(1, edits)):
        row, column = mutable[int(rng.integers(0, len(mutable)))]
        grid[row][column] = str(rng.choice((WALL, FLOOR, FLOOR, HAZARD)))
    return _validated_or_parent(grid, parent)


def crossover_dungeons(
    first: DungeonSpec,
    second: DungeonSpec,
    rng: np.random.Generator,
) -> DungeonSpec:
    """Combine row bands and keep the child only when structurally valid."""
    cut = int(rng.integers(2, first.SIZE - 2))
    rows = list(first.rows[:cut] + second.rows[cut:])
    # Preserve one coherent set of special tiles from the first parent.
    for tile in (START, GOAL, KEY, DOOR):
        for row_index, row in enumerate(rows):
            rows[row_index] = row.replace(tile, FLOOR)
        position = first.position(tile)
        if position is not None:
            row = list(rows[position[0]])
            row[position[1]] = tile
            rows[position[0]] = "".join(row)
    return _validated_or_parent([list(row) for row in rows], first)


def _validated_or_parent(
    grid: list[list[str]],
    parent: DungeonSpec,
) -> DungeonSpec:
    try:
        child = DungeonSpec(rows=tuple("".join(row) for row in grid))
    except ValidationError:
        return parent
    return child if shortest_path_length(child) is not None else parent


def _carve_monotonic_path(
    grid: list[list[str]],
    start: tuple[int, int],
    goal: tuple[int, int],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    position = start
    path = [position]
    grid[position[0]][position[1]] = FLOOR
    while position != goal:
        row, column = position
        moves: list[tuple[int, int]] = []
        if row != goal[0]:
            moves.append((row + (1 if goal[0] > row else -1), column))
        if column != goal[1]:
            moves.append((row, column + (1 if goal[1] > column else -1)))
        position = moves[int(rng.integers(0, len(moves)))]
        grid[position[0]][position[1]] = FLOOR
        path.append(position)
    return path

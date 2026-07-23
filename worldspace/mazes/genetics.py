"""Random generation and validity-preserving maze variation."""

from __future__ import annotations

import numpy as np
from pydantic import ValidationError

from worldspace.mazes.evaluation import shortest_path_length
from worldspace.mazes.spec import FLOOR, GOAL, START, WALL, MazeSpec


def random_maze(rng: np.random.Generator) -> MazeSpec:
    """Generate a solvable maze via recursive-backtracker carve on odd cells."""
    size = MazeSpec.SIZE
    grid = [[WALL] * size for _ in range(size)]
    # Carve on odd coordinates for corridor structure.
    cells = [
        (row, column)
        for row in range(1, size - 1, 2)
        for column in range(1, size - 1, 2)
    ]
    start_cell = cells[int(rng.integers(0, len(cells)))]
    stack = [start_cell]
    visited = {start_cell}
    grid[start_cell[0]][start_cell[1]] = FLOOR
    while stack:
        row, column = stack[-1]
        options: list[tuple[int, int, int, int]] = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = row + dr, column + dc
            if 1 <= nr < size - 1 and 1 <= nc < size - 1 and (nr, nc) not in visited:
                options.append((nr, nc, row + dr // 2, column + dc // 2))
        if not options:
            stack.pop()
            continue
        nr, nc, wr, wc = options[int(rng.integers(0, len(options)))]
        grid[wr][wc] = FLOOR
        grid[nr][nc] = FLOOR
        visited.add((nr, nc))
        stack.append((nr, nc))
    # Place S/G on open cells far apart when possible.
    open_cells = [
        (r, c)
        for r in range(1, size - 1)
        for c in range(1, size - 1)
        if grid[r][c] == FLOOR
    ]
    start = open_cells[int(rng.integers(0, len(open_cells)))]
    goal_candidates = sorted(
        open_cells,
        key=lambda p: abs(p[0] - start[0]) + abs(p[1] - start[1]),
        reverse=True,
    )
    goal = goal_candidates[0]
    grid[start[0]][start[1]] = START
    grid[goal[0]][goal[1]] = GOAL
    # Optional random wall punches for loops / dead-ends diversity.
    for _ in range(int(rng.integers(2, 8))):
        r = int(rng.integers(1, size - 1))
        c = int(rng.integers(1, size - 1))
        if grid[r][c] == WALL and rng.random() < 0.7:
            grid[r][c] = FLOOR
    spec = MazeSpec(rows=tuple("".join(row) for row in grid))
    if shortest_path_length(spec) is None:
        return random_maze(rng)
    return spec


def mutate_maze(
    parent: MazeSpec,
    rng: np.random.Generator,
    *,
    edits: int = 4,
) -> MazeSpec:
    """Flip local wall/floor tiles; revert to parent if invalid/unsolvable."""
    grid = [list(row) for row in parent.rows]
    mutable = [
        (row, column)
        for row in range(1, parent.SIZE - 1)
        for column in range(1, parent.SIZE - 1)
        if grid[row][column] not in (START, GOAL)
    ]
    for _ in range(max(1, edits)):
        row, column = mutable[int(rng.integers(0, len(mutable)))]
        grid[row][column] = FLOOR if rng.random() < 0.67 else WALL
    return _validated_or_parent(grid, parent)


def crossover_mazes(
    first: MazeSpec,
    second: MazeSpec,
    rng: np.random.Generator,
) -> MazeSpec:
    """Combine row bands; keep child only when structurally valid and solvable."""
    cut = int(rng.integers(2, first.SIZE - 2))
    rows = list(first.rows[:cut] + second.rows[cut:])
    # Re-assert unique S/G from first parent if merge broke counts.
    flat = "".join(rows)
    if flat.count(START) != 1 or flat.count(GOAL) != 1:
        return first
    return _validated_or_parent([list(row) for row in rows], first)


def _validated_or_parent(grid: list[list[str]], parent: MazeSpec) -> MazeSpec:
    try:
        spec = MazeSpec(rows=tuple("".join(row) for row in grid))
    except ValidationError:
        return parent
    if shortest_path_length(spec) is None:
        return parent
    return spec

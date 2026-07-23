"""Fixed static feature schema for the maze surrogate (future filter arms)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from worldspace.mazes.evaluation import shortest_path_length
from worldspace.mazes.spec import FLOOR, WALL, MazeSpec

FEATURE_NAMES = (
    "wall_density",
    "floor_density",
    "shortest_path_norm",
    "open_degree_mean",
    "open_degree_std",
    "dead_end_density",
    "junction_density",
    "horizontal_symmetry",
    "vertical_symmetry",
    "quadrant_wall_0",
    "quadrant_wall_1",
    "quadrant_wall_2",
    "quadrant_wall_3",
    "start_goal_manhattan",
)


def extract_features(spec: MazeSpec) -> NDArray[np.float64]:
    """Extract deterministic, target-free structural features."""
    size = spec.SIZE
    total = float(size * size)
    flat = "".join(spec.rows)
    open_positions = [
        (row, column)
        for row in range(1, size - 1)
        for column in range(1, size - 1)
        if spec.rows[row][column] != WALL
    ]
    degrees = np.asarray(
        [
            sum(
                spec.rows[r][c] != WALL
                for r, c in (
                    (row - 1, column),
                    (row + 1, column),
                    (row, column - 1),
                    (row, column + 1),
                )
            )
            for row, column in open_positions
        ],
        dtype=np.float64,
    )
    path = shortest_path_length(spec)
    matrix = np.asarray([list(row) for row in spec.rows])
    half = size // 2
    quadrants = (
        matrix[:half, :half],
        matrix[:half, half:],
        matrix[half:, :half],
        matrix[half:, half:],
    )
    start = spec.position("S")
    goal = spec.position("G")
    assert start is not None and goal is not None
    max_manhattan = float(2 * (size - 3))
    values = (
        flat.count(WALL) / total,
        flat.count(FLOOR) / total,
        float(np.clip((float(path or 0) - 8.0) / 36.0, 0.0, 1.0)),
        float(np.mean(degrees) / 4.0) if degrees.size else 0.0,
        float(np.std(degrees) / 4.0) if degrees.size else 0.0,
        float(np.mean(degrees == 1)) if degrees.size else 0.0,
        float(np.mean(degrees >= 3)) if degrees.size else 0.0,
        float(np.mean(matrix == np.fliplr(matrix))),
        float(np.mean(matrix == np.flipud(matrix))),
        *(float(np.mean(q == WALL)) for q in quadrants),
        float(
            np.clip(
                (abs(start[0] - goal[0]) + abs(start[1] - goal[1])) / max_manhattan,
                0.0,
                1.0,
            )
        ),
    )
    return np.asarray(values, dtype=np.float64)

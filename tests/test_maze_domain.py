"""Golden tests for the self-contained maze QD domain."""

from __future__ import annotations

import unittest

import numpy as np
from pydantic import ValidationError

from worldspace.mazes.evaluation import evaluate_maze, shortest_path_length
from worldspace.mazes.features import FEATURE_NAMES, extract_features
from worldspace.mazes.genetics import crossover_mazes, mutate_maze, random_maze
from worldspace.mazes.spec import MazeSpec


def _spec(interior: list[str]) -> MazeSpec:
    rows = ["#" * 16]
    rows.extend("#" + row.ljust(14, "#")[:14] + "#" for row in interior)
    rows.extend("#" * 16 for _ in range(14 - len(interior)))
    rows.append("#" * 16)
    return MazeSpec(rows=tuple(rows))


class TestMazeDomain(unittest.TestCase):
    def test_schema_and_canonical_hash(self) -> None:
        maze = _spec(["S............G"])
        self.assertEqual(maze, MazeSpec.model_validate(maze.to_json_dict()))
        self.assertEqual(maze.candidate_hash(), maze.candidate_hash())
        with self.assertRaises(ValidationError):
            MazeSpec(rows=tuple("." * 16 for _ in range(16)))

    def test_shortest_path_open_corridor(self) -> None:
        maze = _spec(["S............G"])
        self.assertEqual(shortest_path_length(maze), 13)
        result = evaluate_maze(maze)
        self.assertTrue(result.solvable)
        self.assertEqual(result.shortest_path, 13)
        self.assertGreater(result.fitness, 0.0)

    def test_blocked_maze_has_zero_fitness(self) -> None:
        blocked = _spec(
            [
                "S.............",
                "##############",
                ".............G",
            ]
        )
        self.assertIsNone(shortest_path_length(blocked))
        result = evaluate_maze(blocked)
        self.assertFalse(result.solvable)
        self.assertEqual(result.fitness, 0.0)

    def test_features_are_fixed_and_bounded(self) -> None:
        values = extract_features(_spec(["S............G"]))
        self.assertEqual(values.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(values >= 0.0))
        self.assertTrue(np.all(values <= 1.0))

    def test_random_and_genetic_operators_preserve_validity(self) -> None:
        rng = np.random.default_rng(17)
        first = random_maze(rng)
        second = random_maze(rng)
        for _ in range(50):
            first = mutate_maze(first, rng)
            self.assertIsNotNone(shortest_path_length(first))
        child = crossover_mazes(first, second, rng)
        self.assertIsNotNone(shortest_path_length(child))


if __name__ == "__main__":
    unittest.main()

"""Golden tests for the self-contained dungeon QD domain."""

from __future__ import annotations

import unittest

import numpy as np
from pydantic import ValidationError

from worldspace.dungeons.evaluation import evaluate_dungeon, shortest_path_length
from worldspace.dungeons.features import FEATURE_NAMES, extract_features
from worldspace.dungeons.genetics import (
    crossover_dungeons,
    mutate_dungeon,
    random_dungeon,
)
from worldspace.dungeons.spec import DungeonSpec


def _spec(interior: list[str]) -> DungeonSpec:
    rows = ["#" * 16]
    rows.extend("#" + row.ljust(14, "#")[:14] + "#" for row in interior)
    rows.extend("#" * 16 for _ in range(14 - len(interior)))
    rows.append("#" * 16)
    return DungeonSpec(rows=tuple(rows))


class TestDungeonDomain(unittest.TestCase):
    def test_schema_and_canonical_hash(self) -> None:
        dungeon = _spec(["S............G"])
        self.assertEqual(dungeon, DungeonSpec.model_validate(dungeon.to_json_dict()))
        self.assertEqual(dungeon.candidate_hash(), dungeon.candidate_hash())
        with self.assertRaises(ValidationError):
            DungeonSpec(rows=tuple("." * 16 for _ in range(16)))

    def test_shortest_path_open_corridor(self) -> None:
        dungeon = _spec(["S............G"])
        self.assertEqual(shortest_path_length(dungeon), 13)
        result = evaluate_dungeon(dungeon, seed=3)
        self.assertTrue(result.solvable)
        self.assertEqual(result.shortest_path, 13)
        self.assertEqual(result.robustness, 1.0)
        self.assertGreater(result.fitness, 0.5)

    def test_key_is_required_before_door(self) -> None:
        valid = _spec(
            [
                "S.K.D........G",
                "..............",
            ]
        )
        self.assertIsNotNone(shortest_path_length(valid))
        blocked = _spec(
            [
                "S..D.........G",
                "##############",
                "K.............",
            ]
        )
        self.assertIsNone(shortest_path_length(blocked))

    def test_features_are_fixed_and_bounded(self) -> None:
        values = extract_features(_spec(["S............G"]))
        self.assertEqual(values.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(values >= 0.0))
        self.assertTrue(np.all(values <= 1.0))

    def test_random_and_genetic_operators_preserve_validity(self) -> None:
        rng = np.random.default_rng(17)
        first = random_dungeon(rng)
        second = random_dungeon(rng)
        for _ in range(50):
            first = mutate_dungeon(first, rng)
            self.assertIsNotNone(shortest_path_length(first))
        child = crossover_dungeons(first, second, rng)
        self.assertIsNotNone(shortest_path_length(child))

    def test_seeded_hazard_rollouts_are_reproducible(self) -> None:
        dungeon = _spec(
            [
                "S.H.H.H......G",
                "..............",
            ]
        )
        self.assertEqual(
            evaluate_dungeon(dungeon, seed=9),
            evaluate_dungeon(dungeon, seed=9),
        )


if __name__ == "__main__":
    unittest.main()

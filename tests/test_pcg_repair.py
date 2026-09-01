"""Named PCG repair is identity or structural_counts, not maze solvability_repair."""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from worldspace.pcg.emitters import random_spec
from worldspace.pcg.repair import (
    apply_repair,
    sokoban_astar_eligible,
)
from worldspace.pcg.smoke import run_pcg_smoke, seeded_initial_archive
from worldspace.pcg.spec import SOKOBAN_V0, ZELDA_V0, PcgSpec, hamming_tiles


def _solid_sokoban() -> PcgSpec:
    return PcgSpec.from_task_grid(SOKOBAN_V0, [[0] * 5] * 5)


def _solid_zelda() -> PcgSpec:
    return PcgSpec.from_task_grid(ZELDA_V0, [[0] * 11] * 7)


class _ToyEnv:
    def quality(self, contents: object) -> tuple[float, float, dict[str, object]]:
        grid = contents
        zeros = sum(tile == 0 for row in grid for tile in row)
        crates = sum(tile == 3 for row in grid for tile in row)
        players = sum(tile == 2 for row in grid for tile in row)
        info = {
            "players": players,
            "crates": crates,
            "targets": crates,
            "content": grid,
            "heuristic": -1,
            "solution": [0] * zeros,
        }
        quality = min(1.0, 0.05 * (players + 1))
        return 0.0, quality, info


class TestPcgRepairIdentity(unittest.TestCase):
    def test_identity_is_noop(self) -> None:
        spec = random_spec(SOKOBAN_V0, np.random.default_rng(11))
        repaired, meta = apply_repair(spec, "identity")
        self.assertEqual(repaired, spec)
        self.assertEqual(meta["tiles_changed"], 0)
        self.assertEqual(meta["repair_kind"], "identity")
        self.assertEqual(hamming_tiles(spec, repaired), 0)


class TestPcgRepairStructuralCounts(unittest.TestCase):
    def test_sokoban_one_player_crates_match_targets(self) -> None:
        repaired, meta = apply_repair(_solid_sokoban(), "structural_counts")
        grid = repaired.to_nested_list()
        players = sum(tile == 2 for row in grid for tile in row)
        crates = sum(tile == 3 for row in grid for tile in row)
        targets = sum(tile == 4 for row in grid for tile in row)
        self.assertEqual(players, 1)
        self.assertGreaterEqual(crates, 1)
        self.assertEqual(crates, targets)
        self.assertTrue(sokoban_astar_eligible(grid))
        self.assertEqual(meta["astar_eligible"], True)
        self.assertGreater(int(meta["tiles_changed"]), 0)

    def test_zelda_sprite_counts(self) -> None:
        repaired, _meta = apply_repair(_solid_zelda(), "structural_counts")
        grid = repaired.to_nested_list()
        self.assertEqual(sum(tile == 2 for row in grid for tile in row), 1)
        self.assertEqual(sum(tile == 3 for row in grid for tile in row), 1)
        self.assertEqual(sum(tile == 4 for row in grid for tile in row), 1)
        self.assertEqual(sum(tile == 5 for row in grid for tile in row), 3)

    def test_deterministic(self) -> None:
        spec = random_spec(SOKOBAN_V0, np.random.default_rng(12))
        first, _ = apply_repair(spec, "structural_counts")
        second, _ = apply_repair(spec, "structural_counts")
        self.assertEqual(first, second)

    def test_does_not_import_maze_repair(self) -> None:
        import worldspace.pcg.repair as repair_mod

        source = inspect.getsource(repair_mod)
        self.assertNotIn("mazes", source)
        self.assertNotIn("repair_solvable_mutation", source)
        self.assertNotIn("coerce_solvable_mutation", source)


class TestPcgSmokeWithNamedRepair(unittest.TestCase):
    def test_identity_default_still_runs_on_toy_env(self) -> None:
        from worldspace.pcg.descriptors import bin_edges_from_measures

        env = _ToyEnv()
        edges = bin_edges_from_measures(
            [(float(i), float(j)) for i in range(0, 26, 5) for j in range(0, 26, 5)],
            measure_names=("solution_length", "crates"),
            problem_name="sokoban-v0",
        )
        floor = seeded_initial_archive(
            env, SOKOBAN_V0, edges, seed=5, n_random=8, repair_kind="identity"
        )
        result, _archive = run_pcg_smoke(
            env,
            SOKOBAN_V0,
            edges,
            generator="genetic",
            selector="uniform_frontier",
            seed=6,
            evaluations=12,
            initial_archive=floor,
            repair_kind="identity",
        )
        self.assertEqual(result.structurally_invalid, 0)
        self.assertEqual(result.repair_kind, "identity")
        self.assertEqual(result.tiles_changed_mean, 0.0)

    def test_structural_counts_keeps_toy_eval_valid(self) -> None:
        from worldspace.pcg.descriptors import bin_edges_from_measures

        env = _ToyEnv()
        edges = bin_edges_from_measures(
            [(float(i), float(j)) for i in range(0, 26, 5) for j in range(0, 26, 5)],
            measure_names=("solution_length", "crates"),
            problem_name="sokoban-v0",
        )
        result, _archive = run_pcg_smoke(
            env,
            SOKOBAN_V0,
            edges,
            generator="random",
            selector="uniform_frontier",
            seed=8,
            evaluations=10,
            initial_random=4,
            repair_kind="structural_counts",
        )
        self.assertEqual(result.structurally_invalid, 0)
        self.assertEqual(result.repair_kind, "structural_counts")
        self.assertEqual(result.astar_eligible, result.proposals)


if __name__ == "__main__":
    unittest.main()

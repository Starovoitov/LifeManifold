"""PCG Benchmark genotype, miss-policy, mutation, and smoke gates."""

from __future__ import annotations

import unittest

import numpy as np

from worldspace.pcg.descriptors import PcgBinEdges, bin_edges_from_measures
from worldspace.pcg.emitters import mutate_one_tile, random_spec
from worldspace.pcg.env import _one_info_dict
from worldspace.pcg.evaluation import evaluate_payload, evaluate_spec
from worldspace.pcg.smoke import niche_jaccard, run_pcg_smoke, seeded_initial_archive
from worldspace.pcg.spec import SOKOBAN_V0, hamming_tiles, try_parse_grid


class _ToyEnv:
    def __init__(self) -> None:
        self.calls = 0

    def quality(self, contents: object) -> tuple[float, float, dict[str, object]]:
        self.calls += 1
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


def _edges() -> PcgBinEdges:
    return PcgBinEdges(
        resolution=10,
        measure_names=("solution_length", "crates"),
        axis0_min=0.0,
        axis0_max=20.0,
        axis1_min=0.0,
        axis1_max=10.0,
        n_samples=8,
        problem_name="sokoban-v0",
    )


class TestPcgSpec(unittest.TestCase):
    def test_canonical_json_roundtrip_and_hash(self) -> None:
        spec = random_spec(SOKOBAN_V0, np.random.default_rng(1))
        parsed = try_parse_grid(spec.to_nested_list(), SOKOBAN_V0)
        self.assertEqual(parsed, spec)
        self.assertEqual(len(spec.candidate_hash()), 16)
        self.assertTrue(spec.genotype_sha256().startswith(spec.candidate_hash()))
        self.assertNotIn(" ", spec.canonical_json())

    def test_unknown_tile_and_shape_do_not_parse(self) -> None:
        self.assertIsNone(try_parse_grid([[9] * 5] * 5, SOKOBAN_V0))
        self.assertIsNone(try_parse_grid([[0] * 4] * 5, SOKOBAN_V0))


class TestPcgEvaluation(unittest.TestCase):
    def test_invalid_payload_never_calls_evaluator(self) -> None:
        env = _ToyEnv()
        result = evaluate_payload("not-a-grid", env, _edges(), SOKOBAN_V0)
        self.assertFalse(result.structurally_valid)
        self.assertIsNone(result.fitness)
        self.assertEqual(env.calls, 0)

    def test_fitness_is_quality_not_diversity(self) -> None:
        env = _ToyEnv()
        spec = random_spec(SOKOBAN_V0, np.random.default_rng(2))
        result = evaluate_spec(spec, env, _edges(), SOKOBAN_V0)
        self.assertTrue(result.structurally_valid)
        self.assertGreater(result.fitness or 0.0, 0.0)
        self.assertLessEqual(result.fitness or 1.0, 1.0)
        self.assertEqual(
            result.measures[0],
            float(sum(tile == 0 for row in spec.grid for tile in row)),
        )
        self.assertEqual(env.calls, 1)


class TestPcgEnvInfoDict(unittest.TestCase):
    def test_one_content_dict_is_returned(self) -> None:
        self.assertEqual(_one_info_dict({"players": 1}, origin="info"), {"players": 1})

    def test_singleton_list_is_unwrapped(self) -> None:
        self.assertEqual(_one_info_dict([{"crates": 2}], origin="info"), {"crates": 2})

    def test_batch_or_non_dict_fails_closed(self) -> None:
        with self.assertRaises(TypeError):
            _one_info_dict([{"a": 1}, {"b": 2}], origin="info")
        with self.assertRaises(TypeError):
            _one_info_dict([1], origin="quality")


class TestPcgSmoke(unittest.TestCase):
    def test_mutation_changes_exactly_one_tile(self) -> None:
        parent = random_spec(SOKOBAN_V0, np.random.default_rng(3))
        child = mutate_one_tile(parent, np.random.default_rng(4))
        self.assertEqual(hamming_tiles(parent, child), 1)

    def test_genetic_smoke_has_headroom_and_selector_split(self) -> None:
        env = _ToyEnv()
        edges = bin_edges_from_measures(
            [(float(i), float(j)) for i in range(0, 26, 5) for j in range(0, 26, 5)],
            measure_names=("solution_length", "crates"),
            problem_name="sokoban-v0",
        )
        floor = seeded_initial_archive(env, SOKOBAN_V0, edges, seed=5, n_random=20)
        uniform, uniform_archive = run_pcg_smoke(
            env,
            SOKOBAN_V0,
            edges,
            generator="genetic",
            selector="uniform_frontier",
            seed=6,
            evaluations=80,
            initial_archive=floor,
        )
        minfit, minfit_archive = run_pcg_smoke(
            env,
            SOKOBAN_V0,
            edges,
            generator="genetic",
            selector="min_fitness_frontier",
            seed=7,
            evaluations=80,
            initial_archive=floor,
        )
        self.assertEqual(uniform.structurally_invalid, 0)
        self.assertLess(uniform.coverage, 0.95)
        jaccard = niche_jaccard(
            uniform_archive.occupied_bins(),
            minfit_archive.occupied_bins(),
        )
        self.assertLess(jaccard, 0.80)
        self.assertLess(minfit.coverage, 0.95)


if __name__ == "__main__":
    unittest.main()

"""Tests for B2 / RQ4 pyribs adapter (T1 genome, eval, BC, metrics, batch)."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
from ribs.archives import GridArchive

from worldspace.illuminators.evaluation import bin_index, evaluate_candidate
from worldspace.illuminators.pyribs_adapter import (
    ARCHIVE_DIMS,
    ARCHIVE_RANGES,
    GENOME_SIZE,
    PyribsEvalKnobs,
    coverage_pct,
    evaluate_solution,
    evaluate_solutions_batch,
    flat_cell_index,
    mean_best_fitness,
    measures_vector,
    mid_bounds_x0,
    solution_to_world_spec,
    world_spec_to_solution,
)
from worldspace.simulator_perf import SimulatorPerformanceOptions
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec

_FAST = PyribsEvalKnobs(
    grid_size=16,
    steps=200,
    resolution=50,
    early_extinction_step=200,
    enforce_min_steps=True,
    performance=SimulatorPerformanceOptions(parallel_eval=False),
)

_BASE_SPEC = WorldSpec(
    birth=[3],
    survival=[2, 3],
    noise=0.05,
    resource_regen=0.1,
    predation=0.2,
    cell_types=CANONICAL_CELL_TYPES.copy(),
    neighborhood="moore",
    grid_size=16,
    steps=200,
    seed=0,
)


class TestPyribsAdapterGenome(unittest.TestCase):
    def test_mid_bounds_x0_shape_and_values(self) -> None:
        x0 = mid_bounds_x0()
        self.assertEqual(x0.shape, (GENOME_SIZE,))
        self.assertTrue(np.allclose(x0[:18], 0.5))
        self.assertAlmostEqual(float(x0[18]), 0.1)
        self.assertAlmostEqual(float(x0[19]), 0.25)
        self.assertAlmostEqual(float(x0[20]), 0.5)

    def test_threshold_decode_at_half(self) -> None:
        theta = np.zeros(GENOME_SIZE, dtype=np.float64)
        theta[:18] = 0.5
        spec = solution_to_world_spec(
            theta, grid_size=16, steps=200, decode_mode="threshold"
        )
        self.assertEqual(len(spec.birth), 9)
        self.assertEqual(len(spec.survival), 9)

    def test_rint_vs_threshold_differ_at_half(self) -> None:
        theta = mid_bounds_x0().copy()
        rint_spec = solution_to_world_spec(
            theta, grid_size=16, steps=200, decode_mode="rint"
        )
        threshold_spec = solution_to_world_spec(
            theta, grid_size=16, steps=200, decode_mode="threshold"
        )
        self.assertEqual(len(rint_spec.birth), 1)
        self.assertEqual(len(threshold_spec.birth), 9)

    def test_bernoulli_decode_is_reproducible_with_rng(self) -> None:
        theta = mid_bounds_x0()
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        spec_a = solution_to_world_spec(
            theta, grid_size=16, steps=200, decode_mode="bernoulli", rng=rng_a
        )
        spec_b = solution_to_world_spec(
            theta, grid_size=16, steps=200, decode_mode="bernoulli", rng=rng_b
        )
        self.assertEqual(spec_a.birth, spec_b.birth)
        self.assertEqual(spec_a.survival, spec_b.survival)

    def test_float_genes_round_trip_within_clip(self) -> None:
        theta = world_spec_to_solution(_BASE_SPEC)
        theta[18] = 0.15
        theta[19] = 0.4
        theta[20] = 0.7
        restored = solution_to_world_spec(theta, grid_size=16, steps=200)
        again = world_spec_to_solution(restored)
        self.assertAlmostEqual(float(again[18]), 0.15)
        self.assertAlmostEqual(float(again[19]), 0.4)
        self.assertAlmostEqual(float(again[20]), 0.7)

    def test_rule_bits_stable_after_rint_decode(self) -> None:
        theta = world_spec_to_solution(_BASE_SPEC).copy()
        # Continuous relaxation near bits; decode should rint then stay binary.
        theta[:18] = np.clip(theta[:18] + 0.2, 0.0, 1.0)
        spec = solution_to_world_spec(theta, grid_size=16, steps=200)
        once = world_spec_to_solution(spec)
        twice_spec = solution_to_world_spec(once, grid_size=16, steps=200)
        twice = world_spec_to_solution(twice_spec)
        self.assertTrue(np.all((once[:18] == 0.0) | (once[:18] == 1.0)))
        np.testing.assert_array_equal(once[:18], twice[:18])


class TestPyribsAdapterEvalParity(unittest.TestCase):
    def test_evaluate_solution_matches_evaluate_candidate(self) -> None:
        theta = world_spec_to_solution(_BASE_SPEC)
        obj, measures, via_adapter = evaluate_solution(theta, knobs=_FAST)
        direct = evaluate_candidate(
            replace(_BASE_SPEC),
            resolution=_FAST.resolution,
            early_extinction_step=_FAST.early_extinction_step,
            enforce_min_steps=_FAST.enforce_min_steps,
            performance=_FAST.performance,
        )
        self.assertAlmostEqual(obj, float(direct.fitness))
        self.assertAlmostEqual(float(measures[0]), float(direct.measures["stability"]))
        self.assertAlmostEqual(float(measures[1]), float(direct.measures["diversity"]))
        self.assertEqual(via_adapter.early_extinct, direct.early_extinct)
        self.assertEqual(via_adapter.bin, direct.bin)


class TestPyribsAdapterBc(unittest.TestCase):
    def test_measures_vector_order(self) -> None:
        vec = measures_vector({"stability": 0.2, "diversity": 0.8})
        np.testing.assert_allclose(vec, [0.2, 0.8])

    def test_bin_index_agrees_with_pyribs_index_of(self) -> None:
        archive = GridArchive(
            solution_dim=GENOME_SIZE,
            dims=ARCHIVE_DIMS,
            ranges=list(ARCHIVE_RANGES),
        )
        points = [
            (0.0, 0.0),
            (1.0, 1.0),
            (0.5, 0.5),
            (0.02, 0.98),
            (0.999, 0.001),
            (0.25, 0.75),
        ]
        for stability, diversity in points:
            with self.subTest(s=stability, d=diversity):
                lm = flat_cell_index(stability, diversity, resolution=50)
                i, j = bin_index(stability, diversity, 50)
                self.assertEqual(lm, i * 50 + j)
                ribs_idx = int(
                    archive.index_of(
                        np.asarray([[stability, diversity]], dtype=np.float64)
                    )[0]
                )
                self.assertEqual(lm, ribs_idx)


class TestPyribsAdapterMetrics(unittest.TestCase):
    def test_empty_archive_metrics(self) -> None:
        archive = GridArchive(
            solution_dim=GENOME_SIZE,
            dims=ARCHIVE_DIMS,
            ranges=list(ARCHIVE_RANGES),
        )
        self.assertEqual(coverage_pct(archive), 0.0)
        self.assertIsNone(mean_best_fitness(archive))

    def test_coverage_and_mean_after_inserts(self) -> None:
        archive = GridArchive(
            solution_dim=GENOME_SIZE,
            dims=ARCHIVE_DIMS,
            ranges=list(ARCHIVE_RANGES),
        )
        solutions = np.zeros((3, GENOME_SIZE), dtype=np.float64)
        objectives = np.asarray([0.2, 0.4, 0.6], dtype=np.float64)
        measures = np.asarray(
            [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]],
            dtype=np.float64,
        )
        archive.add(solutions, objectives, measures)
        self.assertEqual(int(archive.stats.num_elites), 3)
        self.assertAlmostEqual(coverage_pct(archive), 100.0 * 3 / 2500)
        self.assertAlmostEqual(mean_best_fitness(archive) or -1.0, 0.4)


class TestPyribsAdapterBatch(unittest.TestCase):
    def test_batch_sequential_shapes(self) -> None:
        thetas = np.vstack(
            [
                world_spec_to_solution(_BASE_SPEC),
                mid_bounds_x0(),
            ]
        )
        # Second row must be a valid genome after decode; mid_bounds is fine.
        batch = evaluate_solutions_batch(thetas, knobs=_FAST, eval_pool=None)
        self.assertEqual(batch.objectives.shape, (2,))
        self.assertEqual(batch.measures.shape, (2, 2))
        self.assertEqual(len(batch.results), 2)
        self.assertTrue(np.all(np.isfinite(batch.objectives)))
        self.assertTrue(np.all(np.isfinite(batch.measures)))

        # First row matches single-eval path.
        obj, meas, _ = evaluate_solution(thetas[0], knobs=_FAST)
        self.assertAlmostEqual(float(batch.objectives[0]), obj)
        np.testing.assert_allclose(batch.measures[0], meas)


if __name__ == "__main__":
    unittest.main()

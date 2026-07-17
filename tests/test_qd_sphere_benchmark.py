"""Golden tests for supplementary Sphere/Rastrigin QD benchmarks."""

from __future__ import annotations

import unittest

import numpy as np

from worldspace.benchmarks.qd_sphere import (
    CLIP_BOUND,
    DEFAULT_SOLUTION_DIM,
    SPHERE_SHIFT,
    archive_ranges,
    clip_solution,
    linear_projection_measures,
    rastrigin_objective,
    sphere_objective,
)


class TestQDSphereBenchmark(unittest.TestCase):
    def test_sphere_optimum_and_worst(self) -> None:
        optimum = np.full(DEFAULT_SOLUTION_DIM, SPHERE_SHIFT)
        worst = np.full(DEFAULT_SOLUTION_DIM, -CLIP_BOUND)
        self.assertAlmostEqual(float(sphere_objective(optimum)), 100.0)
        self.assertAlmostEqual(float(sphere_objective(worst)), 0.0)

    def test_sphere_batch_matches_reference_formula(self) -> None:
        solutions = np.vstack(
            (
                np.zeros(DEFAULT_SOLUTION_DIM),
                np.full(DEFAULT_SOLUTION_DIM, SPHERE_SHIFT),
            )
        )
        actual = np.asarray(sphere_objective(solutions))
        worst = DEFAULT_SOLUTION_DIM * (-CLIP_BOUND - SPHERE_SHIFT) ** 2
        expected_zero = 100.0 * (1.0 - DEFAULT_SOLUTION_DIM * SPHERE_SHIFT**2 / worst)
        np.testing.assert_allclose(actual, [expected_zero, 100.0])

    def test_rastrigin_is_deterministic_and_bounded(self) -> None:
        zero = np.zeros(DEFAULT_SOLUTION_DIM)
        edge = np.full(DEFAULT_SOLUTION_DIM, CLIP_BOUND)
        outside = np.full(DEFAULT_SOLUTION_DIM, 99.0)
        self.assertAlmostEqual(float(rastrigin_objective(zero)), 100.0)
        self.assertGreaterEqual(float(rastrigin_objective(edge)), 0.0)
        self.assertLessEqual(float(rastrigin_objective(edge)), 100.0)
        self.assertAlmostEqual(
            float(rastrigin_objective(edge)),
            float(rastrigin_objective(outside)),
        )

    def test_linear_projection_measures_single_and_batch(self) -> None:
        solution = np.arange(DEFAULT_SOLUTION_DIM, dtype=np.float64) - 10.0
        clipped = clip_solution(solution)
        expected = np.asarray(
            [clipped[:10].sum(), clipped[10:].sum()],
            dtype=np.float64,
        )
        np.testing.assert_allclose(linear_projection_measures(solution), expected)
        batch = np.vstack((solution, np.zeros(DEFAULT_SOLUTION_DIM)))
        measures = linear_projection_measures(batch)
        self.assertEqual(measures.shape, (2, 2))
        np.testing.assert_allclose(measures[0], expected)
        np.testing.assert_allclose(measures[1], [0.0, 0.0])

    def test_archive_ranges(self) -> None:
        self.assertEqual(
            archive_ranges(DEFAULT_SOLUTION_DIM),
            ((-51.2, 51.2), (-51.2, 51.2)),
        )
        with self.assertRaises(ValueError):
            archive_ranges(19)


if __name__ == "__main__":
    unittest.main()

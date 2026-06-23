"""Tests for runtime Langton activity metric (langton_lambda_runtime)."""

from __future__ import annotations

import unittest
from dataclasses import replace


from worldspace import math as ws_math
from worldspace.simulator import run_world
from worldspace.specs.spec import WorldSpec


class TestLangtonLambdaRuntime(unittest.TestCase):
    def test_helper_clips_and_handles_zero_steps(self) -> None:
        self.assertEqual(ws_math.langton_lambda_runtime(0.0, 0), 0.0)
        self.assertAlmostEqual(ws_math.langton_lambda_runtime(1.5, 2), 0.75)

    def test_high_noise_increases_runtime_lambda(self) -> None:
        base = WorldSpec(
            birth=[3],
            survival=[2, 3],
            noise=0.0,
            resource_regen=0.02,
            predation=0.0,
            cell_types=["life", "food"],
            grid_size=32,
            steps=100,
            seed=7,
        )
        quiet = run_world(base, early_extinction_step=200)
        noisy = run_world(replace(base, noise=0.15), early_extinction_step=200)
        self.assertGreater(
            noisy.metrics.langton_lambda_runtime,
            quiet.metrics.langton_lambda_runtime,
        )

    def test_metric_in_world_metrics_vector(self) -> None:
        spec = WorldSpec(
            birth=[3],
            survival=[2, 3],
            noise=0.01,
            resource_regen=0.02,
            predation=0.1,
            cell_types=["life", "food"],
            grid_size=16,
            steps=50,
            seed=1,
        )
        metrics = run_world(spec, early_extinction_step=200).metrics
        vector = metrics.as_vector()
        self.assertEqual(vector.shape[0], 13)
        self.assertAlmostEqual(vector[-1], metrics.langton_lambda_runtime)
        self.assertGreaterEqual(metrics.langton_lambda_runtime, 0.0)
        self.assertLessEqual(metrics.langton_lambda_runtime, 1.0)


if __name__ == "__main__":
    unittest.main()

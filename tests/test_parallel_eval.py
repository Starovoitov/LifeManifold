"""Tests for parallel illuminator candidate evaluation."""

from __future__ import annotations

import unittest

import numpy as np

from worldspace.illuminators.archive import GridArchive
from worldspace.illuminators.evaluation import (
    evaluate_candidate,
    eval_result_from_simulation,
    simulate_candidate,
)
from worldspace.illuminators.parallel_eval import (
    ParallelEvalPool,
    evaluate_batch_parallel,
)
from worldspace.simulator_perf import SimulatorPerformanceOptions
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec


def _sample_spec(*, seed: int = 0) -> WorldSpec:
    return WorldSpec(
        birth=[1, 3],
        survival=[2, 3],
        noise=0.02,
        resource_regen=0.05,
        predation=0.1,
        cell_types=CANONICAL_CELL_TYPES.copy(),
        grid_size=8,
        steps=200,
        seed=seed,
    )


def _eval_tuple(result) -> tuple:
    return (
        result.fitness,
        result.bin,
        result.early_extinct,
        tuple(result.metrics.as_vector()),
        tuple(sorted(result.measures.items())),
    )


class TestParallelEval(unittest.TestCase):
    def test_evaluate_batch_parallel_empty(self) -> None:
        perf = SimulatorPerformanceOptions(parallel_eval=True)
        pool = ParallelEvalPool(2)
        try:
            self.assertEqual(
                evaluate_batch_parallel(
                    [],
                    early_extinction_step=200,
                    enforce_min_steps=True,
                    performance=perf,
                    workers=2,
                    eval_pool=pool,
                ),
                [],
            )
        finally:
            pool.terminate()
            pool.join()

    def test_evaluate_batch_parallel_matches_sequential(self) -> None:
        specs = [_sample_spec(seed=index) for index in range(3)]
        archive = GridArchive(5)
        perf = SimulatorPerformanceOptions(parallel_eval=True, parallel_workers=2)
        pool = ParallelEvalPool(2)
        try:
            parallel_outcomes = evaluate_batch_parallel(
                specs,
                early_extinction_step=200,
                enforce_min_steps=True,
                performance=perf,
                workers=2,
                eval_pool=pool,
            )
        finally:
            pool.terminate()
            pool.join()
        self.assertEqual(len(parallel_outcomes), 3)
        for spec, outcome in zip(specs, parallel_outcomes):
            sequential = evaluate_candidate(
                spec,
                resolution=5,
                archive=archive,
                early_extinction_step=200,
            )
            parallel = eval_result_from_simulation(
                outcome,
                resolution=5,
                archive=archive,
            )
            self.assertEqual(_eval_tuple(parallel), _eval_tuple(sequential))

    def test_parallel_workers_one_matches_simulate_candidate(self) -> None:
        spec = _sample_spec()
        perf = SimulatorPerformanceOptions(parallel_eval=True, parallel_workers=1)
        pool = ParallelEvalPool(1)
        try:
            outcomes = evaluate_batch_parallel(
                [spec],
                early_extinction_step=200,
                enforce_min_steps=True,
                performance=perf,
                workers=1,
                eval_pool=pool,
            )
        finally:
            pool.terminate()
            pool.join()
        expected = simulate_candidate(spec, early_extinction_step=200)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].fitness, expected.fitness)
        self.assertEqual(outcomes[0].early_extinct, expected.early_extinct)
        np.testing.assert_allclose(
            outcomes[0].metrics.as_vector(),
            expected.metrics.as_vector(),
        )


if __name__ == "__main__":
    unittest.main()

"""Unit tests for simulator performance options."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from worldspace.simulator_perf import (
    DEFAULT_SIMULATOR_PERFORMANCE,
    SimulatorPerformanceOptions,
    effective_llm_parallel_workers,
    effective_numba_enabled,
    effective_parallel_workers,
    resolve_simulator_performance,
    validate_simulator_performance,
)


class TestSimulatorPerformanceOptions(unittest.TestCase):
    def test_defaults_are_safe(self) -> None:
        self.assertEqual(DEFAULT_SIMULATOR_PERFORMANCE, SimulatorPerformanceOptions())
        self.assertFalse(DEFAULT_SIMULATOR_PERFORMANCE.numba_simulator)
        self.assertFalse(DEFAULT_SIMULATOR_PERFORMANCE.parallel_eval)
        self.assertFalse(DEFAULT_SIMULATOR_PERFORMANCE.verify_against_reference)
        self.assertTrue(DEFAULT_SIMULATOR_PERFORMANCE.numba_cache)

    def test_effective_numba_disabled_for_ca_step_trace(self) -> None:
        perf = SimulatorPerformanceOptions(numba_simulator=True)
        self.assertFalse(effective_numba_enabled(perf, ca_step_trace=True))
        self.assertTrue(effective_numba_enabled(perf, ca_step_trace=False))

    def test_effective_parallel_workers(self) -> None:
        perf = SimulatorPerformanceOptions(parallel_eval=True, parallel_workers=0)
        workers = effective_parallel_workers(perf, batch_size=4)
        self.assertGreaterEqual(workers, 1)
        self.assertLessEqual(workers, 4)
        capped = effective_parallel_workers(
            SimulatorPerformanceOptions(parallel_workers=8),
            batch_size=2,
        )
        self.assertEqual(capped, 2)

    def test_resolve_from_yaml_block(self) -> None:
        resolved = resolve_simulator_performance(
            {
                "numba_simulator": True,
                "parallel_workers": 4,
                "verify_against_reference": True,
            }
        )
        self.assertTrue(resolved.numba_simulator)
        self.assertFalse(resolved.parallel_eval)
        self.assertEqual(resolved.parallel_workers, 4)
        self.assertTrue(resolved.verify_against_reference)

    def test_validate_rejects_numba_with_parallel_eval(self) -> None:
        perf = SimulatorPerformanceOptions(
            numba_simulator=True,
            parallel_eval=True,
        )
        with self.assertRaises(ValueError) as ctx:
            validate_simulator_performance(perf)
        self.assertIn("numba_simulator and parallel_eval", str(ctx.exception))

    def test_effective_llm_parallel_workers(self) -> None:
        perf = SimulatorPerformanceOptions(llm_parallel_emit=True)
        self.assertEqual(
            effective_llm_parallel_workers(perf, llm_slot_count=10),
            10,
        )
        capped = SimulatorPerformanceOptions(
            llm_parallel_emit=True,
            llm_parallel_workers=4,
        )
        self.assertEqual(
            effective_llm_parallel_workers(capped, llm_slot_count=10),
            4,
        )
        self.assertEqual(
            effective_llm_parallel_workers(capped, llm_slot_count=2),
            2,
        )

    def test_resolve_llm_parallel_workers_from_yaml_and_env(self) -> None:
        resolved = resolve_simulator_performance(
            {"llm_parallel_emit": True, "llm_parallel_workers": 3}
        )
        self.assertEqual(resolved.llm_parallel_workers, 3)
        with mock.patch.dict(
            os.environ,
            {"LIFEMANIFOLD_LLM_PARALLEL_WORKERS": "5"},
            clear=False,
        ):
            env_resolved = resolve_simulator_performance({"llm_parallel_workers": 3})
        self.assertEqual(env_resolved.llm_parallel_workers, 5)

    def test_env_overrides_yaml(self) -> None:
        env = {
            "LIFEMANIFOLD_NUMBA_SIM": "0",
            "LIFEMANIFOLD_PARALLEL_EVAL": "1",
            "LIFEMANIFOLD_VERIFY_SIM": "0",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            resolved = resolve_simulator_performance({"numba_simulator": True})
        self.assertFalse(resolved.numba_simulator)
        self.assertTrue(resolved.parallel_eval)
        self.assertFalse(resolved.verify_against_reference)


if __name__ == "__main__":
    unittest.main()

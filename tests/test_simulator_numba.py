"""Tests for optional numba simulator path."""

from __future__ import annotations

import io
import unittest
from dataclasses import replace

import numpy as np

from tests.test_simulator_perf_golden import _GOLDEN_CASES, _BASE_SMOKE
from worldspace.metrics import METRICS_VECTOR_DIM
from worldspace.simulator import run_world
from worldspace.simulator_perf import SimulatorPerformanceOptions
from worldspace.specs.spec import WorldSpec

_NUMBA_PERF = SimulatorPerformanceOptions(numba_simulator=True)
_VERIFY_PERF = SimulatorPerformanceOptions(
    numba_simulator=True,
    verify_against_reference=True,
)


def _has_numba() -> bool:
    try:
        import numba  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(_has_numba(), "numba not installed (uv sync --group perf)")
class TestSimulatorNumba(unittest.TestCase):
    def test_numba_matches_golden_vectors(self) -> None:
        for case in _GOLDEN_CASES:
            with self.subTest(case=case.name):
                result = run_world(
                    case.spec,
                    early_extinction_step=case.early_extinction_step,
                    performance=_NUMBA_PERF,
                )
                got = result.metrics.as_vector()
                self.assertEqual(METRICS_VECTOR_DIM, got.shape[0])
                np.testing.assert_allclose(
                    got,
                    case.expected,
                    rtol=0.0,
                    atol=0.0,
                )

    def test_verify_against_reference(self) -> None:
        result = run_world(
            _BASE_SMOKE,
            early_extinction_step=200,
            performance=_VERIFY_PERF,
        )
        self.assertEqual(METRICS_VECTOR_DIM, result.metrics.as_vector().shape[0])

    def test_trace_forces_numpy_path(self) -> None:
        buf = io.StringIO()
        result = run_world(
            _BASE_SMOKE,
            early_extinction_step=200,
            ca_step_trace_file=buf,
            performance=_NUMBA_PERF,
        )
        self.assertGreater(len(buf.getvalue().strip().splitlines()), 0)
        self.assertEqual(METRICS_VECTOR_DIM, result.metrics.as_vector().shape[0])

    def test_grid_parity_short_run(self) -> None:
        spec = WorldSpec(
            birth=[3],
            survival=[2, 3],
            noise=0.01,
            resource_regen=0.02,
            predation=0.3,
            cell_types=["life", "food"],
            grid_size=16,
            steps=5,
            seed=7,
        )
        numpy_result = run_world(
            spec,
            early_extinction_step=200,
            performance=SimulatorPerformanceOptions(numba_simulator=False),
        )
        numba_result = run_world(
            spec,
            early_extinction_step=200,
            performance=_NUMBA_PERF,
        )
        np.testing.assert_array_equal(
            numpy_result.final_life,
            numba_result.final_life,
        )
        np.testing.assert_array_equal(
            numpy_result.final_food,
            numba_result.final_food,
        )
        self.assertEqual(numpy_result.early_extinct, numba_result.early_extinct)

    def test_numba_cache_warmup_smoke(self) -> None:
        spec = replace(_BASE_SMOKE, grid_size=32, steps=20)
        run_world(
            spec,
            early_extinction_step=200,
            performance=SimulatorPerformanceOptions(
                numba_simulator=True,
                numba_cache=True,
            ),
        )
        run_world(
            spec,
            early_extinction_step=200,
            performance=SimulatorPerformanceOptions(
                numba_simulator=True,
                numba_cache=False,
            ),
        )


if __name__ == "__main__":
    unittest.main()

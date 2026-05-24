"""Unit tests for surrogate uncertainty calibration (SA-5)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from worldspace.surrogate.calibration import (
    UncertaintyCalibrator,
    apply_calibrated_uncertainty,
    expected_calibration_error,
    fit_uncertainty_calibrator,
    load_uncertainty_calibration,
    save_uncertainty_calibration,
)
from worldspace.surrogate.surrogate import build_surrogate_facade
from worldspace.surrogate.types import SurrogatePrediction


class TestUncertaintyCalibrator(unittest.TestCase):
    def test_fit_is_monotonic(self) -> None:
        raw = np.linspace(0.05, 0.9, 40)
        errors = raw * 0.5 + 0.01
        calibrator = fit_uncertainty_calibrator(raw, errors)
        samples = [0.1, 0.3, 0.6, 0.85]
        calibrated = [calibrator.apply(value) for value in samples]
        self.assertEqual(calibrated, sorted(calibrated))

    def test_round_trip_pickle(self) -> None:
        calibrator = fit_uncertainty_calibrator(
            np.array([0.1, 0.3, 0.5, 0.8]),
            np.array([0.05, 0.12, 0.18, 0.25]),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.pkl"
            save_uncertainty_calibration(calibrator, path)
            loaded = load_uncertainty_calibration(path)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertAlmostEqual(loaded.apply(0.5), calibrator.apply(0.5))

    def test_ece_perfect_calibration_is_near_zero(self) -> None:
        predicted = np.linspace(0.05, 0.5, 50)
        actual = predicted.copy()
        ece = expected_calibration_error(predicted, actual, n_bins=5)
        self.assertLess(ece, 0.02)

    def test_apply_without_calibrator_returns_raw(self) -> None:
        self.assertAlmostEqual(
            apply_calibrated_uncertainty(None, 0.42, calibration_configured=False),
            0.42,
        )

    def test_missing_calibration_warns_once(self) -> None:
        from worldspace.surrogate import calibration as calibration_module

        calibration_module._missing_calibration_warned = False
        with self.assertLogs(
            "worldspace.surrogate.calibration", level="WARNING"
        ) as logs:
            apply_calibrated_uncertainty(None, 0.2, calibration_configured=True)
            apply_calibrated_uncertainty(None, 0.3, calibration_configured=True)
        self.assertEqual(len(logs.output), 1)


class TestSurrogateFacadeCalibration(unittest.TestCase):
    def test_facade_applies_calibrator(self) -> None:
        from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec

        calibrator = UncertaintyCalibrator(
            schema_version="1.0",
            method="isotonic_v1",
            x_thresholds=(0.0, 1.0),
            y_thresholds=(0.05, 0.05),
        )
        model = mock.MagicMock()
        model.predict_components.return_value = {
            "stability": 0.2,
            "diversity": 0.2,
            "oscillation_score": 0.1,
            "topology_interface_index": 0.1,
            "topology_window_heterogeneity": 0.1,
            "final_density": 0.1,
            "early_extinction_prob": 0.1,
        }
        model.predict_uncertainty.return_value = 0.9
        facade = build_surrogate_facade(
            model,
            uncertainty_fallback=0.5,
            calibration_path=None,
        )
        facade.calibrator = calibrator
        facade.calibration_configured = True
        spec = WorldSpec(
            birth=[1, 3],
            survival=[2, 3],
            noise=0.02,
            resource_regen=0.05,
            predation=0.1,
            cell_types=CANONICAL_CELL_TYPES.copy(),
            grid_size=8,
            steps=200,
            seed=0,
        )
        prediction = facade.predict(spec)
        self.assertAlmostEqual(prediction.uncertainty, 0.05)
        self.assertNotAlmostEqual(prediction.uncertainty, 0.9)


class TestResolveSurrogateStubCalibration(unittest.TestCase):
    def test_resolve_stub_uses_calibrated_uncertainty(self) -> None:
        from worldspace.illuminators.scheduler import (
            SchedulerConfig,
            resolve_surrogate_stub,
        )
        from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec

        class _Stub:
            def predict(self, world_spec: WorldSpec) -> SurrogatePrediction:
                _ = world_spec
                return SurrogatePrediction(
                    components={},
                    measures={"stability": 0.1, "diversity": 0.1},
                    fitness=0.1,
                    uncertainty=0.07,
                )

        config = SchedulerConfig(
            schema_version="1.2",
            iterations=1,
            batch_size=1,
            grid_resolution=5,
            early_extinction_step=200,
            min_steps=200,
            batch_emitters=("random",),
            initial_random_candidates=0,
            llm_enabled=False,
            surrogate_enabled=True,
            surrogate_model_type="lightgbm",
            surrogate_checkpoint="artifacts/surrogate/checkpoints/micro.pkl",
            surrogate_buffer_path="artifacts/surrogate/buffer.jsonl",
            surrogate_stub_mean=0.5,
            surrogate_stub_uncertainty=1.0,
            genetic_mutation_scale=0.02,
        )
        spec = WorldSpec(
            birth=[1, 3],
            survival=[2, 3],
            noise=0.02,
            resource_regen=0.05,
            predation=0.1,
            cell_types=CANONICAL_CELL_TYPES.copy(),
            grid_size=8,
            steps=200,
            seed=0,
        )
        mean, uncertainty = resolve_surrogate_stub(config, _Stub(), spec)
        self.assertAlmostEqual(mean, 0.1)
        self.assertAlmostEqual(uncertainty, 0.07)


if __name__ == "__main__":
    unittest.main()

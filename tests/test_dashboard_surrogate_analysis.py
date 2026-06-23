"""Unit tests for surrogate analysis helpers (no Streamlit)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class TestDashboardSurrogateAnalysis(unittest.TestCase):
    def test_regression_metrics_perfect_prediction(self) -> None:
        from dashboard.utils.surrogate_analysis import regression_metrics

        y = np.array([0.2, 0.5, 0.8], dtype=np.float64)
        mae, r2 = regression_metrics(y, y)
        self.assertAlmostEqual(mae, 0.0, places=6)
        self.assertAlmostEqual(r2, 1.0, places=6)

    def test_build_prediction_frame_calls_predict_fn(self) -> None:
        from dashboard.utils.surrogate_analysis import build_prediction_frame

        frame = pd.DataFrame(
            {
                "fitness": [0.3],
                "world_spec": [{"grid_size": 10, "steps": 20, "seed": 1}],
            }
        )

        def predict_fn(spec: dict) -> dict[str, float]:
            self.assertIn("grid_size", spec)
            return {"fitness": 0.25, "uncertainty": 0.1}

        result = build_prediction_frame(frame, predict_fn)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(float(result.iloc[0]["pred_fitness"]), 0.25)

    def test_calibration_table_has_expected_bins(self) -> None:
        from dashboard.utils.surrogate_analysis import calibration_table

        rng = np.random.default_rng(0)
        y_true = rng.random(40)
        y_pred = y_true + rng.normal(0.0, 0.05, 40)
        uncertainty = rng.random(40)
        table = calibration_table(y_true, y_pred, uncertainty, n_bins=4)
        self.assertEqual(len(table), 4)
        self.assertGreater(float(table["mae"].max()), 0.0)

    def test_build_prediction_frame_skips_failed_predictions(self) -> None:
        from dashboard.utils.surrogate_analysis import build_prediction_frame

        frame = pd.DataFrame(
            {
                "fitness": [0.2, 0.4],
                "world_spec": [{"ok": True}, {"bad": True}],
            }
        )

        def predict_fn(spec: dict) -> dict[str, float] | None:
            if "ok" in spec:
                return {"fitness": 0.15, "uncertainty": 0.2}
            return None

        result = build_prediction_frame(frame, predict_fn)
        self.assertEqual(len(result), 1)

    def test_sample_collapsed_rows_respects_max(self) -> None:
        from dashboard.utils.surrogate_analysis import sample_collapsed_rows

        frame = pd.DataFrame(
            {
                "fitness": [float(index) / 10.0 for index in range(20)],
                "world_spec": [{"seed": index} for index in range(20)],
            }
        )
        sample = sample_collapsed_rows(frame, max_rows=5, seed=1)
        self.assertEqual(len(sample), 5)

    def test_load_checkpoint_training_summary(self) -> None:
        from dashboard.utils.surrogate_analysis import (
            checkpoint_summary_path,
            load_checkpoint_training_summary,
            training_summary_holdout_metrics,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "nightly_v2.pkl"
            checkpoint.write_bytes(b"placeholder")
            summary_path = checkpoint_summary_path(checkpoint)
            summary_path.write_text(
                json.dumps(
                    {
                        "sample_count": 1088,
                        "train_count": 870,
                        "holdout_count": 218,
                        "quality_passed": True,
                        "hints_ok": False,
                        "holdout_metrics": {
                            "r2_fitness": 0.91,
                            "mae_fitness": 0.02,
                            "mae_stability": 0.03,
                        },
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_checkpoint_training_summary(checkpoint)
            assert loaded is not None
            meta = training_summary_holdout_metrics(loaded)
            self.assertEqual(meta["sample_count"], 1088)
            self.assertAlmostEqual(float(meta["r2_fitness"]), 0.91)
            self.assertTrue(meta["quality_passed"])
            self.assertFalse(meta["hints_ok"])

    def test_load_checkpoint_training_summary_missing_returns_none(self) -> None:
        from dashboard.utils.surrogate_analysis import load_checkpoint_training_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "missing.pkl"
            checkpoint.write_bytes(b"x")
            self.assertIsNone(load_checkpoint_training_summary(checkpoint))


if __name__ == "__main__":
    unittest.main()

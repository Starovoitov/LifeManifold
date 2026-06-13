"""Train vs analyze hold-out metric parity."""

from __future__ import annotations

import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

from worldspace.surrogate.buffer_analysis import analyze_buffer_path
from worldspace.surrogate.evaluation import hints_thresholds_met
from worldspace.surrogate.model import TARGET_KEYS
from worldspace.surrogate.synthetic_buffer import write_synthetic_buffer
from worldspace.surrogate.training_runtime import train_from_buffer


class TestTrainAnalyzeParity(unittest.TestCase):
    def test_lightgbm_holdout_metrics_match_train_summary(self) -> None:
        if find_spec("lightgbm") is None:
            self.skipTest("lightgbm not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            buffer_path = root / "buffer.jsonl"
            checkpoint_path = root / "model.pkl"
            summary_path = root / "model.summary.json"
            write_synthetic_buffer(buffer_path, n_samples=240, seed=42)
            train_result = train_from_buffer(
                buffer_path=buffer_path,
                checkpoint_path=checkpoint_path,
                summary_path=summary_path,
                model_type="lightgbm",
                micro=True,
                require_quality_gate=False,
                consistency_weight=0.0,
            )
            self.assertTrue(train_result.success, msg=train_result.error_message)
            report = analyze_buffer_path(
                buffer_path,
                fit_model=True,
                model_type="lightgbm",
                ensemble_size=8,
                random_state=42,
                test_fraction=0.2,
                consistency_weight=0.0,
            )
            _assert_holdout_metrics_parity(
                self,
                train_result.holdout_metrics,
                report["model_holdout"],
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                summary["hints_ok"],
                hints_thresholds_met(train_result.holdout_metrics),
            )
            _assert_per_target_parity(
                self,
                summary["per_target_holdout"],
                report["per_target_holdout"],
            )


def _assert_holdout_metrics_parity(
    testcase: unittest.TestCase,
    train_metrics: dict[str, float],
    analyze_metrics: dict[str, float],
) -> None:
    for key in ("r2_fitness", "mae_fitness", "mae_stability"):
        testcase.assertAlmostEqual(
            analyze_metrics[key],
            train_metrics[key],
            places=3,
            msg=f"mismatch on {key}",
        )


def _assert_per_target_parity(
    testcase: unittest.TestCase,
    train_rows: list[dict[str, float | str]],
    analyze_rows: list[dict[str, float | str]],
) -> None:
    train_by_target = {str(row["target"]): row for row in train_rows}
    analyze_by_target = {str(row["target"]): row for row in analyze_rows}
    testcase.assertEqual(set(train_by_target), set(analyze_by_target))
    for target in TARGET_KEYS:
        train_row = train_by_target[target]
        analyze_row = analyze_by_target[target]
        testcase.assertAlmostEqual(
            float(train_row["mae"]),
            float(analyze_row["mae"]),
            places=3,
            msg=f"per-target mae mismatch on {target}",
        )
        testcase.assertAlmostEqual(
            float(train_row["r2"]),
            float(analyze_row["r2"]),
            places=3,
            msg=f"per-target r2 mismatch on {target}",
        )


if __name__ == "__main__":
    unittest.main()

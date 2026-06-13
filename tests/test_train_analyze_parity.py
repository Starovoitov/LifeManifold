"""Train vs analyze hold-out metric parity."""

from __future__ import annotations

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

from worldspace.surrogate.buffer_analysis import analyze_buffer_path
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
            train_metrics = train_result.holdout_metrics
            analyze_metrics = report["model_holdout"]
            for key in ("r2_fitness", "mae_fitness", "mae_stability"):
                self.assertAlmostEqual(
                    analyze_metrics[key],
                    train_metrics[key],
                    places=3,
                    msg=f"mismatch on {key}",
                )


if __name__ == "__main__":
    unittest.main()

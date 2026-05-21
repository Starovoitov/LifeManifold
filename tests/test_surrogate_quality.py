"""Hold-out quality gates for surrogate training."""

from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

from worldspace.surrogate.evaluation import (
    QUALITY_MAE_FITNESS_MAX,
    QUALITY_MAE_STABILITY_MAX,
    QUALITY_R2_FITNESS_MIN,
    evaluate_holdout,
    quality_thresholds_met,
)
from worldspace.surrogate.model import SurrogateModel
from worldspace.surrogate.synthetic_buffer import write_synthetic_buffer
from worldspace.surrogate.training import holdout_split, load_buffer


class TestSurrogateHoldoutQuality(unittest.TestCase):
    def test_holdout_evaluate_without_sklearn_feature_name_warnings(self) -> None:
        import warnings

        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            write_synthetic_buffer(buffer_path, n_samples=400, seed=11)
            features, targets = load_buffer(buffer_path)
            x_train, y_train, x_holdout, y_holdout = holdout_split(features, targets)
            model = SurrogateModel(
                model_type="lightgbm", random_state=42, ensemble_size=8
            )
            model.fit(x_train, y_train)
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                metrics = evaluate_holdout(model, x_holdout, y_holdout)
            self.assertIn("r2_fitness", metrics)

    def test_synthetic_buffer_meets_mvp_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            write_synthetic_buffer(buffer_path, n_samples=2400, seed=42)
            features, targets = load_buffer(buffer_path)
            x_train, y_train, x_holdout, y_holdout = holdout_split(features, targets)
            model = SurrogateModel(
                model_type="lightgbm", random_state=42, ensemble_size=8
            )
            model.fit(x_train, y_train)
            self.assertTrue(model._uses_lightgbm)
            metrics = evaluate_holdout(model, x_holdout, y_holdout)
            self.assertGreater(metrics["r2_fitness"], QUALITY_R2_FITNESS_MIN)
            self.assertLess(metrics["mae_fitness"], QUALITY_MAE_FITNESS_MAX)
            self.assertLess(metrics["mae_stability"], QUALITY_MAE_STABILITY_MAX)
            self.assertTrue(quality_thresholds_met(metrics))

    def test_train_script_micro_checkpoint_on_synthetic_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            buffer_path = root / "buffer.jsonl"
            write_synthetic_buffer(buffer_path, n_samples=160, seed=7)
            checkpoint_path = root / "micro.pkl"
            summary_path = root / "micro.summary.json"

            import os
            import subprocess

            repo_root = Path(__file__).resolve().parents[1]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root)
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_surrogate.py",
                    "--buffer-path",
                    str(buffer_path),
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--summary-path",
                    str(summary_path),
                    "--micro",
                ],
                cwd=repo_root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(checkpoint_path.is_file())
            with checkpoint_path.open("rb") as fh:
                loaded = pickle.load(fh)
            self.assertIsInstance(loaded, SurrogateModel)
            self.assertTrue(loaded._uses_lightgbm)


if __name__ == "__main__":
    unittest.main()

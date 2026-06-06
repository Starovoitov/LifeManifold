"""Hold-out quality gates for surrogate training."""

from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from worldspace.specs.world_param_bounds import (
    NOISE_MAX,
    PREDATION_MAX,
    RESOURCE_REGEN_MAX,
)
from worldspace.surrogate.evaluation import (
    QUALITY_MAE_FITNESS_MAX,
    QUALITY_MAE_STABILITY_MAX,
    QUALITY_R2_FITNESS_MIN,
    evaluate_holdout,
    quality_thresholds_met,
)
from worldspace.surrogate.genome_features import FEATURE_DIM
from worldspace.surrogate.model import SurrogateModel
from worldspace.surrogate.synthetic_buffer import (
    _targets_from_features,
    _v1_scaled_predation,
    _v1_scaled_regen,
    write_synthetic_buffer,
)
from worldspace.surrogate.training import holdout_split, load_buffer


class TestSurrogateHoldoutQuality(unittest.TestCase):
    def test_synthetic_target_formulas_use_v1_float_magnitudes(self) -> None:
        self.assertAlmostEqual(_v1_scaled_regen(RESOURCE_REGEN_MAX), 0.15)
        self.assertAlmostEqual(_v1_scaled_predation(PREDATION_MAX), 0.25)

        features = np.zeros(FEATURE_DIM, dtype=float)
        features[18] = NOISE_MAX
        features[19] = RESOURCE_REGEN_MAX
        features[20] = PREDATION_MAX
        targets = _targets_from_features(features)

        self.assertAlmostEqual(targets["stability"], 0.10 * NOISE_MAX)
        self.assertAlmostEqual(targets["diversity"], 0.20 * 0.15 + 0.15 * 0.25)
        self.assertAlmostEqual(
            targets["oscillation_score"], 0.40 * NOISE_MAX + 0.20 * 0.25
        )
        self.assertAlmostEqual(
            targets["final_density"], 0.40 * 0.15 + 0.30 * (1.0 - 0.25)
        )

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

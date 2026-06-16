"""Tests for PyTorch MLP surrogate backend (Strategy A multi-task)."""

from __future__ import annotations

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import numpy as np

from worldspace.surrogate.checkpoint_io import (
    load_surrogate_checkpoint,
    save_surrogate_checkpoint,
)
from worldspace.surrogate.evaluation import evaluate_holdout
from worldspace.surrogate.model import (
    EXPECTED_FEATURE_DIM,
    FITNESS_TARGET_KEY,
    SurrogateModel,
    checkpoint_feature_dim,
)
from worldspace.surrogate.mlp_model import MlpTrainConfig
from worldspace.surrogate.synthetic_buffer import write_synthetic_buffer
from worldspace.surrogate.training import holdout_split, load_buffer
from worldspace.surrogate.utils import resolve_surrogate_fitness
from worldspace.surrogate.types import SurrogatePrediction


def _fast_mlp_config(**overrides: object) -> MlpTrainConfig:
    base = {
        "max_epochs": 40,
        "patience": 8,
        "batch_size": 64,
    }
    base.update(overrides)
    return MlpTrainConfig(**base)  # type: ignore[arg-type]


@unittest.skipIf(find_spec("torch") is None, "torch not installed")
class TestSurrogateMlp(unittest.TestCase):
    def _fit_mlp(
        self,
        feature_matrix: np.ndarray,
        targets: dict[str, np.ndarray],
        *,
        ensemble_size: int = 2,
        seed: int = 42,
    ) -> SurrogateModel:
        model = SurrogateModel(
            model_type="mlp",
            random_state=seed,
            ensemble_size=ensemble_size,
        )
        x_train, y_train, x_hold, y_hold = holdout_split(
            feature_matrix,
            targets,
            random_state=seed,
        )
        with patch(
            "worldspace.surrogate.mlp_model.MlpTrainConfig",
            side_effect=lambda **kw: _fast_mlp_config(**kw),
        ):
            model.fit(
                x_train,
                y_train,
                val_features=x_hold,
                val_targets=y_hold,
                fitness_loss_weight=1.0,
            )
        return model

    def test_mlp_predict_is_not_constant_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            write_synthetic_buffer(buffer_path, n_samples=120, seed=11)
            feature_matrix, targets = load_buffer(buffer_path)
            model = self._fit_mlp(feature_matrix, targets)
        self.assertTrue(model._uses_mlp)
        self.assertGreater(len(model._mlp_members), 0)

        preds = [
            model.predict_components(feature_matrix[index])["stability"]
            for index in range(min(20, feature_matrix.shape[0]))
        ]
        self.assertGreater(float(np.std(preds)), 1e-4)

    def test_mlp_predictions_differ_from_global_mean(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            write_synthetic_buffer(buffer_path, n_samples=120, seed=13)
            feature_matrix, targets = load_buffer(buffer_path)
            model = self._fit_mlp(feature_matrix, targets)

        train_mean = float(np.mean(targets["stability"]))
        row_features = feature_matrix[0]
        predicted = model.predict_components(row_features)["stability"]
        self.assertNotAlmostEqual(predicted, train_mean, places=2)

    def test_mlp_checkpoint_roundtrip_and_feature_dim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            checkpoint = Path(tmpdir) / "mlp.pkl"
            write_synthetic_buffer(buffer_path, n_samples=100, seed=7)
            feature_matrix, targets = load_buffer(buffer_path)
            model = self._fit_mlp(feature_matrix, targets, ensemble_size=2)
            save_surrogate_checkpoint(model, checkpoint)
            loaded = load_surrogate_checkpoint(checkpoint)

        self.assertTrue(loaded._uses_mlp)
        self.assertEqual(checkpoint_feature_dim(loaded), EXPECTED_FEATURE_DIM)
        row = feature_matrix[0]
        before = model.predict_components(row)
        after = loaded.predict_components(row)
        for key in before:
            self.assertAlmostEqual(before[key], after[key], places=5)

    def test_mlp_training_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            write_synthetic_buffer(buffer_path, n_samples=100, seed=19)
            feature_matrix, targets = load_buffer(buffer_path)
            model_a = self._fit_mlp(feature_matrix, targets, ensemble_size=2, seed=42)
            model_b = self._fit_mlp(feature_matrix, targets, ensemble_size=2, seed=42)

        row = feature_matrix[0]
        pred_a = model_a.predict_components(row)
        pred_b = model_b.predict_components(row)
        for key in pred_a:
            self.assertAlmostEqual(pred_a[key], pred_b[key], places=6)

    def test_mlp_direct_fitness_head_and_holdout_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            write_synthetic_buffer(buffer_path, n_samples=160, seed=23)
            feature_matrix, targets = load_buffer(buffer_path)
            if FITNESS_TARGET_KEY not in targets:
                targets[FITNESS_TARGET_KEY] = np.clip(
                    0.2 + 0.6 * np.mean(feature_matrix, axis=1),
                    0.0,
                    1.0,
                )
            x_train, y_train, x_hold, y_hold = holdout_split(
                feature_matrix,
                targets,
                random_state=42,
            )
            model = SurrogateModel(model_type="mlp", random_state=42, ensemble_size=2)
            with patch(
                "worldspace.surrogate.mlp_model.MlpTrainConfig",
                side_effect=lambda **kw: _fast_mlp_config(**kw),
            ):
                model.fit(
                    x_train,
                    y_train,
                    val_features=x_hold,
                    val_targets=y_hold,
                )
            self.assertTrue(model._has_fitness_head)
            holdout = evaluate_holdout(model, x_hold, y_hold)
            self.assertIn("r2_fitness_direct", holdout)

            row_features = x_hold[0]
            components = model.predict_components(row_features)
            prediction = SurrogatePrediction(
                components=components,
                measures={
                    "stability": float(components["stability"]),
                    "diversity": float(components["diversity"]),
                },
                fitness=0.0,
                uncertainty=0.0,
            )
            direct = model.predict_fitness(row_features)
            self.assertIsNotNone(direct)
            resolved = resolve_surrogate_fitness(model, row_features, prediction)
            self.assertAlmostEqual(resolved, float(direct))

    def test_mlp_predict_uncertainty_nonzero_with_ensemble(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            write_synthetic_buffer(buffer_path, n_samples=100, seed=29)
            feature_matrix, targets = load_buffer(buffer_path)
            model = self._fit_mlp(feature_matrix, targets, ensemble_size=4)

        uncertainty = model.predict_uncertainty(feature_matrix[0])
        self.assertGreaterEqual(uncertainty, 0.0)

    def test_mlp_predict_uses_trained_input_dim_not_runtime_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            write_synthetic_buffer(buffer_path, n_samples=120, seed=31)
            feature_matrix, targets = load_buffer(buffer_path)
            legacy_features = feature_matrix[:, :21]
            model = self._fit_mlp(legacy_features, targets, ensemble_size=2)

        self.assertEqual(model._trained_input_dim, 21)
        prediction = model.predict_components(legacy_features[0])
        self.assertIn("stability", prediction)

    def test_mlp_checkpoint_uses_batch_norm_and_gelu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            write_synthetic_buffer(buffer_path, n_samples=120, seed=37)
            feature_matrix, targets = load_buffer(buffer_path)
            model = self._fit_mlp(feature_matrix, targets, ensemble_size=1)

        from worldspace.surrogate.mlp_model import mlp_state_dict_uses_batch_norm

        state_dict = model._mlp_members[0]
        self.assertTrue(mlp_state_dict_uses_batch_norm(state_dict))
        self.assertIn("net.1.running_mean", state_dict)
        self.assertIn("net.4.running_mean", state_dict)

    def test_legacy_relu_checkpoint_still_loads(self) -> None:
        from worldspace.surrogate.mlp_model import (
            build_strategy_a_mlp,
            predict_mlp_state_dict,
        )

        legacy = build_strategy_a_mlp(
            input_dim=21,
            hidden_dims=(64, 64),
            activation="relu",
            batch_norm=False,
        )
        state_dict = legacy.state_dict()
        features = np.zeros(21, dtype=np.float32)
        preds = predict_mlp_state_dict(state_dict, features, hidden_dims=(64, 64))
        self.assertEqual(preds.shape, (1, 8))

    def test_mlp_train_config_uses_huber_and_cosine_defaults(self) -> None:
        from worldspace.surrogate.mlp_model import MlpTrainConfig, _make_loss_fn

        cfg = MlpTrainConfig()
        self.assertEqual(cfg.huber_delta, 0.05)
        self.assertEqual(cfg.cosine_eta_min_factor, 0.01)
        loss_fn = _make_loss_fn(cfg)
        self.assertEqual(loss_fn.beta, 0.05)


if __name__ == "__main__":
    unittest.main()

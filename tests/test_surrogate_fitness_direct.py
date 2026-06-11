"""Tests for direct illuminator fitness head (Strategy A + fitness target)."""

from __future__ import annotations

import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from worldspace.illuminators.evaluation import extinction_probability
from worldspace.surrogate.buffer import buffer_record, world_spec_dict_for_buffer
from worldspace.surrogate.evaluation import evaluate_holdout
from worldspace.surrogate.model import FITNESS_TARGET_KEY, SurrogateModel
from worldspace.surrogate.synthetic_buffer import (
    _random_features,
    _targets_from_features,
    _world_spec_from_features,
)
from worldspace.surrogate.training import holdout_split, load_buffer
from worldspace.surrogate.utils import resolve_surrogate_fitness
from worldspace.surrogate.types import SurrogatePrediction


def _write_direct_fitness_buffer(path: Path, *, n_samples: int, seed: int) -> None:
    """Labels with learnable direct fitness and noisy extinction compose inputs."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for index in range(n_samples):
        features = _random_features(rng)
        targets = _targets_from_features(features)
        targets["final_density"] = 0.49 if index % 2 == 0 else 0.51
        targets["early_extinction_prob"] = float(
            np.clip(
                extinction_probability(targets["final_density"])
                + rng.normal(0.0, 0.06),
                0.0,
                1.0,
            )
        )
        targets[FITNESS_TARGET_KEY] = float(
            np.clip(0.12 + 0.75 * float(np.mean(features)), 0.0, 1.0)
        )
        rows.append(
            buffer_record(
                features=features,
                targets=targets,
                emitter_type="synthetic",
                world_spec=world_spec_dict_for_buffer(
                    _world_spec_from_features(features)
                ),
            )
        )
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


class TestSurrogateFitnessDirect(unittest.TestCase):
    def test_load_buffer_includes_optional_fitness_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            _write_direct_fitness_buffer(buffer_path, n_samples=12, seed=3)
            _, targets = load_buffer(buffer_path)
        self.assertIn(FITNESS_TARGET_KEY, targets)
        self.assertEqual(targets[FITNESS_TARGET_KEY].shape[0], 12)
        self.assertTrue(np.all(np.isfinite(targets[FITNESS_TARGET_KEY])))

    def test_direct_fitness_head_beats_composed_holdout_r2(self) -> None:
        if find_spec("lightgbm") is None:
            self.skipTest("lightgbm not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            _write_direct_fitness_buffer(buffer_path, n_samples=320, seed=17)
            feature_matrix, targets = load_buffer(buffer_path)
            x_train, y_train, x_holdout, y_holdout = holdout_split(
                feature_matrix,
                targets,
                random_state=42,
            )
            model = SurrogateModel(
                model_type="lightgbm", random_state=42, ensemble_size=4
            )
            model.fit(x_train, y_train)
            self.assertTrue(model._has_fitness_head)

            holdout_metrics = evaluate_holdout(model, x_holdout, y_holdout)
            self.assertIn("r2_fitness_direct", holdout_metrics)
            self.assertGreater(
                holdout_metrics["r2_fitness_direct"],
                holdout_metrics["r2_fitness"],
                msg=(
                    f"direct R² ({holdout_metrics['r2_fitness_direct']:.3f}) should exceed "
                    f"composed R² ({holdout_metrics['r2_fitness']:.3f})"
                ),
            )

    def test_resolve_surrogate_fitness_prefers_direct_head(self) -> None:
        if find_spec("lightgbm") is None:
            self.skipTest("lightgbm not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            _write_direct_fitness_buffer(buffer_path, n_samples=80, seed=5)
            feature_matrix, targets = load_buffer(buffer_path)
            model = SurrogateModel(
                model_type="lightgbm", random_state=42, ensemble_size=2
            )
            model.fit(feature_matrix, targets)
            row_features = feature_matrix[0]
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


if __name__ == "__main__":
    unittest.main()

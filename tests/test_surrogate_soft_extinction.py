"""Tests for surrogate-only soft extinction compose."""

from __future__ import annotations

import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from worldspace.illuminators.evaluation import extinction_probability
from worldspace.surrogate.buffer import buffer_record, world_spec_dict_for_buffer
from worldspace.surrogate.evaluation import evaluate_fitness_compose_ab
from worldspace.surrogate.model import FITNESS_TARGET_KEY, SurrogateModel
from worldspace.surrogate.synthetic_buffer import (
    _random_features,
    _targets_from_features,
    _world_spec_from_features,
)
from worldspace.surrogate.training import holdout_split, load_buffer
from worldspace.surrogate.types import SurrogateConfig, SurrogatePrediction
from worldspace.surrogate.utils import (
    compute_fitness_from_prediction,
    compute_soft_fitness_from_prediction,
    resolve_surrogate_fitness,
)


def _write_threshold_sensitive_buffer(path: Path, *, n_samples: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for index in range(n_samples):
        features = _random_features(rng)
        targets = _targets_from_features(features)
        targets.pop(FITNESS_TARGET_KEY, None)
        targets["final_density"] = 0.49 if index % 2 == 0 else 0.51
        targets["early_extinction_prob"] = extinction_probability(
            targets["final_density"]
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


class TestSurrogateSoftExtinction(unittest.TestCase):
    def test_hard_compose_zeros_fitness_above_threshold(self) -> None:
        base_components = {
            "stability": 0.7,
            "diversity": 0.6,
            "oscillation_score": 0.5,
            "topology_interface_index": 0.4,
            "topology_window_heterogeneity": 0.3,
            "final_density": 0.49,
            "early_extinction_prob": 0.49,
        }
        shifted = dict(base_components)
        shifted["early_extinction_prob"] = 0.51
        fitness_below = compute_fitness_from_prediction(
            SurrogatePrediction(
                components=base_components,
                measures={"stability": 0.7, "diversity": 0.6},
                fitness=0.0,
                uncertainty=0.0,
            )
        )
        fitness_above = compute_fitness_from_prediction(
            SurrogatePrediction(
                components=shifted,
                measures={"stability": 0.7, "diversity": 0.6},
                fitness=0.0,
                uncertainty=0.0,
            )
        )
        self.assertGreater(fitness_below, 0.0)
        self.assertEqual(fitness_above, 0.0)

    def test_soft_compose_nonzero_above_hard_threshold(self) -> None:
        components = {
            "stability": 0.7,
            "diversity": 0.6,
            "oscillation_score": 0.5,
            "topology_interface_index": 0.4,
            "topology_window_heterogeneity": 0.3,
            "final_density": 0.49,
            "early_extinction_prob": 0.51,
        }
        prediction = SurrogatePrediction(
            components=components,
            measures={"stability": 0.7, "diversity": 0.6},
            fitness=0.0,
            uncertainty=0.0,
        )
        hard = compute_fitness_from_prediction(prediction)
        soft = compute_soft_fitness_from_prediction(prediction)
        self.assertEqual(hard, 0.0)
        self.assertGreater(soft, 0.0)

    def test_resolve_surrogate_fitness_uses_soft_when_configured(self) -> None:
        if find_spec("lightgbm") is None:
            self.skipTest("lightgbm not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            _write_threshold_sensitive_buffer(buffer_path, n_samples=80, seed=5)
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
            hard = resolve_surrogate_fitness(
                model,
                row_features,
                prediction,
                use_soft_extinction=False,
            )
            soft = resolve_surrogate_fitness(
                model,
                row_features,
                prediction,
                use_soft_extinction=True,
            )
            self.assertEqual(hard, 0.0)
            self.assertGreater(soft, 0.0)

    def test_soft_compose_improves_holdout_r2_on_threshold_buffer(self) -> None:
        if find_spec("lightgbm") is None:
            self.skipTest("lightgbm not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            _write_threshold_sensitive_buffer(buffer_path, n_samples=320, seed=11)
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
            ab = evaluate_fitness_compose_ab(model, x_holdout, y_holdout)
            self.assertGreater(
                ab["soft"]["r2_fitness"],
                ab["hard"]["r2_fitness"],
                msg=(
                    f"soft R² ({ab['soft']['r2_fitness']:.3f}) should exceed "
                    f"hard R² ({ab['hard']['r2_fitness']:.3f})"
                ),
            )

    def test_surrogate_config_soft_extinction_default_false(self) -> None:
        config = SurrogateConfig(
            enabled=True,
            model_type="lightgbm",
            checkpoint=None,
            stub_mean=0.5,
            stub_uncertainty=0.85,
        )
        self.assertFalse(config.use_soft_extinction)


if __name__ == "__main__":
    unittest.main()

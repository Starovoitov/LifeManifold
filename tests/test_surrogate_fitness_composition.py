"""Regression tests for composed-fitness bottleneck (Strategy A compose)."""

from __future__ import annotations

import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

from worldspace.illuminators.evaluation import (
    apply_canonical_seed,
    extinction_probability,
)
from worldspace.surrogate.buffer import buffer_record, world_spec_dict_for_buffer
from worldspace.surrogate.feature_extractor import extract
from worldspace.surrogate.evaluation import evaluate_holdout
from worldspace.surrogate.model import FITNESS_TARGET_KEY, SurrogateModel
from worldspace.surrogate.synthetic_buffer import (
    _random_features,
    _targets_from_features,
    _world_spec_from_features,
)
from worldspace.surrogate.training import holdout_split, load_buffer
from worldspace.surrogate.types import SurrogatePrediction
from worldspace.surrogate.utils import compute_fitness_from_prediction


def _write_threshold_sensitive_buffer(path: Path, *, n_samples: int, seed: int) -> None:
    """Buffer where ``final_density`` toggles around the extinction compose threshold."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for index in range(n_samples):
        raw_features = _random_features(rng)
        world_spec = _world_spec_from_features(raw_features)
        apply_canonical_seed(world_spec)
        features = extract(world_spec)
        targets = _targets_from_features(raw_features)
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
                world_spec=world_spec_dict_for_buffer(world_spec),
            )
        )
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


class TestSurrogateFitnessComposition(unittest.TestCase):
    def test_hard_extinction_step_amplifies_small_component_error(self) -> None:
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

    def test_trained_components_can_outscore_composed_fitness_r2(self) -> None:
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
            self.assertFalse(model._has_fitness_head)

            holdout_metrics = evaluate_holdout(
                model,
                x_holdout,
                y_holdout,
                extinction_gate_threshold=0.5,
            )
            pred_stability = np.asarray(
                [
                    model.predict_components(x_holdout[row_index])["stability"]
                    for row_index in range(x_holdout.shape[0])
                ],
                dtype=float,
            )
            r2_stability = float(r2_score(y_holdout["stability"], pred_stability))
            self.assertGreater(r2_stability, 0.35)
            self.assertGreater(
                r2_stability,
                holdout_metrics["r2_fitness"],
                msg=(
                    f"component R² ({r2_stability:.3f}) should exceed composed "
                    f"fitness R² ({holdout_metrics['r2_fitness']:.3f})"
                ),
            )


if __name__ == "__main__":
    unittest.main()

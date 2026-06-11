"""Unit tests for surrogate acquisition reporting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from worldspace.surrogate.acquisition_config import AcquisitionConfig
from worldspace.surrogate.calibration import UncertaintyCalibrator
from worldspace.surrogate.feature_extractor import FEATURE_NAMES
from worldspace.surrogate.evaluation import fitness_from_target_row
from worldspace.surrogate.model import FITNESS_TARGET_KEY, TARGET_KEYS, SurrogateModel
from worldspace.surrogate.reporting import (
    consistency_mae,
    evaluate_acquisition_replay,
    estimate_false_skip_rate,
    merge_acquisition_into_summary,
)


class TestSurrogateReporting(unittest.TestCase):
    def test_known_false_skip_rate_on_synthetic_rows(self) -> None:
        n = 4
        feature_matrix = np.tile(np.linspace(0.1, 0.9, len(FEATURE_NAMES)), (n, 1))
        targets = {key: np.full(n, 0.1, dtype=float) for key in TARGET_KEYS}
        targets["stability"] = np.array([0.1, 0.8, 0.1, 0.1])
        targets["diversity"] = np.array([0.1, 0.8, 0.1, 0.1])
        model = SurrogateModel(model_type="lightgbm", ensemble_size=2)
        model.set_component_defaults(0.05)
        calibrator = UncertaintyCalibrator(
            schema_version="1.0",
            method="isotonic_v1",
            x_thresholds=(0.0, 1.0),
            y_thresholds=(0.05, 0.05),
        )
        policy = AcquisitionConfig(
            mode="filter",
            min_predicted_fitness=0.25,
            max_uncertainty_to_skip=0.40,
            never_skip_empty_bin=False,
        )
        metrics = evaluate_acquisition_replay(
            model,
            feature_matrix,
            targets,
            policy,
            calibrator=calibrator,
            grid_resolution=2,
        )
        self.assertEqual(metrics.row_count, n)
        self.assertGreater(metrics.policy_skip_count, 0)
        rate = estimate_false_skip_rate(
            model,
            feature_matrix,
            targets,
            policy,
            calibrator=calibrator,
            grid_resolution=2,
        )
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_high_fitness_row_not_counted_as_false_skip_when_eval(self) -> None:
        feature_matrix = np.tile(0.5, (1, len(FEATURE_NAMES))).reshape(1, -1)
        targets = {key: np.array([0.55], dtype=float) for key in TARGET_KEYS}
        targets["early_extinction_prob"] = np.array([0.05], dtype=float)
        model = SurrogateModel()
        model.set_component_defaults(0.55)
        model._component_means["early_extinction_prob"] = 0.05
        policy = AcquisitionConfig(
            mode="filter",
            min_predicted_fitness=0.25,
            max_uncertainty_to_skip=0.40,
            never_skip_empty_bin=False,
        )
        metrics = evaluate_acquisition_replay(
            model,
            feature_matrix,
            targets,
            policy,
            grid_resolution=2,
        )
        self.assertEqual(metrics.policy_skip_count, 0)

    def test_consistency_mae_uses_composed_labels_without_fitness_head(self) -> None:
        components = {
            "stability": 0.7,
            "diversity": 0.6,
            "oscillation_score": 0.5,
            "topology_interface_index": 0.4,
            "topology_window_heterogeneity": 0.3,
            "final_density": 0.6,
            "early_extinction_prob": 0.05,
        }
        composed = fitness_from_target_row(components)
        stored_label = 0.99
        self.assertNotAlmostEqual(composed, stored_label, places=3)

        feature_matrix = np.tile(0.5, (1, len(FEATURE_NAMES))).reshape(1, -1)
        targets = {key: np.array([components[key]], dtype=float) for key in TARGET_KEYS}
        targets[FITNESS_TARGET_KEY] = np.array([stored_label], dtype=float)

        model = SurrogateModel()
        model._component_means = dict(components)
        model._has_fitness_head = False

        self.assertAlmostEqual(
            consistency_mae(model, feature_matrix, targets), 0.0, places=5
        )

    def test_consistency_mae_nan_label_rows_use_composed_with_fitness_head(
        self,
    ) -> None:
        components = {
            "stability": 0.7,
            "diversity": 0.6,
            "oscillation_score": 0.5,
            "topology_interface_index": 0.4,
            "topology_window_heterogeneity": 0.3,
            "final_density": 0.6,
            "early_extinction_prob": 0.05,
        }
        stored_label = 0.82
        feature_matrix = np.tile(0.5, (2, len(FEATURE_NAMES)))
        targets = {
            key: np.array([components[key], components[key]], dtype=float)
            for key in TARGET_KEYS
        }
        targets[FITNESS_TARGET_KEY] = np.array([stored_label, np.nan], dtype=float)

        model = SurrogateModel()
        model._component_means = dict(components)
        model._has_fitness_head = True
        model.predict_fitness = lambda _features: 0.1  # type: ignore[method-assign]

        mae = consistency_mae(model, feature_matrix, targets)
        expected = (abs(0.1 - stored_label) + 0.0) / 2.0
        self.assertAlmostEqual(mae, expected, places=5)

    def test_consistency_mae_uses_stored_labels_with_fitness_head(self) -> None:
        stored_label = 0.82
        feature_matrix = np.tile(0.5, (1, len(FEATURE_NAMES))).reshape(1, -1)
        targets = {key: np.array([0.4], dtype=float) for key in TARGET_KEYS}
        targets[FITNESS_TARGET_KEY] = np.array([stored_label], dtype=float)

        model = SurrogateModel()
        model._component_means = {key: 0.4 for key in TARGET_KEYS}
        model._has_fitness_head = True
        model.predict_fitness = lambda _features: stored_label  # type: ignore[method-assign]

        self.assertAlmostEqual(
            consistency_mae(model, feature_matrix, targets), 0.0, places=5
        )

    def test_merge_acquisition_into_summary_preserves_existing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "model.summary.json"
            merge_acquisition_into_summary(
                summary_path,
                {
                    "recommended_skip_rate": 0.5,
                    "policy_mode": "filter",
                },
            )
            merge_acquisition_into_summary(
                summary_path,
                {
                    "calibration_ece": 0.03,
                    "calibration_holdout_samples": 100,
                },
            )
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        acquisition = payload["acquisition"]
        self.assertEqual(acquisition["recommended_skip_rate"], 0.5)
        self.assertEqual(acquisition["policy_mode"], "filter")
        self.assertAlmostEqual(acquisition["calibration_ece"], 0.03)
        self.assertEqual(acquisition["calibration_holdout_samples"], 100)


if __name__ == "__main__":
    unittest.main()

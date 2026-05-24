"""Unit tests for Surrogate Acquisition scheduler YAML (SA-1)."""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

import yaml

from worldspace.illuminators.scheduler import (
    DEFAULT_MINI_SCHEDULER_PATH,
    load_scheduler,
)
from worldspace.surrogate.acquisition_config import DEFAULT_SURROGATE_ARCHIVE_PATH

_SPECS = Path(__file__).resolve().parents[1] / "worldspace" / "specs"
_SHADOW_PATH = _SPECS / "map_elites_scheduler_mini_surrogate_shadow.yaml"
_FILTER_PATH = _SPECS / "map_elites_scheduler_mini_surrogate_filter.yaml"


class TestAcquisitionSchedulerYaml(unittest.TestCase):
    def test_legacy_mini_yaml_defaults_acquisition_off(self) -> None:
        config = load_scheduler(DEFAULT_MINI_SCHEDULER_PATH)
        self.assertEqual(config.acquisition.mode, "off")
        self.assertEqual(config.acquisition.policy, "threshold_gate")
        self.assertEqual(config.surrogate_archive_path, DEFAULT_SURROGATE_ARCHIVE_PATH)
        self.assertFalse(config.retrain.enabled)
        self.assertIsNone(config.surrogate_calibration)

    def test_load_shadow_spec(self) -> None:
        config = load_scheduler(_SHADOW_PATH)
        self.assertTrue(config.surrogate_enabled)
        self.assertEqual(config.acquisition.mode, "shadow")
        self.assertAlmostEqual(config.acquisition.min_predicted_fitness, 0.25)
        self.assertEqual(config.iterations, 2)

    def test_load_filter_spec(self) -> None:
        config = load_scheduler(_FILTER_PATH)
        self.assertEqual(config.acquisition.mode, "filter")
        self.assertAlmostEqual(config.acquisition.min_predicted_fitness, 0.99)
        self.assertEqual(
            config.surrogate_checkpoint,
            "artifacts/surrogate/checkpoints/latest.pkl",
        )
        self.assertEqual(
            config.surrogate_calibration,
            "artifacts/surrogate/checkpoints/calibration.pkl",
        )

    def test_calibration_disabled_via_null_empty_or_false(self) -> None:
        base = yaml.safe_load(_FILTER_PATH.read_text(encoding="utf-8"))
        cases: list[tuple[str, object]] = [
            ("omit", None),
            ("null", None),
            ("empty", ""),
            ("false", False),
        ]
        for label, calibration_value in cases:
            with self.subTest(label=label):
                doc = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
                if label == "omit":
                    doc["surrogate"].pop("calibration", None)
                else:
                    doc["surrogate"]["calibration"] = calibration_value
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = Path(tmpdir) / "sched.yaml"
                    path.write_text(
                        yaml.safe_dump(doc, sort_keys=False), encoding="utf-8"
                    )
                    config = load_scheduler(path)
                self.assertIsNone(config.surrogate_calibration)

    def test_full_acquisition_block_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sched.yaml"
            doc = yaml.safe_load(
                DEFAULT_MINI_SCHEDULER_PATH.read_text(encoding="utf-8")
            )
            doc["surrogate"]["enabled"] = True
            doc["surrogate"][
                "calibration"
            ] = "artifacts/surrogate/checkpoints/calibration.pkl"
            doc["surrogate"]["acquisition"] = {
                "mode": "shadow",
                "policy": "threshold_gate",
                "min_predicted_fitness": 0.3,
                "max_uncertainty_to_skip": 0.5,
                "never_skip_empty_bin": False,
                "exploration_weight": 0.2,
            }
            doc["surrogate"]["retrain"] = {
                "enabled": True,
                "every_iterations": 10,
                "min_new_buffer_rows": 100,
            }
            path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
            config = load_scheduler(path)
        self.assertEqual(config.acquisition.mode, "shadow")
        self.assertFalse(config.acquisition.never_skip_empty_bin)
        self.assertAlmostEqual(config.acquisition.exploration_weight, 0.2)
        self.assertTrue(config.retrain.enabled)
        self.assertEqual(config.retrain.every_iterations, 10)
        self.assertEqual(
            config.surrogate_calibration,
            "artifacts/surrogate/checkpoints/calibration.pkl",
        )

    def test_filter_without_surrogate_forces_off(self) -> None:
        with self.assertLogs(level=logging.WARNING) as captured:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "sched.yaml"
                doc = yaml.safe_load(
                    DEFAULT_MINI_SCHEDULER_PATH.read_text(encoding="utf-8")
                )
                doc["surrogate"]["enabled"] = False
                doc["surrogate"]["acquisition"] = {"mode": "filter"}
                path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
                config = load_scheduler(path)
        self.assertEqual(config.acquisition.mode, "off")
        self.assertTrue(
            any("forcing mode off" in message for message in captured.output)
        )

    def test_ucb_policy_falls_back_to_threshold_gate(self) -> None:
        with self.assertLogs(level=logging.WARNING) as captured:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "sched.yaml"
                doc = yaml.safe_load(
                    DEFAULT_MINI_SCHEDULER_PATH.read_text(encoding="utf-8")
                )
                doc["surrogate"]["acquisition"] = {
                    "mode": "off",
                    "policy": "ucb_promote",
                }
                path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
                config = load_scheduler(path)
        self.assertEqual(config.acquisition.policy, "threshold_gate")
        self.assertTrue(
            any("not implemented yet" in message for message in captured.output)
        )


if __name__ == "__main__":
    unittest.main()

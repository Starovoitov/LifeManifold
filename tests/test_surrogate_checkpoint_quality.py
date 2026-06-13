"""Tests for surrogate checkpoint quality gate helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldspace.surrogate.checkpoint_quality import (
    checkpoint_quality_allows_hints,
    load_checkpoint_summary,
)
from worldspace.surrogate.evaluation import hints_thresholds_met, quality_thresholds_met
from worldspace.surrogate.feature_extractor import FEATURE_SCHEMA_VERSION
from worldspace.surrogate.genome_features import FEATURE_DIM_V21


def _write_summary(checkpoint: Path, payload: dict[str, object]) -> None:
    summary = checkpoint.with_name(f"{checkpoint.stem}.summary.json")
    summary.write_text(json.dumps(payload), encoding="utf-8")


class TestCheckpointQuality(unittest.TestCase):
    def test_missing_summary_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pkl"
            checkpoint.write_bytes(b"placeholder")
            self.assertIsNone(load_checkpoint_summary(checkpoint))
            self.assertFalse(checkpoint_quality_allows_hints(checkpoint))

    def test_pilot_tier_hints_ok_allows_hints_without_quality_passed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "nightly_v2.pkl"
            checkpoint.write_bytes(b"placeholder")
            _write_summary(
                checkpoint,
                {
                    "hints_ok": True,
                    "quality_passed": False,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "feature_dim": FEATURE_DIM_V21,
                },
            )
            self.assertTrue(checkpoint_quality_allows_hints(checkpoint))

    def test_production_tier_legacy_quality_passed_allows_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "nightly_v2.pkl"
            checkpoint.write_bytes(b"placeholder")
            _write_summary(
                checkpoint,
                {
                    "quality_passed": True,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "feature_dim": FEATURE_DIM_V21,
                },
            )
            self.assertTrue(checkpoint_quality_allows_hints(checkpoint))

    def test_below_pilot_tier_rejects_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "nightly_v2.pkl"
            checkpoint.write_bytes(b"placeholder")
            _write_summary(
                checkpoint,
                {
                    "hints_ok": False,
                    "quality_passed": False,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "feature_dim": FEATURE_DIM_V21,
                },
            )
            self.assertFalse(checkpoint_quality_allows_hints(checkpoint))

    def test_hints_thresholds_met_pilot_tier(self) -> None:
        metrics = {
            "r2_fitness": 0.35,
            "mae_fitness": 0.02,
            "mae_stability": 0.07,
        }
        self.assertTrue(hints_thresholds_met(metrics))
        self.assertFalse(quality_thresholds_met(metrics))

    def test_hints_thresholds_boundary_at_r2_0_30(self) -> None:
        metrics = {
            "r2_fitness": 0.30,
            "mae_fitness": 0.02,
            "mae_stability": 0.07,
        }
        self.assertTrue(hints_thresholds_met(metrics))


if __name__ == "__main__":
    unittest.main()

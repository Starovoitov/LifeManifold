"""Tests for surrogate checkpoint quality gate helpers (E5.2)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldspace.surrogate.checkpoint_quality import (
    checkpoint_quality_allows_hints,
    load_checkpoint_summary,
)
from worldspace.surrogate.feature_extractor import FEATURE_SCHEMA_VERSION
from worldspace.surrogate.genome_features import FEATURE_DIM_V21


class TestCheckpointQuality(unittest.TestCase):
    def test_missing_summary_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pkl"
            checkpoint.write_bytes(b"placeholder")
            self.assertIsNone(load_checkpoint_summary(checkpoint))
            self.assertFalse(checkpoint_quality_allows_hints(checkpoint))

    def test_quality_passed_v2_summary_allows_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "nightly_v2.pkl"
            checkpoint.write_bytes(b"placeholder")
            summary = checkpoint.with_name("nightly_v2.summary.json")
            summary.write_text(
                json.dumps(
                    {
                        "quality_passed": True,
                        "feature_schema_version": FEATURE_SCHEMA_VERSION,
                        "feature_dim": FEATURE_DIM_V21,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(checkpoint_quality_allows_hints(checkpoint))

    def test_failed_quality_summary_rejects_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "nightly_v2.pkl"
            checkpoint.write_bytes(b"placeholder")
            summary = checkpoint.with_name("nightly_v2.summary.json")
            summary.write_text(
                json.dumps(
                    {
                        "quality_passed": False,
                        "feature_schema_version": FEATURE_SCHEMA_VERSION,
                        "feature_dim": FEATURE_DIM_V21,
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(checkpoint_quality_allows_hints(checkpoint))


if __name__ == "__main__":
    unittest.main()

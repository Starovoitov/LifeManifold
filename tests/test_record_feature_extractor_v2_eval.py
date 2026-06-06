"""Tests for feature extractor v2 eval memo script (E6.3)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.record_feature_extractor_v2_eval import (
    build_eval_payload,
    load_training_summary,
    resolve_checkpoint_from_summary,
)


class TestRecordFeatureExtractorV2Eval(unittest.TestCase):
    def test_resolve_checkpoint_from_summary_stem(self) -> None:
        summary = Path("/tmp/nightly_v2.summary.json")
        checkpoint = resolve_checkpoint_from_summary(summary, None)
        self.assertEqual(checkpoint, Path("/tmp/nightly_v2.pkl"))

    def test_build_eval_payload_stub_decisions_when_quality_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "micro.summary.json"
            summary = {
                "feature_schema_version": "2.0",
                "feature_dim": 21,
                "sample_count": 200,
                "train_count": 160,
                "holdout_count": 40,
                "quality_passed": False,
                "holdout_metrics": {
                    "r2_fitness": -0.04,
                    "mae_fitness": 0.12,
                    "mae_stability": 0.08,
                },
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            loaded = load_training_summary(summary_path)
            payload = build_eval_payload(
                loaded,
                checkpoint_path=root / "micro.pkl",
                summary_path=summary_path,
                buffer_path=root / "buffer.jsonl",
                archive_path=None,
                dataset_source="synthetic",
                notes="smoke",
            )
            self.assertFalse(payload["quality_passed"])
            decisions = payload["decisions"]
            assert isinstance(decisions, dict)
            self.assertEqual(decisions["llm_hints"], "stub")
            self.assertEqual(decisions["acquisition_filter"], "off")
            self.assertAlmostEqual(
                payload["holdout_metrics"]["r2_fitness"],  # type: ignore[index]
                -0.04,
            )


if __name__ == "__main__":
    unittest.main()

"""Unit tests for nightly pipeline summary JSON (no full run)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldspace.illuminators.nightly_report import NightlyRunReport
from worldspace.scripts.run_map_elites_nightly import _write_pipeline_summary

_REPO_ROOT = Path(__file__).resolve().parents[1]


class TestNightlyPipelineSummary(unittest.TestCase):
    def test_write_pipeline_summary(self) -> None:
        baseline = NightlyRunReport(
            schema_version="1.0",
            scheduler_path=str(
                _REPO_ROOT / "worldspace/specs/map_elites_scheduler_nightly.yaml"
            ),
            seed=0,
            iterations=2,
            evaluations=100,
            filled_cells=10,
            grid_resolution=50,
            archive_type="grid",
            n_cells=50 * 50,
            coverage=0.004,
            jsonl_raw_lines=12,
            jsonl_collapsed_cells=10,
            elapsed_seconds=1.0,
            llm_enabled=False,
            surrogate_enabled=False,
            archive_jsonl_path=str(
                _REPO_ROOT
                / "artifacts/map_elites_nightly/baseline/map_elites_archive.jsonl"
            ),
        )
        surrogate = NightlyRunReport(
            schema_version="1.0",
            scheduler_path=str(
                _REPO_ROOT
                / "worldspace/specs/map_elites_scheduler_nightly_surrogate.yaml"
            ),
            seed=0,
            iterations=2,
            evaluations=100,
            filled_cells=11,
            grid_resolution=50,
            archive_type="grid",
            n_cells=50 * 50,
            coverage=0.0044,
            jsonl_raw_lines=13,
            jsonl_collapsed_cells=11,
            elapsed_seconds=2.0,
            llm_enabled=False,
            surrogate_enabled=True,
            archive_jsonl_path=str(
                _REPO_ROOT
                / "artifacts/map_elites_nightly/surrogate/map_elites_archive.jsonl"
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            training_path = Path(tmp) / "nightly_v2.summary.json"
            training_path.write_text(
                json.dumps({"quality_passed": True, "sample_count": 2500}),
                encoding="utf-8",
            )
            out = Path(tmp) / "nightly_pipeline_summary.json"
            _write_pipeline_summary(
                out,
                baseline=baseline,
                surrogate=surrogate,
                training_summary_path=training_path,
                checkpoint_path=Path(tmp) / "nightly_v2.pkl",
            )
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(payload["baseline"]["surrogate_enabled"])
            self.assertTrue(payload["surrogate_run"]["surrogate_enabled"])
            self.assertEqual(payload["training"]["sample_count"], 2500)
            self.assertEqual(
                payload["surrogate_run"]["resumed_from"], baseline.archive_jsonl_path
            )


if __name__ == "__main__":
    unittest.main()

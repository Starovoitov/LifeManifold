"""Unit tests for MAP-Elites nightly post-run validation (no full 500k run)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldspace.illuminators.archive import load_and_collapse_jsonl
from worldspace.illuminators.illuminator import MapElitesRunResult
from worldspace.illuminators.nightly_report import (
    build_nightly_report,
    write_nightly_summary,
)
from worldspace.illuminators.scheduler import (
    DEFAULT_MINI_SCHEDULER_PATH,
    RunCounters,
    load_scheduler,
)
from tests.test_map_elites_archive import _record_for_bin

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_JSONL = (
    _REPO_ROOT / "artifacts" / "map_elites_smoke" / "map_elites_archive.jsonl"
)


class TestMapElitesNightlyReport(unittest.TestCase):
    """Validate summary builder against an existing smoke archive when present."""

    def test_build_and_write_summary_from_smoke_jsonl(self) -> None:
        if not _SMOKE_JSONL.is_file():
            self.skipTest("smoke archive missing; run make smoke-map-elites first")
        config = load_scheduler(DEFAULT_MINI_SCHEDULER_PATH)
        collapsed = load_and_collapse_jsonl(
            _SMOKE_JSONL, resolution=config.grid_resolution
        )
        filled_cells = collapsed.filled_count()
        evaluations = config.iterations * config.batch_size
        result = MapElitesRunResult(
            iterations=config.iterations,
            evaluations=evaluations,
            filled_cells=filled_cells,
            archive_jsonl_path=_SMOKE_JSONL,
            counters=RunCounters(candidates_evaluated=evaluations),
        )
        report = build_nightly_report(
            result=result,
            config=config,
            scheduler_path=DEFAULT_MINI_SCHEDULER_PATH,
            seed=42,
            elapsed_seconds=1.5,
        )
        self.assertEqual(report.filled_cells, report.jsonl_collapsed_cells)
        self.assertLessEqual(report.jsonl_collapsed_cells, config.grid_resolution**2)
        self.assertFalse(report.llm_enabled)
        self.assertGreater(report.jsonl_raw_lines, 0)

        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "nightly_run_summary.json"
            write_nightly_summary(summary_path, report)
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["filled_cells"], report.filled_cells)
            self.assertIn("coverage", payload)
            self.assertEqual(
                payload["jsonl_collapsed_cells"], report.jsonl_collapsed_cells
            )

    def test_skips_invalid_jsonl_line_like_collapse(self) -> None:
        config = load_scheduler(DEFAULT_MINI_SCHEDULER_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "archive.jsonl"
            jsonl_path.write_text(
                "not json\n"
                + json.dumps(_record_for_bin((0, 0), 0.4, elite_id="ok"))
                + "\n",
                encoding="utf-8",
            )
            result = MapElitesRunResult(
                iterations=1,
                evaluations=1,
                filled_cells=1,
                archive_jsonl_path=jsonl_path,
                counters=RunCounters(candidates_evaluated=1),
            )
            with self.assertLogs("worldspace.illuminators.archive", level="WARNING"):
                report = build_nightly_report(
                    result=result,
                    config=config,
                    scheduler_path=DEFAULT_MINI_SCHEDULER_PATH,
                    seed=0,
                    elapsed_seconds=0.0,
                )
            self.assertEqual(report.filled_cells, 1)
            self.assertEqual(report.jsonl_raw_lines, 1)
            self.assertEqual(report.jsonl_collapsed_cells, 1)

    def test_skips_blank_jsonl_lines_like_collapse(self) -> None:
        config = load_scheduler(DEFAULT_MINI_SCHEDULER_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "archive.jsonl"
            record_line = json.dumps(_record_for_bin((0, 0), 0.4, elite_id="ok"))
            jsonl_path.write_text(f"\n{record_line}\n\n\n", encoding="utf-8")
            result = MapElitesRunResult(
                iterations=1,
                evaluations=1,
                filled_cells=1,
                archive_jsonl_path=jsonl_path,
                counters=RunCounters(candidates_evaluated=1),
            )
            report = build_nightly_report(
                result=result,
                config=config,
                scheduler_path=DEFAULT_MINI_SCHEDULER_PATH,
                seed=0,
                elapsed_seconds=0.0,
            )
            self.assertEqual(report.filled_cells, 1)
            self.assertEqual(report.jsonl_raw_lines, 1)
            self.assertEqual(report.jsonl_collapsed_cells, 1)

    def test_filled_cells_mismatch_raises(self) -> None:
        if not _SMOKE_JSONL.is_file():
            self.skipTest("smoke archive missing")
        config = load_scheduler(DEFAULT_MINI_SCHEDULER_PATH)
        result = MapElitesRunResult(
            iterations=1,
            evaluations=4,
            filled_cells=9999,
            archive_jsonl_path=_SMOKE_JSONL,
            counters=RunCounters(candidates_evaluated=4),
        )
        with self.assertRaises(RuntimeError):
            build_nightly_report(
                result=result,
                config=config,
                scheduler_path=DEFAULT_MINI_SCHEDULER_PATH,
                seed=0,
                elapsed_seconds=0.0,
            )


if __name__ == "__main__":
    unittest.main()

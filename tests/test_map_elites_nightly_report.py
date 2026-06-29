"""Unit tests for MAP-Elites nightly post-run validation (no full 500k run)."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from worldspace.illuminators.archive import (
    load_and_collapse_jsonl,
)
from worldspace.illuminators.illuminator import MapElitesRunResult
from worldspace.illuminators.nightly_report import (
    LLM_STACK_VERSION,
    _collapsed_archive_for_validation,
    build_nightly_report,
    write_nightly_summary,
)
from worldspace.illuminators.scheduler import (
    DEFAULT_MINI_SCHEDULER_PATH,
    RunCounters,
    load_scheduler,
)
from worldspace.simulator_perf import DEFAULT_SIMULATOR_PERFORMANCE
from tests.test_map_elites_archive import _record_for_bin

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LLM_SPEC = _REPO_ROOT / "worldspace" / "specs" / "llm_world_generator_qwen.yaml"
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
        self.assertLessEqual(report.jsonl_collapsed_cells, config.n_cells)
        self.assertEqual(report.archive_type, "grid")
        self.assertEqual(report.n_cells, config.grid_resolution**2)
        self.assertFalse(report.llm_enabled)
        self.assertFalse(report.surrogate_enabled)
        self.assertGreater(report.jsonl_raw_lines, 0)

        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "nightly_run_summary.json"
            write_nightly_summary(summary_path, report)
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["filled_cells"], report.filled_cells)
            self.assertIn("coverage", payload)
            self.assertEqual(payload["archive_type"], "grid")
            self.assertEqual(payload["n_cells"], report.n_cells)
            self.assertEqual(
                payload["jsonl_collapsed_cells"], report.jsonl_collapsed_cells
            )
            self.assertEqual(payload["surrogate_enabled"], report.surrogate_enabled)

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

    def test_build_nightly_report_merges_resume_archive(self) -> None:
        """Phase-3 JSONL is append-only delta; validation must merge baseline."""
        config = load_scheduler(DEFAULT_MINI_SCHEDULER_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp) / "baseline.jsonl"
            run_path = Path(tmp) / "run.jsonl"
            lines = [
                json.dumps(_record_for_bin((0, 0), 0.4, elite_id="base-a")),
                json.dumps(_record_for_bin((1, 1), 0.5, elite_id="base-b")),
            ]
            base_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            run_path.write_text(
                json.dumps(_record_for_bin((2, 2), 0.6, elite_id="run-c")) + "\n",
                encoding="utf-8",
            )
            merged = _collapsed_archive_for_validation(
                run_path,
                config=config,
                resume_archive_path=base_path,
            )
            self.assertEqual(merged.filled_count(), 3)
            result = MapElitesRunResult(
                iterations=1,
                evaluations=4,
                filled_cells=3,
                archive_jsonl_path=run_path,
                counters=RunCounters(candidates_evaluated=4),
            )
            report = build_nightly_report(
                result=result,
                config=config,
                scheduler_path=DEFAULT_MINI_SCHEDULER_PATH,
                seed=0,
                elapsed_seconds=0.0,
                resume_archive_path=base_path,
            )
            self.assertEqual(report.filled_cells, 3)
            self.assertEqual(report.jsonl_collapsed_cells, 3)
            self.assertEqual(report.jsonl_raw_lines, 1)

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

    def test_build_nightly_report_llm_observability_fields(self) -> None:
        config = replace(
            load_scheduler(DEFAULT_MINI_SCHEDULER_PATH),
            llm_enabled=True,
            performance=replace(
                DEFAULT_SIMULATOR_PERFORMANCE,
                llm_parallel_emit=True,
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "archive.jsonl"
            jsonl_path.write_text(
                json.dumps(_record_for_bin((0, 0), 0.4, elite_id="ok")) + "\n",
                encoding="utf-8",
            )
            counters = RunCounters(
                candidates_evaluated=4,
                llm_emit_attempts=10,
                llm_emit_fallbacks=1,
                emit_llm_seconds=12.345,
                eval_seconds=3.21,
            )
            result = MapElitesRunResult(
                iterations=1,
                evaluations=4,
                filled_cells=1,
                archive_jsonl_path=jsonl_path,
                counters=counters,
            )
            report = build_nightly_report(
                result=result,
                config=config,
                scheduler_path=DEFAULT_MINI_SCHEDULER_PATH,
                seed=7,
                elapsed_seconds=20.0,
                llm_spec_path=_LLM_SPEC,
            )
            self.assertEqual(report.llm_stack_version, LLM_STACK_VERSION)
            self.assertEqual(report.llm_model, "qwen-turbo")
            self.assertTrue(report.llm_parallel_emit)
            self.assertEqual(report.llm_parallel_workers, 1)
            self.assertEqual(report.llm_emit_attempts, 10)
            self.assertEqual(report.llm_emit_fallbacks, 1)
            self.assertAlmostEqual(report.llm_fallback_rate_pct, 10.0)
            self.assertEqual(report.emit_llm_seconds, 12.345)
            self.assertEqual(report.eval_seconds, 3.21)
            self.assertIsNotNone(report.prompt_version)

            summary_path = Path(tmp) / "nightly_run_summary.json"
            write_nightly_summary(summary_path, report)
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["llm_stack_version"], LLM_STACK_VERSION)
            self.assertEqual(payload["llm_model"], "qwen-turbo")
            self.assertEqual(payload["llm_fallback_rate_pct"], 10.0)


if __name__ == "__main__":
    unittest.main()

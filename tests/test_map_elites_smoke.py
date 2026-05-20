"""MAP-Elites CI smoke: mini-scheduler end-to-end with persistent run artifacts."""

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path

from worldspace.illuminators.archive import (
    ARCHIVE_SCHEMA_VERSION,
    archive_record_to_elite,
    load_and_collapse_jsonl,
)
from worldspace.illuminators.illuminator import (
    MapElitesIlluminator,
    archive_jsonl_path,
)
from worldspace.illuminators.scheduler import (
    DEFAULT_MINI_SCHEDULER_PATH,
    load_scheduler,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_OUTPUT_DIR = _REPO_ROOT / "artifacts" / "map_elites_smoke"
SMOKE_SEED = 42
SMOKE_GRID_SIZE = 8
SMOKE_STEPS = 200
_MAX_SMOKE_SECONDS = 120.0


def _validate_jsonl(path: Path, *, resolution: int) -> int:
    """Parse every JSONL line; return line count."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return 0
    lines = text.splitlines()
    for line in lines:
        record = json.loads(line)
        elite = archive_record_to_elite(record)
        if elite.bin[0] < 0 or elite.bin[0] >= resolution:
            msg = f"bin i out of range: {elite.bin}"
            raise ValueError(msg)
        if elite.bin[1] < 0 or elite.bin[1] >= resolution:
            msg = f"bin j out of range: {elite.bin}"
            raise ValueError(msg)
        if record.get("schema_version") != ARCHIVE_SCHEMA_VERSION:
            msg = f"unexpected schema_version: {record.get('schema_version')}"
            raise ValueError(msg)
    return len(lines)


class TestMapElitesSmoke(unittest.TestCase):
    """End-to-end smoke using the mini scheduler (no LLM network calls)."""

    def test_mini_scheduler_smoke_leaves_artifacts(self) -> None:
        config = load_scheduler(DEFAULT_MINI_SCHEDULER_PATH)
        self.assertFalse(config.llm_enabled)
        expected_evaluations = config.iterations * config.batch_size

        SMOKE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        result = MapElitesIlluminator().run(
            scheduler_path=DEFAULT_MINI_SCHEDULER_PATH,
            output_dir=SMOKE_OUTPUT_DIR,
            seed=SMOKE_SEED,
            grid_size=SMOKE_GRID_SIZE,
            steps=SMOKE_STEPS,
        )
        elapsed = time.perf_counter() - started

        jsonl_path = archive_jsonl_path(SMOKE_OUTPUT_DIR)
        self.assertEqual(result.archive_jsonl_path, jsonl_path)
        self.assertTrue(jsonl_path.is_file(), "archive JSONL must exist on disk")
        self.assertGreater(result.filled_cells, 0, "archive must not be empty")
        self.assertEqual(result.evaluations, expected_evaluations)
        self.assertLess(
            elapsed,
            _MAX_SMOKE_SECONDS,
            f"smoke exceeded {_MAX_SMOKE_SECONDS}s budget",
        )

        line_count = _validate_jsonl(jsonl_path, resolution=config.grid_resolution)
        self.assertGreater(line_count, 0)
        self.assertLessEqual(line_count, config.grid_resolution**2)

        collapsed = load_and_collapse_jsonl(
            jsonl_path, resolution=config.grid_resolution
        )
        self.assertEqual(collapsed.filled_count(), result.filled_cells)

        summary_path = SMOKE_OUTPUT_DIR / "smoke_run_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": ARCHIVE_SCHEMA_VERSION,
                    "scheduler": str(DEFAULT_MINI_SCHEDULER_PATH),
                    "seed": SMOKE_SEED,
                    "iterations": result.iterations,
                    "evaluations": result.evaluations,
                    "filled_cells": result.filled_cells,
                    "jsonl_lines": line_count,
                    "elapsed_seconds": round(elapsed, 3),
                    "llm_enabled": config.llm_enabled,
                },
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.assertTrue(summary_path.is_file())


if __name__ == "__main__":
    unittest.main()

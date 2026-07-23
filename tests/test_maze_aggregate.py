"""Tests for maze experiment aggregation."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestMazeAggregate(unittest.TestCase):
    def test_summary_csv_includes_maze_benchmark_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "genetic_filter" / "seed_0"
            run_dir.mkdir(parents=True)
            summary = {
                "schema_version": "maze-1.0",
                "benchmark": "maze",
                "maze_benchmark": True,
                "condition": "genetic_filter",
                "seed": 0,
                "iterations": 5,
                "proposals": 250,
                "evaluations": 180,
                "skipped": 70,
                "skip_rate": 0.28,
                "filled_cells": 27,
                "coverage": 0.03,
                "mean_best_fitness": 0.53,
                "qd_score": 14.3,
                "archive_type": "grid",
                "grid_resolution": 30,
                "llm_enabled": False,
                "surrogate_enabled": True,
                "archive_jsonl": str((run_dir / "maze_archive.jsonl").resolve()),
                "elapsed_seconds": 1.0,
            }
            (run_dir / "nightly_run_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n",
                encoding="utf-8",
            )
            (run_dir / "maze_archive.jsonl").write_text("", encoding="utf-8")
            output = root / "summary.csv"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/aggregate_experiment_runs.py"),
                    "--root",
                    str(root),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Wrote 1 rows", completed.stdout)
            with output.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["condition"], "genetic_filter")
            self.assertEqual(rows[0]["maze_benchmark"], "True")
            self.assertEqual(rows[0]["skip_rate_pct"], "28.0")


if __name__ == "__main__":
    unittest.main()

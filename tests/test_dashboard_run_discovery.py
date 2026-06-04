"""Unit tests for MAP-Elites run discovery (no Streamlit)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_ARCHIVE = (
    _REPO_ROOT / "artifacts" / "map_elites_smoke" / "map_elites_archive.jsonl"
)


class TestScanDirEntries(unittest.TestCase):
    def test_config_run_scan_dirs_replaces_defaults(self) -> None:
        from dashboard.utils.run_discovery import _scan_dir_entries

        cfg = {"paths": {"run_scan_dirs": ["custom/*", "artifacts/*"]}}
        entries = _scan_dir_entries(cfg)
        self.assertEqual(entries, ["custom/*", "artifacts/*"])

    def test_empty_or_missing_run_scan_dirs_uses_defaults(self) -> None:
        from dashboard.utils.run_discovery import (
            _scan_dir_entries,
            default_scan_dir_entries,
        )

        defaults = default_scan_dir_entries()
        self.assertEqual(_scan_dir_entries({}), defaults)
        self.assertEqual(_scan_dir_entries({"paths": {}}), defaults)
        self.assertEqual(_scan_dir_entries({"paths": {"run_scan_dirs": []}}), defaults)


class TestExpandScanDirEntries(unittest.TestCase):
    def test_literal_path_unchanged(self) -> None:
        from dashboard.utils.run_discovery import expand_scan_dir_entries

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = expand_scan_dir_entries([str(root)])
            self.assertEqual(entries, [root.resolve()])

    def test_wildcard_expands_immediate_subdirs(self) -> None:
        from dashboard.utils.run_discovery import expand_scan_dir_entries

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "artifacts"
            run_a = base / "run_a"
            run_b = base / "run_b"
            run_a.mkdir(parents=True)
            run_b.mkdir(parents=True)
            (base / "skip.txt").write_text("", encoding="utf-8")

            entries = expand_scan_dir_entries([str(base / "*")])
            self.assertEqual(
                {path.resolve() for path in entries},
                {run_a.resolve(), run_b.resolve()},
            )

    def test_wildcard_no_match_returns_empty(self) -> None:
        from dashboard.utils.run_discovery import expand_scan_dir_entries

        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "nowhere"
            empty.mkdir()
            self.assertEqual(
                expand_scan_dir_entries([str(empty / "missing_*")]),
                [],
            )

    def test_discover_via_wildcard_scan_dir(self) -> None:
        from dashboard.utils.run_discovery import discover_runs

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "artifacts"
            run_dir = base / "custom_run"
            run_dir.mkdir(parents=True)
            archive = run_dir / "map_elites_archive.jsonl"
            archive.write_text("{}\n", encoding="utf-8")
            cfg = {
                "paths": {
                    "archives": [],
                    "run_scan_dirs": [str(base / "*")],
                }
            }
            runs = discover_runs(cfg)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].archive_path.resolve(), archive.resolve())


class TestDashboardRunDiscovery(unittest.TestCase):
    def test_discover_finds_smoke_archive_when_present(self) -> None:
        from dashboard.utils.run_discovery import discover_runs

        if not _SMOKE_ARCHIVE.is_file():
            self.skipTest("smoke archive not on disk")
        runs = discover_runs()
        archives = {run.archive_path.resolve() for run in runs}
        self.assertIn(_SMOKE_ARCHIVE.resolve(), archives)

    def test_discover_archive_without_summary(self) -> None:
        from dashboard.utils.run_discovery import discover_runs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "map_elites_archive.jsonl"
            archive.write_text("{}\n", encoding="utf-8")
            cfg = {
                "paths": {
                    "archives": [],
                    "run_scan_dirs": [str(root)],
                }
            }
            runs = discover_runs(cfg)
            self.assertEqual(len(runs), 1)
            self.assertIsNone(runs[0].summary_path)
            self.assertIsNone(runs[0].summary)

    def test_discover_pairs_archive_with_nightly_summary(self) -> None:
        from dashboard.utils.run_discovery import discover_runs, summary_get

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "map_elites_archive.jsonl"
            archive.write_text("{}\n", encoding="utf-8")
            summary = root / "nightly_run_summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "filled_cells": 42,
                        "coverage": 0.0168,
                        "seed": 7,
                        "llm_enabled": True,
                        "surrogate_enabled": False,
                    }
                ),
                encoding="utf-8",
            )
            cfg = {"paths": {"archives": [], "run_scan_dirs": [str(root)]}}
            runs = discover_runs(cfg)
            self.assertEqual(len(runs), 1)
            self.assertIsNotNone(runs[0].summary)
            assert runs[0].summary is not None
            self.assertEqual(summary_get(runs[0].summary, "filled_cells"), 42)
            self.assertEqual(summary_get(runs[0].summary, "seed"), 7)

    def test_load_summary_json_invalid_returns_none(self) -> None:
        from dashboard.utils.run_discovery import load_summary_json

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("not json", encoding="utf-8")
            self.assertIsNone(load_summary_json(path))

    def test_smoke_summary_fields_if_file_exists(self) -> None:
        from dashboard.utils.run_discovery import discover_runs, summary_get

        summary_path = _SMOKE_ARCHIVE.parent / "smoke_run_summary.json"
        if not summary_path.is_file():
            self.skipTest("smoke_run_summary.json not on disk")
        runs = [r for r in discover_runs() if r.summary_path == summary_path]
        self.assertGreaterEqual(len(runs), 1)
        summary = runs[0].summary
        self.assertIsNotNone(summary)
        assert summary is not None
        disk = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(
            summary_get(summary, "filled_cells"),
            disk.get("filled_cells"),
        )
        self.assertEqual(summary_get(summary, "seed"), disk.get("seed"))


if __name__ == "__main__":
    unittest.main()

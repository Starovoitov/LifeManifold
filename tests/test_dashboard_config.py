"""Unit tests for dashboard config path resolution (no Streamlit)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_ARCHIVE = (
    _REPO_ROOT / "artifacts" / "map_elites_smoke" / "map_elites_archive.jsonl"
)


class TestSortArchivePathsByMtime(unittest.TestCase):
    def test_sorts_newest_first_and_skips_missing(self) -> None:
        from dashboard.utils.config import sort_archive_paths_by_mtime

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = root / "old.jsonl"
            newer = root / "new.jsonl"
            older.write_text("{}", encoding="utf-8")
            newer.write_text("{}", encoding="utf-8")
            old_mtime = older.stat().st_mtime
            new_mtime = newer.stat().st_mtime
            deleted = root / "gone.jsonl"

            result = sort_archive_paths_by_mtime(
                [
                    (older, old_mtime),
                    (deleted, old_mtime),
                    (newer, new_mtime),
                ]
            )
            self.assertEqual(result, [newer, older])


class TestDashboardConfig(unittest.TestCase):
    def test_repo_root_points_at_repository(self) -> None:
        from dashboard.utils.config import repo_root

        self.assertEqual(repo_root(), _REPO_ROOT)

    def test_existing_archive_paths_includes_run_scan_discovery(self) -> None:
        from dashboard.utils.config import existing_archive_paths, load_config

        cfg = load_config()
        self.assertEqual(cfg["defaults"]["grid_resolution"], 50)
        paths = existing_archive_paths(cfg)
        baseline = _REPO_ROOT / "artifacts" / "baseline" / "map_elites_archive.jsonl"
        surrogate = _REPO_ROOT / "artifacts" / "surrogate" / "map_elites_archive.jsonl"
        resolved = {p.resolve() for p in paths}
        if baseline.is_file():
            self.assertIn(baseline.resolve(), resolved)
        if surrogate.is_file():
            self.assertIn(surrogate.resolve(), resolved)
        if not baseline.is_file() and not surrogate.is_file():
            self.skipTest(
                "no artifacts/baseline or artifacts/surrogate archives on disk"
            )


if __name__ == "__main__":
    unittest.main()

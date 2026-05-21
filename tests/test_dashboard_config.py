"""Unit tests for dashboard config path resolution (no Streamlit)."""

from __future__ import annotations

import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_ARCHIVE = (
    _REPO_ROOT / "artifacts" / "map_elites_smoke" / "map_elites_archive.jsonl"
)


class TestDashboardConfig(unittest.TestCase):
    def test_repo_root_points_at_repository(self) -> None:
        from dashboard.utils.config import repo_root

        self.assertEqual(repo_root(), _REPO_ROOT)

    def test_smoke_archive_resolves_and_exists(self) -> None:
        from dashboard.utils.config import existing_archive_paths, load_config

        cfg = load_config()
        self.assertEqual(cfg["defaults"]["grid_resolution"], 50)
        paths = existing_archive_paths(cfg)
        self.assertTrue(
            any(p.resolve() == _SMOKE_ARCHIVE.resolve() for p in paths),
            msg=f"expected smoke archive among {paths}",
        )


if __name__ == "__main__":
    unittest.main()

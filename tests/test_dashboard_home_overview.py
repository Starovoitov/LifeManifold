"""Unit tests for Home overview helpers."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from tests.test_dashboard_archive_loader import _cvt_fixture_dir


class TestDashboardHomeOverview(unittest.TestCase):
    def test_archive_run_stats_includes_cvt_metadata(self) -> None:
        from dashboard.components.home_overview import archive_run_stats

        with tempfile.TemporaryDirectory() as tmp:
            archive_path = _cvt_fixture_dir(tmp)
            stats = archive_run_stats(str(archive_path), 0.0)
        self.assertEqual(stats["archive_type"], "cvt")
        self.assertEqual(stats["n_cells"], 9)
        self.assertGreater(stats["filled_cells"], 0)
        self.assertIn("coverage", stats)
        self.assertLessEqual(stats["coverage"], 1.0)

    def test_render_reproducibility_block_includes_archive_type(self) -> None:
        from dashboard.components.home_overview import render_reproducibility_block

        with patch("dashboard.components.home_overview.st") as mock_st:
            render_reproducibility_block(
                {
                    "scheduler": "mini_cvt.yaml",
                    "seed": 42,
                    "archive_type": "cvt",
                    "n_cells": 25,
                    "grid_resolution": 5,
                    "schema_version": "1.3",
                }
            )
        markdown_args = [str(call) for call in mock_st.markdown.call_args_list]
        joined = " ".join(markdown_args)
        self.assertIn("archive type", joined.lower())
        self.assertIn("cvt", joined)
        self.assertIn("n_cells", joined.lower())

    def test_format_percent_expects_ratio_not_percentage(self) -> None:
        from dashboard.components.home_overview import _format_percent

        self.assertEqual(_format_percent(None), "—")
        self.assertEqual(_format_percent(0.0168), "1.68%")
        self.assertEqual(_format_percent(0.32), "32.00%")
        self.assertEqual(_format_percent(1.0), "100.00%")
        self.assertEqual(_format_percent(1.5), "150.00%")

    def test_run_expander_title_uses_archive_stats_without_summary(self) -> None:
        from dashboard.components.home_overview import _run_expander_title
        from dashboard.utils.config import repo_root
        from dashboard.utils.run_discovery import RunInfo

        run_dir = (
            repo_root() / "artifacts" / "experiments" / "q1-min" / "stub" / "seed_0"
        )
        run = RunInfo(
            run_dir=run_dir,
            archive_path=run_dir / "map_elites_archive.jsonl",
            summary_path=None,
            summary=None,
            archive_mtime=0.0,
        )
        title = _run_expander_title(
            run,
            stats={
                "archive_type": "grid",
                "filled_cells": 743,
                "coverage": 0.02972,
            },
        )
        self.assertIn("grid", title)
        self.assertIn("filled=743", title)
        self.assertIn("coverage=3.0%", title)

    def test_list_niches_from_frame_grid_and_cvt(self) -> None:
        import pandas as pd

        from dashboard.components.archive_explorer import list_niches_from_frame

        grid_frame = pd.DataFrame(
            {
                "bin_x": [0, 1],
                "bin_y": [0, 1],
                "cell_id": [0, 1],
                "fitness": [0.2, 0.8],
            }
        )
        cvt_frame = pd.DataFrame(
            {
                "cell_id": [0, 3, 5],
                "bin_x": [0, 3, 5],
                "bin_y": [0, 0, 0],
                "fitness": [0.3, 0.6, 0.9],
            }
        )
        self.assertEqual(list_niches_from_frame(grid_frame, "grid"), [(1, 1), (0, 0)])
        self.assertEqual(list_niches_from_frame(cvt_frame, "cvt"), [5, 3, 0])


if __name__ == "__main__":
    unittest.main()

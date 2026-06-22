"""Unit tests for Archive Explorer helpers (no Streamlit runtime)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_ARCHIVE = (
    _REPO_ROOT / "artifacts" / "map_elites_smoke" / "map_elites_archive.jsonl"
)


class TestDashboardArchiveExplorer(unittest.TestCase):
    def _smoke_collapsed(self) -> pd.DataFrame:
        from dashboard.components.archive_loader import collapse_dataframe
        from dashboard.utils.data_processing import flatten_archive_record

        rows = []
        for line in _SMOKE_ARCHIVE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(flatten_archive_record(json.loads(line)))
        return collapse_dataframe(pd.DataFrame(rows))

    def test_elite_row_for_bin_finds_smoke_elite(self) -> None:
        from dashboard.components.archive_explorer import elite_row_for_bin

        frame = self._smoke_collapsed()
        first = frame.iloc[0]
        bin_x = int(first.at["bin_x"])
        bin_y = int(first.at["bin_y"])
        row = elite_row_for_bin(frame, bin_x, bin_y)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(int(row.at["bin_x"]), bin_x)
        self.assertEqual(int(row.at["bin_y"]), bin_y)

    def test_elite_row_for_bin_missing_returns_none(self) -> None:
        from dashboard.components.archive_explorer import elite_row_for_bin

        frame = self._smoke_collapsed()
        self.assertIsNone(elite_row_for_bin(frame, 999, 999))

    def test_list_bins_from_frame_matches_collapsed_rows(self) -> None:
        from dashboard.components.archive_explorer import list_bins_from_frame

        frame = self._smoke_collapsed()
        bins = list_bins_from_frame(frame)
        self.assertEqual(len(bins), len(frame))
        first = frame.iloc[0]
        self.assertEqual(bins[0], (int(first.at["bin_x"]), int(first.at["bin_y"])))

    def test_format_elite_bin_label_contains_fitness(self) -> None:
        from dashboard.components.archive_explorer import format_elite_bin_label

        frame = self._smoke_collapsed()
        first = frame.iloc[0]
        label = format_elite_bin_label(
            frame,
            (int(first.at["bin_x"]), int(first.at["bin_y"])),
        )
        self.assertIn("fitness=", label)
        self.assertIn("bin (", label)

    def test_list_cells_from_cvt_frame(self) -> None:
        from dashboard.components.archive_explorer import (
            format_elite_cell_label,
            list_cells_from_frame,
        )
        from dashboard.components.archive_loader import load_archive_bundle
        from dashboard.utils.config import load_config
        from tests.test_dashboard_archive_loader import _cvt_fixture_dir

        with __import__("tempfile").TemporaryDirectory() as tmp:
            path = _cvt_fixture_dir(tmp)
            cfg = load_config()
            bundle = load_archive_bundle(path, path.stat().st_mtime, cfg)
            cells = list_cells_from_frame(bundle.collapsed)
            self.assertEqual(cells, [0, 3, 5])
            label = format_elite_cell_label(bundle.collapsed, cells[0])
            self.assertIn("cell 0", label)
            self.assertIn("fitness=", label)

    def test_diagnostic_chart_key_uses_bin_for_grid_schema_1_3(self) -> None:
        from dashboard.components.archive_explorer import (
            _diagnostic_niche_label,
            diagnostic_chart_key,
        )

        row = {
            "archive_type": "grid",
            "cell_id": 23,
            "bin_x": 2,
            "bin_y": 3,
            "fitness": 0.5,
        }
        self.assertEqual(_diagnostic_niche_label(row), "bin (2, 3)")
        self.assertEqual(
            diagnostic_chart_key(row, "abcdef0123456789"),
            "explorer_diagnostic_bin_2_3_abcdef0123456789",
        )

    def test_diagnostic_chart_key_uses_cell_for_cvt(self) -> None:
        from dashboard.components.archive_explorer import (
            _diagnostic_niche_label,
            diagnostic_chart_key,
        )

        row = {
            "archive_type": "cvt",
            "cell_id": 7,
            "bin_x": 7,
            "bin_y": 0,
            "fitness": 0.5,
        }
        self.assertEqual(_diagnostic_niche_label(row), "cell 7")
        self.assertEqual(
            diagnostic_chart_key(row, "deadbeef01234567"),
            "explorer_diagnostic_cell_7_deadbeef01234567",
        )


if __name__ == "__main__":
    unittest.main()

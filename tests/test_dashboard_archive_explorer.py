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


if __name__ == "__main__":
    unittest.main()

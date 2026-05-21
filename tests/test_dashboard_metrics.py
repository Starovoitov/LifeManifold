"""Unit tests for dashboard metric helpers."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from worldspace.metrics import METRIC_KEYS

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_ARCHIVE = (
    _REPO_ROOT / "artifacts" / "map_elites_smoke" / "map_elites_archive.jsonl"
)


class TestDashboardMetrics(unittest.TestCase):
    def test_metrics_dict_from_row_has_all_keys(self) -> None:
        from dashboard.utils.data_processing import flatten_archive_record
        from dashboard.components.metrics import metrics_dict_from_row

        record = json.loads(_SMOKE_ARCHIVE.read_text(encoding="utf-8").splitlines()[0])
        row = flatten_archive_record(record)
        metrics = metrics_dict_from_row(row)
        for key in METRIC_KEYS:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)


if __name__ == "__main__":
    unittest.main()

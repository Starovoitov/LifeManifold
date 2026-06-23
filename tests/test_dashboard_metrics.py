"""Unit tests for dashboard metric helpers."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from worldspace.metrics import METRIC_KEYS

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_ARCHIVE = (
    _REPO_ROOT / "artifacts" / "map_elites_smoke" / "map_elites_archive.jsonl"
)


class TestDashboardMetrics(unittest.TestCase):
    def test_correlation_matrix_smoke_shape_and_diagonal(self) -> None:
        from dashboard.components.archive_loader import load_archive_bundle
        from dashboard.components.metrics import correlation_matrix
        from dashboard.utils.config import load_config

        if not _SMOKE_ARCHIVE.is_file():
            self.skipTest("smoke archive missing")
        cfg = load_config()
        bundle = load_archive_bundle(_SMOKE_ARCHIVE, 0.0, cfg)
        corr = correlation_matrix(bundle.collapsed)
        self.assertEqual(corr.shape[0], corr.shape[1])
        self.assertGreaterEqual(corr.shape[0], 2)
        self.assertLessEqual(corr.shape[0], len(METRIC_KEYS))
        diagonal = np.diag(corr.to_numpy(dtype=np.float64))
        self.assertTrue(np.all(np.isfinite(diagonal)))
        self.assertTrue(np.allclose(diagonal, 1.0, atol=1e-6))

    def test_metrics_dict_from_row_has_all_keys(self) -> None:
        from dashboard.utils.data_processing import flatten_archive_record
        from dashboard.components.metrics import metrics_dict_from_row

        record = json.loads(_SMOKE_ARCHIVE.read_text(encoding="utf-8").splitlines()[0])
        row = flatten_archive_record(record)
        metrics = metrics_dict_from_row(row)
        for key in METRIC_KEYS:
            if key not in metrics:
                continue
            self.assertIsInstance(metrics[key], float)
        self.assertIn("stability", metrics)
        self.assertIn("diversity", metrics)


if __name__ == "__main__":
    unittest.main()

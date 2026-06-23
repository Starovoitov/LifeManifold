"""Unit tests for in-memory archive filters."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_ARCHIVE = (
    _REPO_ROOT / "artifacts" / "map_elites_smoke" / "map_elites_archive.jsonl"
)


class TestDashboardFilters(unittest.TestCase):
    def test_min_fitness_filter_reduces_rows(self) -> None:
        from dashboard.components.archive_loader import load_archive_bundle
        from dashboard.components.filters import (
            FilterState,
            apply_collapsed_filters,
            rebuild_pivots_from_collapsed,
        )
        from dashboard.utils.config import load_config

        cfg = load_config()
        bundle = load_archive_bundle(_SMOKE_ARCHIVE, 0.0, cfg)
        state = FilterState(
            archive_path=_SMOKE_ARCHIVE,
            heatmap_metric="fitness",
            min_fitness=0.5,
            seed=None,
            emitter_type=None,
            resolution=bundle.resolution,
        )
        filtered = apply_collapsed_filters(bundle.collapsed, state)
        self.assertLessEqual(len(filtered), len(bundle.collapsed))
        pivots = rebuild_pivots_from_collapsed(
            filtered,
            list(bundle.pivots.keys()),
            bundle.resolution,
        )
        self.assertEqual(pivots["fitness"].shape, (50, 50))

    def test_langton_lambda_range_filter(self) -> None:
        from dashboard.components.filters import FilterState, apply_collapsed_filters

        collapsed = pd.DataFrame(
            {
                "fitness": [0.2, 0.6, 0.8],
                "langton_lambda_runtime": [0.05, 0.20, 0.40],
            }
        )
        state = FilterState(
            archive_path=_SMOKE_ARCHIVE,
            heatmap_metric="fitness",
            min_fitness=0.0,
            seed=None,
            emitter_type=None,
            resolution=50,
            langton_lambda_min=0.15,
            langton_lambda_max=0.35,
        )
        filtered = apply_collapsed_filters(collapsed, state)
        self.assertEqual(len(filtered), 1)
        self.assertAlmostEqual(float(filtered.iloc[0]["langton_lambda_runtime"]), 0.20)


if __name__ == "__main__":
    unittest.main()

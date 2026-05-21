"""Unit tests for dashboard Plotly visualizations (no Streamlit runtime)."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_ARCHIVE = (
    _REPO_ROOT / "artifacts" / "map_elites_smoke" / "map_elites_archive.jsonl"
)


class TestDashboardVisualizations(unittest.TestCase):
    def test_create_archive_heatmap_from_smoke_pivot(self) -> None:
        from dashboard.components.archive_loader import load_archive_bundle
        from dashboard.components.visualizations import create_archive_heatmap
        from dashboard.utils.config import load_config

        cfg = load_config()
        bundle = load_archive_bundle(_SMOKE_ARCHIVE, 0.0, cfg)
        fig = create_archive_heatmap(
            pivot=bundle.pivots["fitness"],
            metric="fitness",
            resolution=50,
        )
        self.assertGreater(len(fig.data), 0)
        self.assertEqual(fig.layout.height, 620)
        x_title = fig.layout.xaxis.title.text
        y_title = fig.layout.yaxis.title.text
        self.assertIn("Diversity", x_title)
        self.assertIn("Stability", y_title)

    def test_create_metrics_radar_with_enough_keys(self) -> None:
        from dashboard.components.visualizations import create_metrics_radar

        metrics = {
            "stability": 0.8,
            "diversity": 0.3,
            "topology_interface_index": 0.5,
            "fitness": 0.42,
        }
        fig = create_metrics_radar(metrics)
        self.assertEqual(fig.data[0].type, "scatterpolar")

    def test_add_boundary_overlay_skips_none(self) -> None:
        from dashboard.components.visualizations import add_boundary_overlay
        import plotly.graph_objects as go

        fig = go.Figure()
        add_boundary_overlay(fig, None)
        self.assertEqual(len(fig.data), 0)

    def test_add_boundary_overlay_adds_contour(self) -> None:
        from dashboard.components.visualizations import add_boundary_overlay
        import plotly.graph_objects as go

        fig = go.Figure()
        interface = np.linspace(0.0, 1.0, 16).reshape(4, 4)
        add_boundary_overlay(fig, interface)
        self.assertEqual(fig.data[0].type, "contour")

    def test_plot_real_vs_predicted_has_scatter_and_reference(self) -> None:
        from dashboard.components.visualizations import plot_real_vs_predicted

        rng = np.random.default_rng(0)
        y_true = rng.random(20)
        y_pred = y_true + rng.normal(0.0, 0.05, 20)
        uncertainty = rng.random(20)
        fig = plot_real_vs_predicted(
            y_true,
            y_pred,
            uncertainty,
            metric_name="fitness",
        )
        types = {trace.type for trace in fig.data}
        self.assertIn("scatter", types)
        self.assertGreaterEqual(len(fig.data), 2)

    def test_create_diagnostic_dashboard_stub(self) -> None:
        from dashboard.components.visualizations import create_diagnostic_dashboard

        fig = create_diagnostic_dashboard()
        self.assertIsNotNone(fig.layout.title)
        self.assertGreaterEqual(len(fig.layout.annotations), 1)

    def test_pivot_from_dataframe_resolution_50(self) -> None:
        from dashboard.components.visualizations import _pivot_from_dataframe
        import pandas as pd

        df = pd.DataFrame(
            {
                "bin_x": [0, 1],
                "bin_y": [0, 1],
                "fitness": [0.2, 0.9],
            }
        )
        grid = _pivot_from_dataframe(df, metric="fitness", resolution=50)
        self.assertEqual(grid.shape, (50, 50))
        self.assertEqual(grid[1, 1], 0.9)


if __name__ == "__main__":
    unittest.main()

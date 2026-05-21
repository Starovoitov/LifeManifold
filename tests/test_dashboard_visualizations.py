"""Unit tests for dashboard Plotly visualizations (no Streamlit runtime)."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_ARCHIVE = (
    _REPO_ROOT / "artifacts" / "map_elites_smoke" / "map_elites_archive.jsonl"
)


def _figure_traces(fig: go.Figure) -> tuple[Any, ...]:
    """Plotly stubs type ``fig.data`` as ``Unknown | Figure``; cast for tests."""
    return tuple(cast(Any, fig).data)


def _figure_layout(fig: go.Figure) -> Any:
    """Same stub issue for ``fig.layout`` / nested layout fields."""
    return cast(Any, fig).layout


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
        self.assertGreater(len(_figure_traces(fig)), 0)
        layout = _figure_layout(fig)
        self.assertEqual(layout.height, 620)
        x_title = layout.xaxis.title.text
        y_title = layout.yaxis.title.text
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
        self.assertEqual(_figure_traces(fig)[0].type, "scatterpolar")

    def test_add_boundary_overlay_skips_none(self) -> None:
        from dashboard.components.visualizations import add_boundary_overlay

        fig = go.Figure()
        add_boundary_overlay(fig, None)
        self.assertEqual(len(_figure_traces(fig)), 0)

    def test_add_boundary_overlay_adds_contour(self) -> None:
        from dashboard.components.visualizations import add_boundary_overlay

        fig = go.Figure()
        interface = np.linspace(0.0, 1.0, 16).reshape(4, 4)
        add_boundary_overlay(fig, interface)
        self.assertEqual(_figure_traces(fig)[0].type, "contour")

    def test_plot_calibration_by_uncertainty_builds_bar(self) -> None:
        from dashboard.components.visualizations import plot_calibration_by_uncertainty

        rng = np.random.default_rng(1)
        y_true = rng.random(32)
        y_pred = y_true + rng.normal(0.0, 0.05, 32)
        uncertainty = rng.random(32)
        fig = plot_calibration_by_uncertainty(y_true, y_pred, uncertainty, n_bins=4)
        trace_types = {trace.type for trace in _figure_traces(fig)}
        self.assertIn("bar", trace_types)

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
        types = {trace.type for trace in _figure_traces(fig)}
        self.assertIn("scatter", types)
        self.assertGreaterEqual(len(_figure_traces(fig)), 2)

    def test_create_diagnostic_dashboard_full_layout(self) -> None:
        import json

        from dashboard.components.visualizations import create_diagnostic_dashboard
        from dashboard.components.world_renderer import run_world_for_spec_dict
        from dashboard.utils.data_processing import flatten_archive_record

        record = json.loads(_SMOKE_ARCHIVE.read_text(encoding="utf-8").splitlines()[0])
        row = flatten_archive_record(record)
        result = run_world_for_spec_dict(row["world_spec"])
        fig = create_diagnostic_dashboard(result, title="Smoke elite")
        trace_types = {trace.type for trace in _figure_traces(fig)}
        self.assertIn("image", trace_types)
        self.assertIn("heatmap", trace_types)
        self.assertIn("bar", trace_types)
        self.assertIn("scatterpolar", trace_types)
        layout = _figure_layout(fig)
        self.assertIsNotNone(layout.title)
        title_text = layout.title.text
        self.assertIn("Smoke elite", title_text)

    def test_create_correlation_heatmap_smoke_matrix(self) -> None:
        from dashboard.components.archive_loader import load_archive_bundle
        from dashboard.components.metrics import correlation_matrix
        from dashboard.components.visualizations import create_correlation_heatmap
        from dashboard.utils.config import load_config

        if not _SMOKE_ARCHIVE.is_file():
            self.skipTest("smoke archive missing")
        cfg = load_config()
        bundle = load_archive_bundle(_SMOKE_ARCHIVE, 0.0, cfg)
        fig = create_correlation_heatmap(correlation_matrix(bundle.collapsed))
        self.assertEqual(_figure_traces(fig)[0].type, "heatmap")

    def test_create_metric_histogram_fallback_when_color_by_all_nan(self) -> None:
        import pandas as pd

        from dashboard.components.visualizations import create_metric_histogram

        frame = pd.DataFrame(
            {
                "fitness": [0.2, 0.4, 0.6],
                "emitter_type": [float("nan"), float("nan"), float("nan")],
            }
        )
        fig = create_metric_histogram(frame, "fitness", color_by="emitter_type")
        self.assertEqual(len(_figure_traces(fig)), 1)
        self.assertEqual(_figure_traces(fig)[0].type, "histogram")

    def test_create_metric_histogram_with_emitter_groups(self) -> None:
        import pandas as pd

        from dashboard.components.visualizations import create_metric_histogram

        frame = pd.DataFrame(
            {
                "fitness": [0.2, 0.4, 0.6, 0.8],
                "stability": [0.1, 0.3, 0.5, 0.7],
                "diversity": [0.15, 0.35, 0.55, 0.75],
                "emitter_type": ["random", "random", "genetic", "genetic"],
            }
        )
        fig = create_metric_histogram(frame, "fitness", color_by="emitter_type")
        trace_types = {trace.type for trace in _figure_traces(fig)}
        self.assertIn("histogram", trace_types)
        self.assertGreaterEqual(len(_figure_traces(fig)), 2)

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

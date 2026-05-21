"""E10: dashboard public API and visualizer deprecation boundaries."""

from __future__ import annotations

import unittest
import warnings

_REQUIRED_VISUALIZATION_EXPORTS = (
    "create_archive_heatmap",
    "create_diagnostic_dashboard",
    "plot_real_vs_predicted",
    "plot_calibration_by_uncertainty",
    "create_correlation_heatmap",
    "create_metric_histogram",
)


class TestDashboardE10Deprecation(unittest.TestCase):
    def test_visualizations_exports_cover_tz_section_4(self) -> None:
        from dashboard.components import visualizations

        for name in _REQUIRED_VISUALIZATION_EXPORTS:
            self.assertIn(name, visualizations.__all__)
            self.assertTrue(callable(getattr(visualizations, name)))

    def test_worldspace_init_no_longer_reexports_visualizer(self) -> None:
        import worldspace

        for name in (
            "plot_world_metrics_pca_scatter_from_jsonl",
            "plot_simulation_final_grid",
            "load_ca_step_trace_jsonl",
        ):
            self.assertNotIn(name, worldspace.__all__)
            self.assertFalse(hasattr(worldspace, name))

    def test_visualizer_import_emits_deprecation_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            import importlib

            importlib.reload(importlib.import_module("worldspace.visualizer"))
        messages = [
            str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        self.assertTrue(any("deprecated" in m.lower() for m in messages))


if __name__ == "__main__":
    unittest.main()

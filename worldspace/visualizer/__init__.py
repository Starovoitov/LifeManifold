"""Deprecated matplotlib PNG helpers (pipeline traces). Prefer the Streamlit dashboard."""

from __future__ import annotations

import warnings

from .plotting import (
    load_ca_step_trace_jsonl,
    plot_ca_step_metrics_timeseries,
    plot_ca_step_pca_trajectories,
    plot_ca_step_umap_trajectories,
    plot_dominant_metric_delta_scatter_from_jsonl,
    plot_simulation_final_grid,
    plot_world_metrics_pca_scatter_from_jsonl,
    plot_world_metrics_umap_scatter_from_jsonl,
    summarize_ca_step_trace_by_world,
)

__all__ = [
    "plot_world_metrics_pca_scatter_from_jsonl",
    "plot_world_metrics_umap_scatter_from_jsonl",
    "plot_dominant_metric_delta_scatter_from_jsonl",
    "plot_simulation_final_grid",
    "load_ca_step_trace_jsonl",
    "summarize_ca_step_trace_by_world",
    "plot_ca_step_metrics_timeseries",
    "plot_ca_step_pca_trajectories",
    "plot_ca_step_umap_trajectories",
]


def _warn_deprecated() -> None:
    warnings.warn(
        "worldspace.visualizer is deprecated. Use the Streamlit dashboard "
        "(cd dashboard && streamlit run Home.py) for MAP-Elites archives. "
        "This package remains for pipeline JSONL PNG export only.",
        DeprecationWarning,
        stacklevel=2,
    )


_warn_deprecated()

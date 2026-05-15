"""Visualization helpers and CLI entry (``python -m worldspace.visualizer``)."""

from .plotting import (
    load_ca_step_trace_jsonl,
    plot_ca_step_metrics_timeseries,
    plot_ca_step_pca_trajectories,
    plot_ca_step_umap_trajectories,
    plot_simulation_final_grid,
    plot_world_embedding,
    plot_world_embedding_from_jsonl,
    summarize_ca_step_trace_by_world,
)

__all__ = [
    "plot_world_embedding",
    "plot_world_embedding_from_jsonl",
    "plot_simulation_final_grid",
    "load_ca_step_trace_jsonl",
    "summarize_ca_step_trace_by_world",
    "plot_ca_step_metrics_timeseries",
    "plot_ca_step_pca_trajectories",
    "plot_ca_step_umap_trajectories",
]

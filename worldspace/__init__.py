"""MVP world-space toolkit: generators -> simulation -> metrics -> streaming JSONL."""

from .metrics import METRICS_VECTOR_DIM, WorldMetrics, metrics_vector_to_dict
from .pipeline import (
    dominant_metric_delta_axis_labels,
    dominant_metric_delta_xy_batch,
    stream_world_space_to_jsonl,
)
from .simulator import SimulationResult, run_world
from .specs.spec import WorldSpec
from .visualizer import (
    plot_ca_step_metrics_timeseries,
    plot_ca_step_pca_trajectories,
    plot_ca_step_umap_trajectories,
    plot_simulation_final_grid,
    plot_world_metrics_pca_scatter_from_jsonl,
    plot_world_metrics_umap_scatter_from_jsonl,
    plot_dominant_metric_delta_scatter_from_jsonl,
    load_ca_step_trace_jsonl,
    summarize_ca_step_trace_by_world,
)

__all__ = [
    "WorldSpec",
    "SimulationResult",
    "METRICS_VECTOR_DIM",
    "WorldMetrics",
    "run_world",
    "metrics_vector_to_dict",
    "stream_world_space_to_jsonl",
    "dominant_metric_delta_xy_batch",
    "dominant_metric_delta_axis_labels",
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

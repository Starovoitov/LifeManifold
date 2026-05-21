"""MVP world-space toolkit: generators -> simulation -> metrics -> streaming JSONL."""

from .metrics import METRICS_VECTOR_DIM, WorldMetrics, metrics_vector_to_dict
from .pipeline import (
    dominant_metric_delta_axis_labels,
    dominant_metric_delta_xy_batch,
    stream_world_space_to_jsonl,
)
from .simulator import SimulationResult, run_world
from .specs.spec import WorldSpec

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
]

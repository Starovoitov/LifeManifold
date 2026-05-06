"""MVP world-space toolkit: generators -> simulation -> metrics -> streaming JSONL."""

from .metrics import METRICS_VECTOR_DIM, WorldMetrics, metrics_vector_to_dict
from .pipeline import stream_world_space_to_jsonl
from .simulator import SimulationResult, run_world
from .specs.spec import WorldSpec
from .viz import (
    plot_simulation_final_grid,
    plot_world_embedding,
    plot_world_embedding_from_jsonl,
)

__all__ = [
    "WorldSpec",
    "SimulationResult",
    "METRICS_VECTOR_DIM",
    "WorldMetrics",
    "run_world",
    "metrics_vector_to_dict",
    "stream_world_space_to_jsonl",
    "plot_world_embedding",
    "plot_world_embedding_from_jsonl",
    "plot_simulation_final_grid",
]

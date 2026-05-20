"""Utility wrappers for surrogate fitness composition."""

from __future__ import annotations

from worldspace.illuminators.evaluation import compute_fitness
from worldspace.metrics import WorldMetrics
from worldspace.surrogate.types import SurrogatePrediction

__all__ = ["compute_fitness_from_prediction"]


def compute_fitness_from_prediction(pred: SurrogatePrediction) -> float:
    """Compute fitness via illuminator source function, not duplicated formula."""
    components = pred.components
    metrics = WorldMetrics(
        entropy=0.0,
        stability=float(components["stability"]),
        average_lifespan=0.0,
        density_mean=float(components["final_density"]),
        oscillation_score=float(components["oscillation_score"]),
        diversity=float(components["diversity"]),
        mo_eoc_indicator=0.0,
        topology_interface_index=float(components["topology_interface_index"]),
        topology_window_heterogeneity=float(components["topology_window_heterogeneity"]),
        compressibility_score=0.0,
        ecology_state_entropy_norm=0.0,
        ecology_resource_adjacency=0.0,
    )
    measures = {
        "stability": float(components["stability"]),
        "diversity": float(components["diversity"]),
    }
    return compute_fitness(
        metrics,
        measures,
        early_extinct=_is_early_extinct(components),
        final_density=float(components["final_density"]),
    )


def _is_early_extinct(components: dict[str, float]) -> bool:
    return float(components["early_extinction_prob"]) >= 0.5

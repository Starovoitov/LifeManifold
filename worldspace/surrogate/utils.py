"""Utility wrappers for surrogate fitness composition."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from worldspace.illuminators.evaluation import compute_fitness
from worldspace.metrics import WorldMetrics
from worldspace.surrogate.types import SurrogatePrediction

if TYPE_CHECKING:
    from worldspace.surrogate.model import SurrogateModel

__all__ = [
    "compute_fitness_from_prediction",
    "compute_soft_fitness_from_prediction",
    "resolve_surrogate_fitness",
]


def compute_fitness_from_prediction(
    pred: SurrogatePrediction,
    *,
    use_soft_extinction: bool = False,
) -> float:
    """Compute fitness via illuminator source function, not duplicated formula."""
    if use_soft_extinction:
        return compute_soft_fitness_from_prediction(pred)
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
        topology_window_heterogeneity=float(
            components["topology_window_heterogeneity"]
        ),
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


def compute_soft_fitness_from_prediction(pred: SurrogatePrediction) -> float:
    """Surrogate-only soft extinction: scale base fitness by ``(1 - p_ext)``."""
    components = pred.components
    p_ext = float(np.clip(components["early_extinction_prob"], 0.0, 1.0))
    metrics = WorldMetrics(
        entropy=0.0,
        stability=float(components["stability"]),
        average_lifespan=0.0,
        density_mean=float(components["final_density"]),
        oscillation_score=float(components["oscillation_score"]),
        diversity=float(components["diversity"]),
        mo_eoc_indicator=0.0,
        topology_interface_index=float(components["topology_interface_index"]),
        topology_window_heterogeneity=float(
            components["topology_window_heterogeneity"]
        ),
        compressibility_score=0.0,
        ecology_state_entropy_norm=0.0,
        ecology_resource_adjacency=0.0,
    )
    measures = {
        "stability": float(components["stability"]),
        "diversity": float(components["diversity"]),
    }
    base = compute_fitness(
        metrics,
        measures,
        early_extinct=False,
        final_density=float(components["final_density"]),
    )
    return float(np.clip((1.0 - p_ext) * base, 0.0, 1.0))


def _is_early_extinct(components: dict[str, float]) -> bool:
    return float(components["early_extinction_prob"]) >= 0.5


def resolve_surrogate_fitness(
    model: SurrogateModel,
    features,
    prediction: SurrogatePrediction,
    *,
    use_soft_extinction: bool = False,
) -> float:
    """Use direct fitness head when trained, otherwise compose from components."""
    direct = model.predict_fitness(features)
    if direct is not None:
        return float(direct)
    return compute_fitness_from_prediction(
        prediction,
        use_soft_extinction=use_soft_extinction,
    )

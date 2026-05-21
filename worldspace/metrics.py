from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Length of ``WorldMetrics.as_vector()`` — keep mmap / PCA / k-means in sync.
METRICS_VECTOR_DIM = 12
METRIC_KEYS = (
    "entropy",
    "stability",
    "average_lifespan",
    "density_mean",
    "oscillation_score",
    "diversity",
    "mo_eoc_indicator",
    "topology_interface_index",
    "topology_window_heterogeneity",
    "compressibility_score",
    "ecology_state_entropy_norm",
    "ecology_resource_adjacency",
)


@dataclass
class WorldMetrics:
    """Behavioral metric vector used as world-space coordinates (``METRICS_VECTOR_DIM`` scalars)."""

    entropy: float
    stability: float
    average_lifespan: float
    density_mean: float
    oscillation_score: float
    diversity: float
    mo_eoc_indicator: float
    topology_interface_index: float
    topology_window_heterogeneity: float
    compressibility_score: float
    ecology_state_entropy_norm: float
    ecology_resource_adjacency: float

    def as_vector(self) -> np.ndarray:
        """Return metrics as a fixed-order numeric vector."""
        return np.array(
            [
                self.entropy,
                self.stability,
                self.average_lifespan,
                self.density_mean,
                self.oscillation_score,
                self.diversity,
                self.mo_eoc_indicator,
                self.topology_interface_index,
                self.topology_window_heterogeneity,
                self.compressibility_score,
                self.ecology_state_entropy_norm,
                self.ecology_resource_adjacency,
            ],
            dtype=float,
        )


def multi_objective_edge_of_chaos_indicator(
    entropy: float,
    stability: float,
    diversity: float,
    oscillation_score: float,
    average_lifespan: float,
    extinction_penalty: float,
) -> float:
    """
    **Multi-Objective + Edge-of-Chaos (MO+EoC) indicator** — scalar fitness for GA / LLM / hybrid.

    Uses **curvature** ``C_H = H(1-H)`` on binary entropy ``H`` (not raw ``H`` alone in the EoC
    gate) and **activity × persistence** ``C_AP = A · P`` with ``A = oscillation_score``,
    ``P = clip(average_lifespan/10, 0, 1)``. See ``docs/FORMULAS.md`` §5 for coefficients.
    """
    h = float(entropy)
    c_h = h * (1.0 - h)
    c_h_norm = c_h / 0.25
    p = float(np.clip(average_lifespan / 10.0, 0.0, 1.0))
    a = float(oscillation_score)
    c_ap = a * p
    mo = h + float(stability) + float(diversity)
    return float(
        mo * (0.50 + 0.30 * c_h_norm + 0.20 * c_ap)
        + 0.15 * a * c_h_norm
        + 0.10 * p
        - float(extinction_penalty)
    )


def metrics_vector_to_dict(v: np.ndarray) -> dict[str, float]:
    """Map a metrics vector back to named fields for JSON export."""
    return {k: float(v[i]) for i, k in enumerate(METRIC_KEYS)}

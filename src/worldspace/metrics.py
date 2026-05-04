from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WorldMetrics:
    """Behavioral metric vector used as world-space coordinates."""

    entropy: float
    stability: float
    average_lifespan: float
    density_mean: float
    oscillation_score: float
    diversity: float

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
            ],
            dtype=float,
        )


def metrics_vector_to_dict(v: np.ndarray) -> dict[str, float]:
    """Map a 6-vector back to named metric fields for JSON export."""
    keys = (
        "entropy",
        "stability",
        "average_lifespan",
        "density_mean",
        "oscillation_score",
        "diversity",
    )
    return {k: float(v[i]) for i, k in enumerate(keys)}

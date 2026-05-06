from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Length of ``WorldMetrics.as_vector()`` — keep mmap / PCA / k-means in sync.
METRICS_VECTOR_DIM = 7

# Index of ``average_lifespan`` in ``WorldMetrics.as_vector()`` (embedding axis 1).
METRIC_INDEX_AVERAGE_LIFESPAN = 2


@dataclass
class WorldMetrics:
    """Behavioral metric vector used as world-space coordinates."""

    entropy: float
    stability: float
    average_lifespan: float
    density_mean: float
    oscillation_score: float
    diversity: float
    interestingness: float

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
                self.interestingness,
            ],
            dtype=float,
        )


def metrics_vector_to_dict(v: np.ndarray) -> dict[str, float]:
    """Map a metrics vector back to named fields for JSON export."""
    keys = (
        "entropy",
        "stability",
        "average_lifespan",
        "density_mean",
        "oscillation_score",
        "diversity",
        "interestingness",
    )
    return {k: float(v[i]) for i, k in enumerate(keys)}

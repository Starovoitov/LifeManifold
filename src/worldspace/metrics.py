from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import math as ws_math
from .simulator import SimulationResult


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


def compute_metrics(result: SimulationResult) -> WorldMetrics:
    """Compute the six core world metrics from simulation results."""
    density = np.array(result.density_series, dtype=float)
    density_mean = float(density.mean()) if density.size else 0.0
    entropy = ws_math.binary_entropy(density_mean)
    stability = float(np.clip(1.0 - (density.std() / (density_mean + 1e-6)), 0.0, 1.0))
    avg_lifespan = float(np.mean(result.death_ages)) if result.death_ages else 0.0
    oscillation = ws_math.oscillation(density)
    diversity = ws_math.pattern_diversity(result.history)
    return WorldMetrics(
        entropy=entropy,
        stability=stability,
        average_lifespan=avg_lifespan,
        density_mean=density_mean,
        oscillation_score=oscillation,
        diversity=diversity,
    )

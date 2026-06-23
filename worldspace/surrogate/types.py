"""Public surrogate contracts for MVP integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from worldspace.specs.spec import WorldSpec

ModelType = Literal["lightgbm", "mlp"]

__all__ = [
    "ModelType",
    "SurrogateConfig",
    "SurrogatePrediction",
    "SurrogateProtocol",
]


@dataclass(frozen=True)
class SurrogateConfig:
    """Runtime surrogate settings consumed by ``get_surrogate``."""

    enabled: bool
    model_type: ModelType
    checkpoint: str | None
    stub_mean: float
    stub_uncertainty: float
    calibration: str | None = None
    require_quality_gate: bool = False
    use_soft_extinction: bool = False


@dataclass(frozen=True)
class SurrogatePrediction:
    """Prediction payload returned by all surrogate implementations."""

    components: dict[str, float]
    measures: dict[str, float]
    fitness: float
    uncertainty: float


class SurrogateProtocol(Protocol):
    """Minimal stable API for scheduler / emitter integration."""

    def predict(self, world_spec: WorldSpec) -> SurrogatePrediction:
        """Return deterministic surrogate estimation for one world spec."""
        ...

    def predict_batch(
        self, world_specs: Sequence[WorldSpec]
    ) -> list[SurrogatePrediction]:
        """Return predictions in the same order as ``world_specs``."""
        ...

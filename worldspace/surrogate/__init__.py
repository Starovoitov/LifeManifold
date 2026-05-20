"""Surrogate factory and MVP-compatible implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from worldspace.illuminators.evaluation import apply_canonical_seed
from worldspace.surrogate.feature_extractor import extract as extract_features
from worldspace.surrogate.types import (
    SurrogateConfig,
    SurrogatePrediction,
    SurrogateProtocol,
)

__all__ = [
    "StubSurrogate",
    "SurrogateFacade",
    "get_surrogate",
]


@dataclass(frozen=True)
class StubSurrogate:
    """Fallback surrogate used when feature/model stack is unavailable."""

    mean: float
    uncertainty: float

    def predict(self, world_spec: Any) -> SurrogatePrediction:
        """Return deterministic placeholder prediction for prompt enrichment."""
        _ = world_spec
        components = {"stability": self.mean, "diversity": self.mean}
        measures = {"stability": self.mean, "diversity": self.mean}
        return SurrogatePrediction(
            components=components,
            measures=measures,
            fitness=self.mean,
            uncertainty=self.uncertainty,
        )


@dataclass(frozen=True)
class SurrogateFacade:
    """Stable wrapper for a checkpoint-backed surrogate predictor."""

    predictor: Callable[[Any], SurrogatePrediction]

    def predict(self, world_spec: Any) -> SurrogatePrediction:
        """Canonicalize spec and delegate to predictor implementation."""
        apply_canonical_seed(world_spec)
        _ = extract_features(world_spec)
        return self.predictor(world_spec)


def get_surrogate(config: SurrogateConfig) -> SurrogateProtocol:
    """Create real surrogate when enabled and checkpoint exists, else stub."""
    if not config.enabled:
        return StubSurrogate(mean=config.stub_mean, uncertainty=config.stub_uncertainty)
    checkpoint = _checkpoint_path(config.checkpoint)
    if checkpoint is None or not checkpoint.is_file():
        return StubSurrogate(mean=config.stub_mean, uncertainty=config.stub_uncertainty)
    return SurrogateFacade(
        predictor=_build_placeholder_predictor(
            mean=config.stub_mean,
            uncertainty=config.stub_uncertainty,
        )
    )


def _checkpoint_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser()


def _build_placeholder_predictor(
    *,
    mean: float,
    uncertainty: float,
) -> Callable[[Any], SurrogatePrediction]:
    """Temporary deterministic predictor until model/trainer wiring lands."""

    def _predict(world_spec: Any) -> SurrogatePrediction:
        _ = world_spec
        components = {"stability": mean, "diversity": mean}
        measures = {"stability": mean, "diversity": mean}
        return SurrogatePrediction(
            components=components,
            measures=measures,
            fitness=mean,
            uncertainty=uncertainty,
        )

    return _predict
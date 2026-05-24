"""Surrogate facade with deterministic canonical-key LRU cache."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from worldspace.specs.spec import WorldSpec
from worldspace.surrogate.calibration import (
    UncertaintyCalibrator,
    apply_calibrated_uncertainty,
    load_uncertainty_calibration,
)
from worldspace.surrogate.canonical_hash import world_spec_canonical_hash
from worldspace.surrogate.checkpoint_io import load_surrogate_checkpoint
from worldspace.surrogate.feature_extractor import extract as extract_features
from worldspace.surrogate.model import SurrogateModel
from worldspace.surrogate.types import SurrogatePrediction
from worldspace.surrogate.utils import compute_fitness_from_prediction

__all__ = [
    "StubSurrogate",
    "SurrogateFacade",
    "build_surrogate_facade",
]


@dataclass(frozen=True)
class StubSurrogate:
    """Fallback surrogate used when feature/model stack is unavailable."""

    mean: float
    uncertainty: float

    def predict(self, world_spec: WorldSpec) -> SurrogatePrediction:
        """Return deterministic placeholder prediction for prompt enrichment."""
        _ = world_spec
        components = _stub_components(self.mean)
        measures = {"stability": self.mean, "diversity": self.mean}
        return SurrogatePrediction(
            components=components,
            measures=measures,
            fitness=self.mean,
            uncertainty=self.uncertainty,
        )


@dataclass
class SurrogateFacade:
    """Checkpoint-backed surrogate predictor with LRU cache."""

    model: SurrogateModel
    uncertainty_fallback: float
    calibrator: UncertaintyCalibrator | None = None
    calibration_configured: bool = False
    cache_capacity: int = 1024
    _cache: OrderedDict[str, SurrogatePrediction] = field(default_factory=OrderedDict)
    _cache_hits: int = 0

    def predict(self, world_spec: WorldSpec) -> SurrogatePrediction:
        """Canonicalize spec, then return cached or newly computed prediction."""
        from worldspace.illuminators.evaluation import apply_canonical_seed

        if not isinstance(world_spec, WorldSpec):
            msg = "surrogate.predict expects WorldSpec"
            raise TypeError(msg)
        apply_canonical_seed(world_spec)
        key = self._cache_key(world_spec)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self._cache_hits += 1
            return cached

        features = extract_features(world_spec)
        components = self.model.predict_components(features)
        raw_uncertainty = self.model.predict_uncertainty(features)
        if raw_uncertainty <= 0.0:
            raw_uncertainty = self.uncertainty_fallback
        uncertainty = apply_calibrated_uncertainty(
            self.calibrator,
            float(raw_uncertainty),
            calibration_configured=self.calibration_configured,
        )
        prediction = SurrogatePrediction(
            components=components,
            measures={
                "stability": float(components["stability"]),
                "diversity": float(components["diversity"]),
            },
            fitness=0.0,
            uncertainty=uncertainty,
        )
        resolved = SurrogatePrediction(
            components=prediction.components,
            measures=prediction.measures,
            fitness=compute_fitness_from_prediction(prediction),
            uncertainty=prediction.uncertainty,
        )
        self._cache_set(key, resolved)
        return resolved

    def cache_hits(self) -> int:
        """Return number of successful cache hits."""
        return self._cache_hits

    def reload(self, checkpoint_path: str | Path) -> None:
        """Load a new checkpoint and clear the prediction LRU cache."""
        path = Path(checkpoint_path).expanduser()
        self.model = load_surrogate_checkpoint(path)
        self._cache.clear()
        self._cache_hits = 0

    def _cache_set(self, key: str, value: SurrogatePrediction) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.cache_capacity:
            self._cache.popitem(last=False)

    def _cache_key(self, world_spec: WorldSpec) -> str:
        return world_spec_canonical_hash(world_spec)


def build_surrogate_facade(
    model: SurrogateModel,
    *,
    uncertainty_fallback: float,
    calibration_path: str | Path | None = None,
    cache_capacity: int = 1024,
) -> SurrogateFacade:
    """Construct a facade with explicit constructor kwargs for type checkers."""
    configured = bool(calibration_path and str(calibration_path).strip())
    calibrator = None
    if calibration_path is not None and configured:
        calibrator = load_uncertainty_calibration(calibration_path)
    return SurrogateFacade(
        model=model,
        uncertainty_fallback=uncertainty_fallback,
        calibrator=calibrator,
        calibration_configured=configured,
        cache_capacity=cache_capacity,
    )


def _stub_components(value: float) -> dict[str, float]:
    return {
        "stability": value,
        "diversity": value,
        "oscillation_score": value,
        "topology_interface_index": value,
        "topology_window_heterogeneity": value,
        "final_density": value,
        "early_extinction_prob": value,
    }

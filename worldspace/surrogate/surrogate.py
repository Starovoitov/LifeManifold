"""Surrogate facade with deterministic canonical-key LRU cache."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field

from worldspace.illuminators.evaluation import apply_canonical_seed
from worldspace.specs.spec import WorldSpec
from worldspace.surrogate.feature_extractor import extract as extract_features
from worldspace.surrogate.model import SurrogateModel
from worldspace.surrogate.types import SurrogatePrediction
from worldspace.surrogate.utils import compute_fitness_from_prediction

__all__ = ["StubSurrogate", "SurrogateFacade"]

_CANONICAL_JSON_KWARGS = {"sort_keys": True, "separators": (",", ":")}


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
    cache_capacity: int = 1024
    _cache: OrderedDict[str, SurrogatePrediction] = field(default_factory=OrderedDict)
    _cache_hits: int = 0

    def predict(self, world_spec: WorldSpec) -> SurrogatePrediction:
        """Canonicalize spec, then return cached or newly computed prediction."""
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
        uncertainty = self.model.predict_uncertainty(features)
        if uncertainty <= 0.0:
            uncertainty = self.uncertainty_fallback
        prediction = SurrogatePrediction(
            components=components,
            measures={
                "stability": float(components["stability"]),
                "diversity": float(components["diversity"]),
            },
            fitness=0.0,
            uncertainty=float(uncertainty),
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

    def _cache_set(self, key: str, value: SurrogatePrediction) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.cache_capacity:
            self._cache.popitem(last=False)

    def _cache_key(self, world_spec: WorldSpec) -> str:
        # Use canonical payload so cache identity is independent of runtime ``seed``.
        payload = world_spec.to_canonical_dict()
        canonical = json.dumps(payload, **_CANONICAL_JSON_KWARGS)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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

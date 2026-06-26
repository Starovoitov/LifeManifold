"""Surrogate facade with deterministic canonical-key LRU cache."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from worldspace.specs.spec import WorldSpec
from worldspace.surrogate.calibration import (
    UncertaintyCalibrator,
    apply_calibrated_uncertainty,
    load_uncertainty_calibration,
)
from worldspace.surrogate.canonical_hash import world_spec_canonical_hash
from worldspace.surrogate.checkpoint_io import load_surrogate_checkpoint
from worldspace.surrogate.feature_extractor import (
    extract_batch as extract_features_batch,
)
from worldspace.surrogate.model import (
    EXPECTED_FEATURE_DIM,
    SurrogateModel,
    checkpoint_feature_dim,
    checkpoint_matches_extractor,
)
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
        return self._placeholder_prediction()

    def predict_batch(
        self, world_specs: Sequence[WorldSpec]
    ) -> list[SurrogatePrediction]:
        """Return one placeholder prediction per spec."""
        placeholder = self._placeholder_prediction()
        return [placeholder for _ in world_specs]

    def _placeholder_prediction(self) -> SurrogatePrediction:
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
    use_soft_extinction: bool = False
    extinction_gate_threshold: float = 0.5
    cache_capacity: int = 1024
    _cache: OrderedDict[str, SurrogatePrediction] = field(default_factory=OrderedDict)
    _cache_hits: int = 0

    def predict(self, world_spec: WorldSpec) -> SurrogatePrediction:
        """Canonicalize spec, then return cached or newly computed prediction."""
        predictions = self.predict_batch([world_spec])
        return predictions[0]

    def predict_batch(
        self, world_specs: Sequence[WorldSpec]
    ) -> list[SurrogatePrediction]:
        """Canonicalize specs, batch model inference for cache misses, preserve order."""
        from worldspace.illuminators.evaluation import apply_canonical_seed

        if not world_specs:
            return []

        keys: list[str] = []
        cached_by_index: dict[int, SurrogatePrediction] = {}
        miss_indices: list[int] = []
        miss_specs: list[WorldSpec] = []

        for index, world_spec in enumerate(world_specs):
            if not isinstance(world_spec, WorldSpec):
                msg = "surrogate.predict_batch expects WorldSpec items"
                raise TypeError(msg)
            apply_canonical_seed(world_spec)
            key = self._cache_key(world_spec)
            keys.append(key)
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self._cache_hits += 1
                cached_by_index[index] = cached
            else:
                miss_indices.append(index)
                miss_specs.append(world_spec)

        if miss_specs:
            feature_matrix = extract_features_batch(miss_specs)
            component_rows = self.model.predict_components_batch(feature_matrix)
            uncertainty_rows = self.model.predict_uncertainty_batch(feature_matrix)
            for miss_offset, row_index in enumerate(miss_indices):
                features = feature_matrix[miss_offset]
                components = component_rows[miss_offset]
                raw_uncertainty = float(uncertainty_rows[miss_offset])
                if raw_uncertainty <= 0.0:
                    raw_uncertainty = self.uncertainty_fallback
                uncertainty = apply_calibrated_uncertainty(
                    self.calibrator,
                    raw_uncertainty,
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
                    fitness=_resolve_surrogate_fitness(
                        self.model,
                        features,
                        prediction,
                        use_soft_extinction=self.use_soft_extinction,
                        extinction_gate_threshold=self.extinction_gate_threshold,
                    ),
                    uncertainty=prediction.uncertainty,
                )
                cached_by_index[row_index] = resolved
                self._cache_set(keys[row_index], resolved)

        return [cached_by_index[index] for index in range(len(world_specs))]

    def cache_hits(self) -> int:
        """Return number of successful cache hits."""
        return self._cache_hits

    def reload(self, checkpoint_path: str | Path) -> None:
        """Load a new checkpoint and clear the prediction LRU cache."""
        path = Path(checkpoint_path).expanduser()
        model = load_surrogate_checkpoint(path)
        if not checkpoint_matches_extractor(model):
            trained_dim = checkpoint_feature_dim(model)
            msg = (
                f"Checkpoint feature_dim={trained_dim}, expected "
                f"{EXPECTED_FEATURE_DIM}: {path}"
            )
            raise ValueError(msg)
        self.model = model
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
    use_soft_extinction: bool = False,
    extinction_gate_threshold: float = 0.5,
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
        use_soft_extinction=use_soft_extinction,
        extinction_gate_threshold=float(extinction_gate_threshold),
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


def _resolve_surrogate_fitness(
    model: SurrogateModel,
    features,
    prediction: SurrogatePrediction,
    *,
    use_soft_extinction: bool = False,
    extinction_gate_threshold: float = 0.5,
) -> float:
    """Use direct fitness head when available, else compose with configured gate."""
    direct = model.predict_fitness(features)
    if direct is not None:
        return float(np.clip(float(direct), 0.0, 1.0))
    return float(
        np.clip(
            compute_fitness_from_prediction(
                prediction,
                use_soft_extinction=use_soft_extinction,
                extinction_gate_threshold=extinction_gate_threshold,
            ),
            0.0,
            1.0,
        )
    )

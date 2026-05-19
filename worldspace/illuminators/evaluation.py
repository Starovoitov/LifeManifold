"""MAP-Elites candidate evaluation helpers (seed, fitness, measures — TZ v1.2)."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from worldspace.metrics import WorldMetrics
from worldspace.specs.spec import WorldSpec

__all__ = [
    "MEASURE_KEYS",
    "apply_canonical_seed",
    "bin_index",
    "bin_index_from_measures",
    "canonical_seed",
    "compute_fitness",
    "extinction_probability",
    "measures_from_metrics",
    "topology_complexity",
]

MEASURE_KEYS: tuple[str, ...] = ("stability", "diversity")

_CANONICAL_JSON_KWARGS = {"sort_keys": True, "separators": (",", ":")}


def canonical_seed(world_spec: WorldSpec) -> int:
    """Derive a deterministic 32-bit seed from the canonical world spec (§4)."""
    digest = hashlib.sha256(_canonical_payload(world_spec).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**32)


def apply_canonical_seed(world_spec: WorldSpec) -> int:
    """Set ``world_spec.seed`` from the canonical hash and return it."""
    seed = canonical_seed(world_spec)
    world_spec.seed = seed
    return seed


def measures_from_metrics(metrics: WorldMetrics) -> dict[str, float]:
    """MAP-Elites behavioral coordinates (BC) for binning and JSONL ``measures`` (§1)."""
    return {
        "stability": _clip_unit(metrics.stability),
        "diversity": _clip_unit(metrics.diversity),
    }


def topology_complexity(metrics: WorldMetrics) -> float:
    """Fitness-only topology proxy; not a behavioral axis."""
    raw = (
        0.5 * metrics.topology_interface_index
        + 0.5 * metrics.topology_window_heterogeneity
    )
    return _clip_unit(raw)


def extinction_probability(final_density: float) -> float:
    """``clip(1.0 - final_density, 0, 1)`` from final life grid."""
    return _clip_unit(1.0 - final_density)


def compute_fitness(
    metrics: WorldMetrics,
    measures: dict[str, float],
    *,
    early_extinct: bool,
    final_density: float,
) -> float:
    """Illuminator interestingness; ``0.0`` when ``early_extinct``."""
    if early_extinct:
        return 0.0
    ext_p = extinction_probability(final_density)
    return _clip_unit(
        0.45 * measures["diversity"]
        + 0.25 * (1.0 - ext_p)
        + 0.20 * _clip_unit(metrics.oscillation_score)
        + 0.10 * topology_complexity(metrics)
    )


def bin_index(stability: float, diversity: float, resolution: int) -> tuple[int, int]:
    """Map BC values to archive cell indices."""
    edges = _bin_edges(resolution)
    s = _clip_unit(stability)
    d = _clip_unit(diversity)
    i = int(np.minimum(np.searchsorted(edges, s, side="right") - 1, resolution - 1))
    j = int(np.minimum(np.searchsorted(edges, d, side="right") - 1, resolution - 1))
    return (i, j)


def bin_index_from_measures(
    measures: dict[str, float], resolution: int
) -> tuple[int, int]:
    """Bin from JSONL-style ``measures`` dict."""
    return bin_index(measures["stability"], measures["diversity"], resolution)


def _canonical_payload(world_spec: WorldSpec) -> str:
    return json.dumps(world_spec.to_canonical_dict(), **_CANONICAL_JSON_KWARGS)


def _bin_edges(resolution: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, resolution + 1)


def _clip_unit(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))

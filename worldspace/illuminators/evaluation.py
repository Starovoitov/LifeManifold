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
    "canonical_seed",
    "measures_from_metrics",
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


def _canonical_payload(world_spec: WorldSpec) -> str:
    return json.dumps(world_spec.to_canonical_dict(), **_CANONICAL_JSON_KWARGS)


def _clip_unit(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))

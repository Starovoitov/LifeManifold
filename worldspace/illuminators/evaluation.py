"""MAP-Elites candidate evaluation helpers (seed, fitness, measures — TZ v1.2)."""

from __future__ import annotations

import hashlib
import json

from worldspace.specs.spec import WorldSpec

__all__ = ["apply_canonical_seed", "canonical_seed"]

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


def _canonical_payload(world_spec: WorldSpec) -> str:
    return json.dumps(world_spec.to_canonical_dict(), **_CANONICAL_JSON_KWARGS)

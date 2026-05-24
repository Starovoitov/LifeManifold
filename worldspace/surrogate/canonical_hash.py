"""Canonical world-spec hashing shared by surrogate cache and SurrogateArchive."""

from __future__ import annotations

import hashlib
import json

from worldspace.specs.spec import WorldSpec

_CANONICAL_JSON_KWARGS = {"sort_keys": True, "separators": (",", ":")}

__all__ = ["world_spec_canonical_hash"]


def world_spec_canonical_hash(world_spec: WorldSpec) -> str:
    """Return SHA-256 hex digest of the canonical world spec payload."""
    payload = world_spec.to_canonical_dict()
    canonical = json.dumps(payload, **_CANONICAL_JSON_KWARGS)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

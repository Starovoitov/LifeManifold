"""Canonical JSON and SHA-256 helpers for attribution manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

SHA256_HEX_LENGTH = 64


def canonical_json_bytes(
    value: BaseModel | Mapping[str, Any] | Sequence[Any],
    *,
    omit_keys: frozenset[str] = frozenset(),
) -> bytes:
    """Serialize a JSON-compatible value deterministically.

    Callers must pass resolved values: this helper deliberately does not expand
    defaults or dereference paths. Pydantic models are dumped in JSON mode
    before selected envelope keys are removed recursively.
    """
    payload: Any
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json")
    else:
        payload = value
    normalized = _without_keys(payload, omit_keys)
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(
    value: BaseModel | Mapping[str, Any] | Sequence[Any],
    *,
    omit_keys: frozenset[str] = frozenset(),
) -> str:
    """Return the full SHA-256 digest of canonical JSON."""
    return hashlib.sha256(canonical_json_bytes(value, omit_keys=omit_keys)).hexdigest()


def is_sha256(value: str) -> bool:
    """Return whether ``value`` is a lowercase full SHA-256 digest."""
    if len(value) != SHA256_HEX_LENGTH:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _without_keys(value: Any, keys: frozenset[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_keys(item, keys)
            for key, item in value.items()
            if str(key) not in keys
        }
    if isinstance(value, tuple | list):
        return [_without_keys(item, keys) for item in value]
    return value

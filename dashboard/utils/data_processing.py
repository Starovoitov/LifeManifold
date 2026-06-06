"""Flatten MAP-Elites JSONL records and canonical world-spec hashing."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from dashboard.utils.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from worldspace.illuminators.archive import ArchiveElite, elite_to_archive_record
from worldspace.specs.spec import WorldSpec

_CANONICAL_JSON_KWARGS = {"sort_keys": True, "separators": (",", ":")}

__all__ = [
    "canonical_world_spec_hash",
    "elite_to_flat_row",
    "flatten_archive_record",
    "try_flatten_archive_record",
    "world_spec_from_dict",
]


def flatten_archive_record(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten one archive JSONL object for tabular use."""
    bin_raw = record.get("bin")
    if not isinstance(bin_raw, list) or len(bin_raw) != 2:
        msg = "bin must be a list of two integers"
        raise ValueError(msg)

    world_spec = record.get("world_spec")
    if not isinstance(world_spec, dict):
        msg = "world_spec must be an object"
        raise ValueError(msg)

    if "fitness" not in record:
        msg = "missing required field: fitness"
        raise ValueError(msg)

    row: dict[str, Any] = {
        "bin_x": int(bin_raw[0]),
        "bin_y": int(bin_raw[1]),
        "fitness": float(record["fitness"]),
        "world_spec": dict(world_spec),
        "seed": int(world_spec.get("seed", 0)),
    }

    measures = record.get("measures")
    if isinstance(measures, dict):
        for key, value in measures.items():
            row[f"measure_{key}"] = float(value)

    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            row[str(key)] = float(value)

    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        row["elite_id"] = metadata.get("id")
        row["parent_id"] = metadata.get("parent_id")
        row["generated_by"] = metadata.get("generated_by")
        row["emitter_type"] = metadata.get("emitter_type")
        row["timestamp"] = metadata.get("timestamp")
        prompt_version = metadata.get("prompt_version")
        if prompt_version == "":
            prompt_version = None
        row["prompt_version"] = prompt_version

    return row


def try_flatten_archive_record(record: Any) -> dict[str, Any] | None:
    """Flatten one JSONL record, or return None to skip malformed lines."""
    if not isinstance(record, dict):
        return None
    try:
        return flatten_archive_record(record)
    except (KeyError, TypeError, ValueError):
        return None


def elite_to_flat_row(elite: ArchiveElite) -> dict[str, Any]:
    """Serialize an in-memory elite to the same flat shape as ``flatten_archive_record``."""
    return flatten_archive_record(elite_to_archive_record(elite))


def world_spec_from_dict(spec: dict[str, Any]) -> WorldSpec:
    """Parse a JSON-like world spec dict into ``WorldSpec``."""
    return WorldSpec.from_json_dict(spec)


def canonical_world_spec_hash(spec_dict: dict[str, Any]) -> str:
    """Return a stable SHA-256 hex digest for cache keys (canonical spec, no runtime seed)."""
    spec = world_spec_from_dict(spec_dict)
    payload = json.dumps(spec.to_canonical_dict(), **_CANONICAL_JSON_KWARGS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

"""Flatten MAP-Elites JSONL records and canonical world-spec hashing."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from dashboard.utils.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from worldspace.illuminators.archive import (
    ARCHIVE_SCHEMA_VERSION,
    ArchiveElite,
    elite_to_archive_record,
)
from worldspace.specs.spec import WorldSpec

_CANONICAL_JSON_KWARGS = {"sort_keys": True, "separators": (",", ":")}

ArchiveType = Literal["grid", "cvt"]

__all__ = [
    "ArchiveType",
    "canonical_world_spec_hash",
    "elite_to_flat_row",
    "flatten_archive_record",
    "try_flatten_archive_record",
    "world_spec_from_dict",
]


def flatten_archive_record(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten one archive JSONL object for tabular use."""
    schema_version = str(record.get("schema_version", "1.2"))
    archive_type: ArchiveType = (
        "cvt" if str(record.get("archive_type", "grid")) == "cvt" else "grid"
    )

    if schema_version == "1.2":
        bin_x, bin_y = _parse_bin_coord(record["bin"])
        cell_id: int | None = None
    elif archive_type == "cvt":
        cell_raw = record.get("cell_id")
        if cell_raw is None:
            msg = "cell_id is required for schema 1.3 CVT records"
            raise ValueError(msg)
        cell_id = int(cell_raw)
        bin_x, bin_y = cell_id, 0
    else:
        if "bin" in record:
            bin_x, bin_y = _parse_bin_coord(record["bin"])
        elif "cell_id" in record:
            bin_x, bin_y = int(record["cell_id"]), 0
        else:
            msg = "bin or cell_id is required for schema 1.3 grid records"
            raise ValueError(msg)
        cell_raw = record.get("cell_id")
        cell_id = int(cell_raw) if cell_raw is not None else None

    world_spec = record.get("world_spec")
    if not isinstance(world_spec, dict):
        msg = "world_spec must be an object"
        raise ValueError(msg)

    if "fitness" not in record:
        msg = "missing required field: fitness"
        raise ValueError(msg)

    row: dict[str, Any] = {
        "bin_x": bin_x,
        "bin_y": bin_y,
        "fitness": float(record["fitness"]),
        "world_spec": dict(world_spec),
        "seed": int(world_spec.get("seed", 0)),
        "schema_version": schema_version,
        "archive_type": archive_type,
    }
    if cell_id is not None:
        row["cell_id"] = cell_id

    measures = record.get("measures")
    if isinstance(measures, dict):
        for key, value in measures.items():
            row[f"measure_{key}"] = float(value)
            if key == "stability":
                row["stability"] = float(value)
            elif key == "diversity":
                row["diversity"] = float(value)

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


def elite_to_flat_row(
    elite: ArchiveElite,
    *,
    archive_type: ArchiveType = "grid",
    resolution: int | None = None,
) -> dict[str, Any]:
    """Serialize an in-memory elite to the same flat shape as ``flatten_archive_record``."""
    if archive_type == "cvt":
        record = elite_to_archive_record(
            elite,
            archive_type="cvt",
            schema_version="1.3",
        )
    elif resolution is not None:
        record = elite_to_archive_record(
            elite,
            archive_type="grid",
            schema_version="1.3",
            resolution=resolution,
        )
    else:
        record = elite_to_archive_record(
            elite,
            schema_version=ARCHIVE_SCHEMA_VERSION,
        )
    return flatten_archive_record(record)


def world_spec_from_dict(spec: dict[str, Any]) -> WorldSpec:
    """Parse a JSON-like world spec dict into ``WorldSpec``."""
    return WorldSpec.from_json_dict(spec)


def canonical_world_spec_hash(spec_dict: dict[str, Any]) -> str:
    """Return a stable SHA-256 hex digest for cache keys (canonical spec, no runtime seed)."""
    spec = world_spec_from_dict(spec_dict)
    payload = json.dumps(spec.to_canonical_dict(), **_CANONICAL_JSON_KWARGS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_bin_coord(bin_raw: object) -> tuple[int, int]:
    if not isinstance(bin_raw, list) or len(bin_raw) != 2:
        msg = "bin must be a list of two integers"
        raise ValueError(msg)
    return int(bin_raw[0]), int(bin_raw[1])

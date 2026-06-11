"""Re-featurize surrogate buffer rows from stored ``world_spec`` dicts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldspace.illuminators.evaluation import apply_canonical_seed
from worldspace.specs.spec import WorldSpec
from worldspace.surrogate.feature_extractor import (
    FEATURE_SCHEMA_VERSION,
    SUPPORTED_FEATURE_SCHEMA_VERSIONS,
    extract,
    feature_dim_for_schema,
)

__all__ = [
    "re_featurize_buffer",
    "re_featurize_buffer_row",
]


def re_featurize_buffer_row(
    row: dict[str, Any],
    *,
    target_schema: str = FEATURE_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Return one buffer row with features recomputed from ``world_spec``."""
    if target_schema not in SUPPORTED_FEATURE_SCHEMA_VERSIONS:
        msg = f"Unsupported target_schema: {target_schema!r}"
        raise ValueError(msg)
    world_spec_raw = row.get("world_spec")
    if not isinstance(world_spec_raw, dict) or not world_spec_raw:
        msg = "buffer row missing world_spec dict"
        raise ValueError(msg)
    spec = WorldSpec.from_json_dict(dict(world_spec_raw))
    apply_canonical_seed(spec)
    vector = extract(spec, schema_version=target_schema)
    migrated = dict(row)
    migrated["feature_schema_version"] = target_schema
    migrated["features"] = [float(value) for value in vector.tolist()]
    expected_dim = feature_dim_for_schema(target_schema)
    if len(migrated["features"]) != expected_dim:
        msg = (
            f"re-featurize produced dim={len(migrated['features'])}, "
            f"expected {expected_dim} for schema {target_schema!r}"
        )
        raise ValueError(msg)
    return migrated


def re_featurize_buffer(
    input_path: Path | str,
    output_path: Path | str,
    *,
    target_schema: str = FEATURE_SCHEMA_VERSION,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Rewrite a buffer JSONL with features extracted from each row's world_spec."""
    source = Path(input_path).expanduser()
    destination = Path(output_path).expanduser()
    if not source.is_file():
        msg = f"Buffer JSONL not found: {source}"
        raise FileNotFoundError(msg)
    if destination.exists() and not overwrite:
        msg = f"Output already exists (use overwrite=True): {destination}"
        raise FileExistsError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)

    rows_read = 0
    rows_written = 0
    source_schemas: dict[str, int] = {}
    with (
        source.open(encoding="utf-8") as handle,
        destination.open("w", encoding="utf-8") as out,
    ):
        for line_no, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                msg = f"Invalid JSON at {source}:{line_no}: {exc}"
                raise ValueError(msg) from exc
            if not isinstance(row, dict):
                msg = f"Invalid row format at {source}:{line_no}"
                raise ValueError(msg)
            rows_read += 1
            schema = str(row.get("feature_schema_version", "unknown"))
            source_schemas[schema] = source_schemas.get(schema, 0) + 1
            migrated = re_featurize_buffer_row(row, target_schema=target_schema)
            out.write(json.dumps(migrated, ensure_ascii=True, sort_keys=True) + "\n")
            rows_written += 1

    if rows_written == 0:
        msg = f"No buffer rows found in {source}"
        raise ValueError(msg)

    return {
        "rows_read": rows_read,
        "rows_written": rows_written,
        "input_path": str(source.resolve()),
        "output_path": str(destination.resolve()),
        "source_schema_versions": source_schemas,
        "target_schema_version": target_schema,
        "target_feature_dim": feature_dim_for_schema(target_schema),
    }

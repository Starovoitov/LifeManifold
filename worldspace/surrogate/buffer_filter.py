"""Filter and deduplicate surrogate training buffer JSONL rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldspace.specs.spec import WorldSpec
from worldspace.surrogate.canonical_hash import world_spec_canonical_hash

BACKFILL_SOURCES = frozenset({"archive_backfill", "archive_backfill_collapsed"})

__all__ = [
    "BACKFILL_SOURCES",
    "filter_buffer_path",
    "filter_buffer_rows",
]


def filter_buffer_path(
    input_path: Path | str,
    output_path: Path | str | None,
    *,
    dedupe: bool = False,
    live_only: bool = False,
    drop_backfill: bool = False,
) -> dict[str, Any]:
    """Read a buffer JSONL, apply filters, optionally write rows, return stats."""
    source = Path(input_path).expanduser()
    if not source.is_file():
        msg = f"Buffer JSONL not found: {source}"
        raise FileNotFoundError(msg)

    rows_read = 0
    parsed_rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            rows_read += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"Invalid JSON at {source}:{line_no}: {exc}"
                raise ValueError(msg) from exc
            if not isinstance(row, dict):
                msg = f"Invalid row format at {source}:{line_no}: expected JSON object"
                raise ValueError(msg)
            parsed_rows.append(row)

    filtered_rows, stats = filter_buffer_rows(
        parsed_rows,
        dedupe=dedupe,
        live_only=live_only,
        drop_backfill=drop_backfill,
    )
    stats["rows_read"] = rows_read
    stats["rows_written"] = len(filtered_rows)

    if output_path is not None:
        target = Path(output_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for row in filtered_rows:
                fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        stats["output_path"] = str(target.resolve())

    return stats


def filter_buffer_rows(
    rows: list[dict[str, Any]],
    *,
    dedupe: bool,
    live_only: bool,
    drop_backfill: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply source filters and optional canonical world_spec dedupe."""
    stats: dict[str, Any] = {
        "filtered_live_only": 0,
        "filtered_drop_backfill": 0,
        "duplicates_dropped": 0,
        "schema_versions": {},
        "metadata_sources": {},
    }
    kept: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for row in rows:
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        source = str(metadata.get("source") or "unknown")

        if live_only and source != "live_eval":
            stats["filtered_live_only"] += 1
            continue
        if drop_backfill and source in BACKFILL_SOURCES:
            stats["filtered_drop_backfill"] += 1
            continue

        if dedupe:
            world_spec = row.get("world_spec")
            if not isinstance(world_spec, dict) or not world_spec:
                msg = "Cannot dedupe row without world_spec"
                raise ValueError(msg)
            spec_hash = world_spec_canonical_hash(WorldSpec.from_json_dict(world_spec))
            if spec_hash in seen_hashes:
                stats["duplicates_dropped"] += 1
                continue
            seen_hashes.add(spec_hash)

        kept.append(row)
        schema = str(row.get("feature_schema_version") or "unknown")
        stats["schema_versions"][schema] = stats["schema_versions"].get(schema, 0) + 1
        stats["metadata_sources"][source] = stats["metadata_sources"].get(source, 0) + 1

    return kept, stats

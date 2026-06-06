"""Load surrogate training buffer JSONL for dashboard stats and tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.utils.bootstrap import ensure_repo_on_path
from dashboard.utils.config import load_config, resolve_surrogate_buffer_path

ensure_repo_on_path()

from worldspace.surrogate.model import TARGET_KEYS

__all__ = [
    "BufferBundle",
    "apply_buffer_filters",
    "buffer_schema_summary",
    "buffer_summary_counts",
    "export_subset_jsonl",
    "flatten_buffer_record",
    "get_buffer_bundle",
    "load_buffer_bundle",
    "read_buffer_jsonl",
    "schema_mix_warnings",
    "show_large_buffer_warning",
    "slice_for_display",
    "try_flatten_buffer_record",
]

TARGET_COLUMN_PREFIX = "target_"


@dataclass(frozen=True)
class BufferBundle:
    """Parsed training buffer rows plus original records for export."""

    records: pd.DataFrame
    raw_records: list[dict[str, Any]]
    line_count_raw: int
    invalid_line_count: int
    large_buffer_mode: bool
    source_path: str


def get_buffer_bundle(path: Path | None = None) -> BufferBundle:
    """Load the training buffer with Streamlit disk cache (path + mtime)."""
    cfg = load_config()
    target = path if path is not None else resolve_surrogate_buffer_path(cfg)
    if not target.is_file():
        msg = f"surrogate buffer file not found: {target}"
        raise FileNotFoundError(msg)
    mtime = float(target.stat().st_mtime)
    performance_digest = _performance_digest(cfg)
    return _cached_load_buffer_bundle(
        str(target.resolve()),
        mtime,
        performance_digest,
    )


def load_buffer_bundle(
    buffer_path: Path,
    mtime: float,
    cfg: dict[str, Any],
) -> BufferBundle:
    """Read JSONL, flatten rows, and detect large-buffer mode."""
    del mtime
    records, raw_records, line_count_raw, invalid_line_count = read_buffer_jsonl(
        buffer_path, cfg
    )
    performance = _performance_section(cfg)
    threshold = int(performance.get("large_archive_line_threshold", 5000))
    return BufferBundle(
        records=records,
        raw_records=raw_records,
        line_count_raw=line_count_raw,
        invalid_line_count=invalid_line_count,
        large_buffer_mode=line_count_raw > threshold,
        source_path=str(buffer_path.resolve()),
    )


def read_buffer_jsonl(
    path: Path,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], int, int]:
    """Return ``(flat_frame, raw_records, raw_line_count, invalid_line_count)``."""
    performance = _performance_section(cfg)
    prefer_polars = bool(performance.get("prefer_polars", True))
    line_count_raw = _count_jsonl_lines(path)

    if prefer_polars:
        frame, raw_records, invalid = _read_buffer_polars(path)
        if frame is not None:
            return frame, raw_records, line_count_raw, invalid

    return _read_buffer_python(path)


def flatten_buffer_record(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten one buffer JSON object into a table row (raises on invalid shape)."""
    row = try_flatten_buffer_record(record)
    if row is None:
        msg = "invalid surrogate buffer record"
        raise ValueError(msg)
    return row


def try_flatten_buffer_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return a flat row dict or ``None`` when the record is invalid."""
    features = record.get("features")
    targets = record.get("targets")
    emitter_type = record.get("emitter_type")
    if not isinstance(features, list) or not features:
        return None
    if not isinstance(targets, dict):
        return None
    if not isinstance(emitter_type, str) or not emitter_type.strip():
        return None
    row: dict[str, Any] = {
        "feature_schema_version": str(record.get("feature_schema_version", "")),
        "emitter_type": emitter_type,
        "feature_dim": len(features),
        "has_world_spec": _record_has_world_spec(record),
    }
    for key in TARGET_KEYS:
        if key not in targets:
            return None
        try:
            row[f"{TARGET_COLUMN_PREFIX}{key}"] = float(targets[key])
        except (TypeError, ValueError):
            return None
    return row


def buffer_summary_counts(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Value counts for emitter, schema, and feature dimension columns."""
    counts: dict[str, pd.Series] = {}
    if frame.empty:
        return counts
    if "emitter_type" in frame.columns:
        counts["emitter_type"] = frame["emitter_type"].value_counts()
    if "feature_schema_version" in frame.columns:
        counts["feature_schema_version"] = frame[
            "feature_schema_version"
        ].value_counts()
    if "feature_dim" in frame.columns:
        counts["feature_dim"] = frame["feature_dim"].value_counts()
    return counts


def buffer_schema_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize schema version mix, feature dimensions, and world_spec coverage."""
    if frame.empty:
        return {
            "schema_versions": [],
            "feature_dims": [],
            "mixed_schema": False,
            "mixed_feature_dim": False,
            "rows_missing_world_spec": 0,
        }
    schema_versions = sorted(
        frame["feature_schema_version"].dropna().astype(str).unique().tolist()
    )
    feature_dims = sorted(
        int(value) for value in frame["feature_dim"].dropna().unique()
    )
    rows_missing_world_spec = 0
    if "has_world_spec" in frame.columns:
        rows_missing_world_spec = int((~frame["has_world_spec"].fillna(False)).sum())
    return {
        "schema_versions": schema_versions,
        "feature_dims": feature_dims,
        "mixed_schema": len(schema_versions) > 1,
        "mixed_feature_dim": len(feature_dims) > 1,
        "rows_missing_world_spec": rows_missing_world_spec,
    }


def schema_mix_warnings(frame: pd.DataFrame) -> list[str]:
    """Return human-readable warnings for mixed or legacy buffer rows."""
    summary = buffer_schema_summary(frame)
    warnings: list[str] = []
    if summary["mixed_schema"]:
        warnings.append(
            "Mixed feature_schema_version values: "
            + ", ".join(summary["schema_versions"])
            + ". Training requires schema 2.0 only."
        )
    if summary["mixed_feature_dim"]:
        dim_text = ", ".join(str(value) for value in summary["feature_dims"])
        warnings.append(
            f"Mixed feature_dim values: {dim_text}. "
            "Training requires schema 2.0 rows only; migrate or replace the buffer."
        )
    missing = int(summary["rows_missing_world_spec"])
    if missing:
        warnings.append(f"{missing} valid rows are missing world_spec.")
    return warnings


def apply_buffer_filters(
    bundle: BufferBundle,
    *,
    emitter_types: list[str],
    schema_versions: list[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Filter in memory; return aligned frame and raw JSON records.

    An empty multiselect selection yields no rows (not «show all»).
    """
    frame = bundle.records
    if frame.empty:
        return frame.copy(), []

    if not emitter_types or not schema_versions:
        return frame.iloc[0:0].copy(), []

    mask = pd.Series(True, index=frame.index)
    mask &= frame["emitter_type"].isin(emitter_types)
    mask &= frame["feature_schema_version"].isin(schema_versions)

    filtered = frame.loc[mask].reset_index(drop=True)
    indices = frame.index[mask].tolist()
    filtered_raw = [bundle.raw_records[int(i)] for i in indices]
    return filtered, filtered_raw


def slice_for_display(
    frame: pd.DataFrame,
    *,
    page: int,
    page_size: int,
    max_rows: int,
) -> pd.DataFrame:
    """Return one page of rows capped by ``max_rows``."""
    if frame.empty:
        return frame.copy()
    capped_size = min(max(1, page_size), max_rows)
    start = max(0, page) * capped_size
    end = start + capped_size
    return frame.iloc[start:end].copy()


def export_subset_jsonl(raw_records: list[dict[str, Any]]) -> str:
    """Serialize filtered buffer rows to JSONL text."""
    if not raw_records:
        return ""
    lines = [json.dumps(record, sort_keys=True) for record in raw_records]
    return "\n".join(lines) + "\n"


def show_large_buffer_warning(bundle: BufferBundle, cfg: dict[str, Any]) -> None:
    """Display performance guard info when the buffer exceeds the configured threshold."""
    if not bundle.large_buffer_mode:
        return
    performance = _performance_section(cfg)
    threshold = int(performance.get("large_archive_line_threshold", 5000))
    table_max = int(performance.get("table_max_rows", 500))
    prefer_polars = bool(performance.get("prefer_polars", True))
    st.warning(
        f"Large buffer mode: {bundle.line_count_raw:,} JSONL lines (threshold {threshold:,}). "
        f"Active guards: Polars read={prefer_polars}, in-memory filters, table cap={table_max}."
    )


@st.cache_data(show_spinner=False)
def _cached_load_buffer_bundle(
    buffer_path_str: str,
    mtime: float,
    performance_digest: str,
) -> BufferBundle:
    del performance_digest
    cfg = load_config()
    return load_buffer_bundle(Path(buffer_path_str), mtime, cfg)


def _read_buffer_polars(
    path: Path,
) -> tuple[pd.DataFrame | None, list[dict[str, Any]], int]:
    try:
        import polars as pl
    except ImportError:
        return None, [], 0

    try:
        frame = pl.read_ndjson(path)
    except Exception:
        return None, [], 0

    rows: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    invalid = 0
    for record in frame.to_dicts():
        if not isinstance(record, dict):
            invalid += 1
            continue
        row = try_flatten_buffer_record(record)
        if row is None:
            invalid += 1
            continue
        rows.append(row)
        raw_records.append(record)
    if not rows:
        return pd.DataFrame(), [], invalid
    return pd.DataFrame(rows), raw_records, invalid


def _read_buffer_python(
    path: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    line_count_raw = 0
    invalid_line_count = 0

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            line_count_raw += 1
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                invalid_line_count += 1
                continue
            if not isinstance(record, dict):
                invalid_line_count += 1
                continue
            row = try_flatten_buffer_record(record)
            if row is None:
                invalid_line_count += 1
                continue
            rows.append(row)
            raw_records.append(record)

    if rows:
        return pd.DataFrame(rows), raw_records, line_count_raw, invalid_line_count
    return pd.DataFrame(), [], line_count_raw, invalid_line_count


def _count_jsonl_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _performance_section(cfg: dict[str, Any]) -> dict[str, Any]:
    section = cfg.get("performance")
    return section if isinstance(section, dict) else {}


def _performance_digest(cfg: dict[str, Any]) -> str:
    performance = _performance_section(cfg)
    return json.dumps(performance, sort_keys=True, separators=(",", ":"))


def _record_has_world_spec(record: dict[str, Any]) -> bool:
    world_spec = record.get("world_spec")
    return isinstance(world_spec, dict) and bool(world_spec)

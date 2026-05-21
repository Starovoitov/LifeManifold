"""Load MAP-Elites archive JSONL into collapsed tables and precomputed heatmap pivots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.utils.bootstrap import ensure_repo_on_path
from dashboard.utils.config import existing_archive_paths, load_config
from dashboard.utils.data_processing import (
    elite_to_flat_row,
    try_flatten_archive_record,
)

ensure_repo_on_path()

from worldspace.illuminators.archive import (
    DEFAULT_GRID_RESOLUTION,
    load_and_collapse_jsonl,
)

__all__ = [
    "ArchiveBundle",
    "build_pivots",
    "collapse_dataframe",
    "get_archive_bundle",
    "load_archive_bundle",
    "read_archive_jsonl",
    "show_large_archive_warning",
]


@dataclass(frozen=True)
class ArchiveBundle:
    """Collapsed archive elites plus precomputed heatmap grids."""

    collapsed: pd.DataFrame
    pivots: dict[str, np.ndarray]
    resolution: int
    line_count_raw: int
    large_archive_mode: bool
    source_path: str


def get_archive_bundle(path: Path | None = None) -> ArchiveBundle:
    """Load an archive with Streamlit disk cache (path + mtime)."""
    cfg = load_config()
    target = _resolve_archive_path(path, cfg)
    mtime = _file_mtime(target)
    return _cached_load_archive_bundle(
        str(target.resolve()), mtime, _performance_digest(cfg)
    )


def load_archive_bundle(
    archive_path: Path,
    mtime: float,
    cfg: dict[str, Any],
) -> ArchiveBundle:
    """Load JSONL, collapse by bin, and precompute pivot arrays."""
    del mtime  # part of Streamlit cache key in ``get_archive_bundle``
    performance = _performance_section(cfg)
    resolution = _resolution_from_config(cfg)
    threshold = int(performance.get("large_archive_line_threshold", 5000))
    heatmap_metrics = _heatmap_metric_names(performance)

    collapsed, line_count_raw = read_archive_jsonl(archive_path, cfg)
    collapsed = collapse_dataframe(collapsed)
    pivots = build_pivots(collapsed, heatmap_metrics, resolution)

    return ArchiveBundle(
        collapsed=collapsed,
        pivots=pivots,
        resolution=resolution,
        line_count_raw=line_count_raw,
        large_archive_mode=line_count_raw > threshold,
        source_path=str(archive_path.resolve()),
    )


def read_archive_jsonl(
    path: Path,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, int]:
    """Read JSONL into a flat DataFrame and return ``(frame, raw_line_count)``."""
    performance = _performance_section(cfg)
    prefer_polars = bool(performance.get("prefer_polars", True))

    line_count = _count_jsonl_lines(path)

    if prefer_polars:
        frame = _read_jsonl_polars(path)
        if frame is not None:
            return frame, line_count

    return _read_jsonl_python(path, line_count)


def collapse_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the highest-fitness elite per ``(bin_x, bin_y)``."""
    if frame.empty:
        return frame.copy()
    required = {"bin_x", "bin_y", "fitness"}
    missing = required - set(frame.columns)
    if missing:
        msg = f"collapse_dataframe missing columns: {sorted(missing)}"
        raise ValueError(msg)
    ordered = frame.sort_values("fitness", ascending=False)
    return ordered.drop_duplicates(subset=["bin_x", "bin_y"], keep="first").reset_index(
        drop=True
    )


def build_pivots(
    collapsed: pd.DataFrame,
    metrics: list[str],
    resolution: int,
) -> dict[str, np.ndarray]:
    """Build ``resolution x resolution`` grids (NaN = empty bin)."""
    pivots: dict[str, np.ndarray] = {}
    if collapsed.empty:
        for metric in metrics:
            pivots[metric] = np.full((resolution, resolution), np.nan)
        return pivots

    for metric in metrics:
        grid = np.full((resolution, resolution), np.nan, dtype=np.float64)
        if metric not in collapsed.columns:
            pivots[metric] = grid
            continue
        subset = collapsed[["bin_x", "bin_y", metric]].dropna(subset=[metric])
        for _, record in subset.iterrows():
            i = int(record["bin_x"])
            j = int(record["bin_y"])
            value = float(record[metric])
            if 0 <= i < resolution and 0 <= j < resolution:
                grid[i, j] = value
        pivots[metric] = grid
    return pivots


def show_large_archive_warning(bundle: ArchiveBundle, cfg: dict[str, Any]) -> None:
    """Display performance guard info when the archive exceeds the configured threshold."""
    if not bundle.large_archive_mode:
        return
    performance = _performance_section(cfg)
    threshold = int(performance.get("large_archive_line_threshold", 5000))
    table_max = int(performance.get("table_max_rows", 500))
    prefer_polars = bool(performance.get("prefer_polars", True))
    st.warning(
        f"Large archive mode: {bundle.line_count_raw:,} JSONL lines (threshold {threshold:,}). "
        f"Active guards: Polars read={prefer_polars}, precomputed pivots, "
        f"in-memory filters, table cap={table_max}."
    )


@st.cache_data(show_spinner=False)
def _cached_load_archive_bundle(
    archive_path_str: str,
    mtime: float,
    performance_digest: str,
) -> ArchiveBundle:
    del performance_digest
    cfg = load_config()
    return load_archive_bundle(Path(archive_path_str), mtime, cfg)


def _resolve_archive_path(path: Path | None, cfg: dict[str, Any]) -> Path:
    if path is not None:
        target = Path(path)
        if not target.is_file():
            msg = f"archive file not found: {target}"
            raise FileNotFoundError(msg)
        return target
    candidates = existing_archive_paths(cfg)
    if not candidates:
        msg = "no archive JSONL found; check dashboard config paths"
        raise FileNotFoundError(msg)
    return candidates[0]


def _file_mtime(path: Path) -> float:
    return float(path.stat().st_mtime)


def _performance_section(cfg: dict[str, Any]) -> dict[str, Any]:
    section = cfg.get("performance")
    return section if isinstance(section, dict) else {}


def _performance_digest(cfg: dict[str, Any]) -> str:
    performance = _performance_section(cfg)
    return json.dumps(performance, sort_keys=True, separators=(",", ":"))


def _resolution_from_config(cfg: dict[str, Any]) -> int:
    defaults = cfg.get("defaults")
    if isinstance(defaults, dict) and "grid_resolution" in defaults:
        return int(defaults["grid_resolution"])
    return DEFAULT_GRID_RESOLUTION


def _heatmap_metric_names(performance: dict[str, Any]) -> list[str]:
    raw = performance.get("heatmap_metrics")
    if isinstance(raw, list) and raw:
        return [str(name) for name in raw]
    return ["fitness", "stability", "diversity"]


def _read_jsonl_polars(path: Path) -> pd.DataFrame | None:
    try:
        import polars as pl
    except ImportError:
        return None

    try:
        frame = pl.read_ndjson(path)
    except Exception:
        return None

    rows: list[dict[str, Any]] = []
    for record in frame.to_dicts():
        row = try_flatten_archive_record(record)
        if row is not None:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _read_jsonl_python(path: Path, line_count: int) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            row = try_flatten_archive_record(record)
            if row is not None:
                rows.append(row)
    if rows:
        return pd.DataFrame(rows), line_count
    return _read_jsonl_via_worldspace(path)


def _count_jsonl_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _read_jsonl_via_worldspace(path: Path) -> tuple[pd.DataFrame, int]:
    line_count = _count_jsonl_lines(path)
    cfg = load_config()
    resolution = _resolution_from_config(cfg)
    archive = load_and_collapse_jsonl(path, resolution=resolution)
    rows: list[dict[str, Any]] = []
    size = archive.resolution
    for i in range(size):
        for j in range(size):
            elite = archive.get(i, j)
            if elite is not None:
                rows.append(elite_to_flat_row(elite))
    return pd.DataFrame(rows), line_count

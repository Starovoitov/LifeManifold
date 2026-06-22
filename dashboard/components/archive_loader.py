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
    ArchiveType,
    elite_to_flat_row,
    try_flatten_archive_record,
)

ensure_repo_on_path()

from worldspace.illuminators.archive import (
    DEFAULT_GRID_RESOLUTION,
    load_and_collapse_jsonl,
)
from worldspace.illuminators.cvt import centroids_path_for_output, load_centroids

__all__ = [
    "ArchiveBundle",
    "build_pivots",
    "collapse_dataframe",
    "detect_archive_type_from_jsonl",
    "get_archive_bundle",
    "load_archive_bundle",
    "load_centroids_for_bundle",
    "read_archive_jsonl",
    "show_centroids_warning",
    "show_large_archive_warning",
]

NIGHTLY_SUMMARY_FILENAME = "nightly_run_summary.json"


@dataclass(frozen=True)
class ArchiveBundle:
    """Collapsed archive elites plus precomputed heatmap grids."""

    collapsed: pd.DataFrame
    pivots: dict[str, np.ndarray]
    resolution: int
    archive_type: ArchiveType
    n_cells: int
    centroids: np.ndarray | None
    centroids_missing: bool
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
    """Load JSONL, collapse by bin or cell_id, and precompute pivot arrays."""
    del mtime  # part of Streamlit cache key in ``get_archive_bundle``
    performance = _performance_section(cfg)
    resolution = _resolution_from_config(cfg)
    threshold = int(performance.get("large_archive_line_threshold", 5000))
    heatmap_metrics = _heatmap_metric_names(performance)

    archive_type = detect_archive_type_from_jsonl(archive_path)
    collapsed, line_count_raw = read_archive_jsonl(
        archive_path,
        cfg,
        archive_type=archive_type,
    )
    collapsed = collapse_dataframe(collapsed, archive_type=archive_type)

    centroids = load_centroids_for_bundle(archive_path)
    centroids_missing = archive_type == "cvt" and centroids is None
    collapsed = _attach_centroid_columns(collapsed, centroids)

    if archive_type == "cvt":
        n_cells = (
            int(centroids.shape[0])
            if centroids is not None
            else _cvt_n_cells_hint(collapsed)
        )
        pivots = _empty_pivots(heatmap_metrics)
        bundle_resolution = max(1, int(np.ceil(np.sqrt(n_cells))))
    else:
        n_cells = resolution * resolution
        pivots = build_pivots(collapsed, heatmap_metrics, resolution)
        bundle_resolution = resolution

    return ArchiveBundle(
        collapsed=collapsed,
        pivots=pivots,
        resolution=bundle_resolution,
        archive_type=archive_type,
        n_cells=n_cells,
        centroids=centroids,
        centroids_missing=centroids_missing,
        line_count_raw=line_count_raw,
        large_archive_mode=line_count_raw > threshold,
        source_path=str(archive_path.resolve()),
    )


def detect_archive_type_from_jsonl(path: Path) -> ArchiveType:
    """Infer archive type from the first valid JSONL line (fallback: grid)."""
    if not path.is_file():
        return "grid"
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        schema_version = str(record.get("schema_version", "1.2"))
        if schema_version == "1.2":
            return "grid"
        archive_type = str(record.get("archive_type", "grid"))
        return "cvt" if archive_type == "cvt" else "grid"
    summary_type = _archive_type_from_nightly_summary(path.parent)
    return summary_type if summary_type is not None else "grid"


def load_centroids_for_bundle(archive_path: Path) -> np.ndarray | None:
    """Load ``cvt_centroids.json`` next to the archive JSONL, if present."""
    centroids_path = centroids_path_for_output(archive_path.parent)
    if not centroids_path.is_file():
        return None
    try:
        centroids = load_centroids(centroids_path)
    except (OSError, ValueError):
        return None
    if centroids.ndim != 2 or centroids.shape[1] != 2 or centroids.shape[0] < 1:
        return None
    return centroids


def read_archive_jsonl(
    path: Path,
    cfg: dict[str, Any],
    *,
    archive_type: ArchiveType = "grid",
) -> tuple[pd.DataFrame, int]:
    """Read JSONL into a flat DataFrame and return ``(frame, raw_line_count)``."""
    performance = _performance_section(cfg)
    prefer_polars = bool(performance.get("prefer_polars", True))

    line_count = _count_jsonl_lines(path)

    if prefer_polars:
        frame = _read_jsonl_polars(path)
        if frame is not None:
            return frame, line_count

    frame, count = _read_jsonl_python(path, line_count)
    if not frame.empty:
        return frame, count
    return _read_jsonl_via_worldspace(path, archive_type=archive_type)


def collapse_dataframe(
    frame: pd.DataFrame,
    *,
    archive_type: ArchiveType = "grid",
) -> pd.DataFrame:
    """Keep the highest-fitness elite per niche (grid bin or CVT cell)."""
    if frame.empty:
        return frame.copy()
    if "fitness" not in frame.columns:
        msg = "collapse_dataframe missing column: fitness"
        raise ValueError(msg)

    ordered = frame.sort_values("fitness", ascending=False)
    if archive_type == "cvt":
        if "cell_id" not in ordered.columns:
            msg = "collapse_dataframe missing column: cell_id"
            raise ValueError(msg)
        return ordered.drop_duplicates(subset=["cell_id"], keep="first").reset_index(
            drop=True
        )

    required = {"bin_x", "bin_y"}
    missing = required - set(ordered.columns)
    if missing:
        msg = f"collapse_dataframe missing columns: {sorted(missing)}"
        raise ValueError(msg)
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
        metric_values = collapsed[metric]
        subset = collapsed.loc[metric_values.notna(), ["bin_x", "bin_y", metric]]
        bin_x = subset["bin_x"].to_numpy(dtype=np.int64, copy=False)
        bin_y = subset["bin_y"].to_numpy(dtype=np.int64, copy=False)
        values = subset[metric].to_numpy(dtype=np.float64, copy=False)
        for idx in range(len(subset)):
            i = int(bin_x[idx])
            j = int(bin_y[idx])
            value = float(values[idx])
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


def show_centroids_warning(bundle: ArchiveBundle) -> None:
    """Warn when a CVT archive lacks ``cvt_centroids.json``."""
    if bundle.archive_type != "cvt" or not bundle.centroids_missing:
        return
    st.warning(
        "CVT centroids file not found next to this archive. "
        "Scatter view is degraded (elite positions only; empty niches hidden)."
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


def _empty_pivots(metrics: list[str]) -> dict[str, np.ndarray]:
    return {metric: np.full((1, 1), np.nan) for metric in metrics}


def _attach_centroid_columns(
    collapsed: pd.DataFrame,
    centroids: np.ndarray | None,
) -> pd.DataFrame:
    if centroids is None or collapsed.empty or "cell_id" not in collapsed.columns:
        return collapsed
    frame = collapsed.copy()
    cell_ids = frame["cell_id"].to_numpy(dtype=np.int64, copy=False)
    valid = (cell_ids >= 0) & (cell_ids < centroids.shape[0])
    centroid_s = np.full(len(frame), np.nan, dtype=np.float64)
    centroid_d = np.full(len(frame), np.nan, dtype=np.float64)
    centroid_s[valid] = centroids[cell_ids[valid], 0]
    centroid_d[valid] = centroids[cell_ids[valid], 1]
    frame["centroid_s"] = centroid_s
    frame["centroid_d"] = centroid_d
    return frame


def _cvt_n_cells_hint(collapsed: pd.DataFrame) -> int:
    if collapsed.empty or "cell_id" not in collapsed.columns:
        return 1
    cell_ids = collapsed["cell_id"].dropna()
    if cell_ids.empty:
        return 1
    cell_max = int(cell_ids.to_numpy(dtype=np.int64, copy=False).max())
    return max(1, cell_max + 1)


def _archive_type_from_nightly_summary(run_dir: Path) -> ArchiveType | None:
    summary_path = run_dir / NIGHTLY_SUMMARY_FILENAME
    if not summary_path.is_file():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    archive_type = str(payload.get("archive_type", "grid"))
    return "cvt" if archive_type == "cvt" else "grid"


def _read_jsonl_polars(path: Path) -> pd.DataFrame | None:
    try:
        import polars as pl
        from polars.exceptions import PanicException
    except ImportError:
        return None

    try:
        frame = pl.read_ndjson(path)
    except (Exception, PanicException):
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
    return pd.DataFrame(), line_count


def _read_jsonl_via_worldspace(
    path: Path,
    *,
    archive_type: ArchiveType,
) -> tuple[pd.DataFrame, int]:
    line_count = _count_jsonl_lines(path)
    cfg = load_config()
    resolution = _resolution_from_config(cfg)
    centroids_path = centroids_path_for_output(path.parent)
    if archive_type == "cvt":
        if not centroids_path.is_file():
            return pd.DataFrame(), line_count
        archive = load_and_collapse_jsonl(
            path,
            archive_type="cvt",
            centroids_path=centroids_path,
        )
    else:
        archive = load_and_collapse_jsonl(path, resolution=resolution)
    rows: list[dict[str, Any]] = []
    for cell_id in range(archive.n_cells):
        elite = archive.get_cell(cell_id)
        if elite is not None:
            rows.append(
                elite_to_flat_row(
                    elite,
                    archive_type=archive_type,
                    resolution=resolution if archive_type == "grid" else None,
                )
            )
    return pd.DataFrame(rows), line_count


def _count_jsonl_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())

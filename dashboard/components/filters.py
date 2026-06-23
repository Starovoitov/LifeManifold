"""Sidebar archive filters and in-memory filtering (no JSONL re-read)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.components.archive_loader import ArchiveBundle, build_pivots
from dashboard.components.metrics import add_metrics_columns
from dashboard.utils.config import load_config

__all__ = [
    "FilterState",
    "apply_collapsed_filters",
    "rebuild_pivots_from_collapsed",
    "render_archive_filters",
]


@dataclass(frozen=True)
class FilterState:
    """User-selected archive filters."""

    archive_path: Path
    heatmap_metric: str
    min_fitness: float
    seed: int | None
    emitter_type: str | None
    resolution: int
    langton_lambda_min: float | None = None
    langton_lambda_max: float | None = None


def render_archive_filters(
    bundle: ArchiveBundle,
    archive_path: Path,
) -> FilterState:
    """Render sidebar controls and return the active filter state."""
    cfg = load_config()
    defaults = cfg.get("defaults")
    default_block = defaults if isinstance(defaults, dict) else {}

    with st.sidebar:
        st.header("Archive filters")
        metric_options = list(bundle.pivots.keys())
        default_metric = str(default_block.get("heatmap_metric", "fitness"))
        metric_index = (
            metric_options.index(default_metric)
            if default_metric in metric_options
            else 0
        )
        heatmap_metric = st.selectbox(
            "Chart metric" if bundle.archive_type == "cvt" else "Heatmap metric",
            metric_options,
            index=metric_index,
        )
        min_fitness = st.slider(
            "Min fitness",
            min_value=0.0,
            max_value=1.0,
            value=float(default_block.get("min_fitness", 0.0)),
            step=0.01,
        )

        seed_labels = ["Any"]
        seed_values: list[int | None] = [None]
        if "seed" in bundle.collapsed.columns:
            for value in sorted(bundle.collapsed["seed"].dropna().unique()):
                seed_values.append(int(value))
                seed_labels.append(str(value))
        seed_choice = st.selectbox("Seed", seed_labels, index=0)
        seed = seed_values[seed_labels.index(seed_choice)]

        emitter_labels = ["Any"]
        emitter_values: list[str | None] = [None]
        if "emitter_type" in bundle.collapsed.columns:
            for value in sorted(bundle.collapsed["emitter_type"].dropna().unique()):
                emitter_values.append(str(value))
                emitter_labels.append(str(value))
        emitter_choice = st.selectbox("Emitter type", emitter_labels, index=0)
        emitter_type = emitter_values[emitter_labels.index(emitter_choice)]

        st.divider()
        st.subheader("Langton λ_runtime")
        langton_series = _langton_runtime_series(bundle.collapsed)
        langton_lambda_min: float | None = None
        langton_lambda_max: float | None = None
        if langton_series is None:
            st.caption(
                "Not stored in this archive JSONL. Re-run MAP-Elites to persist "
                "λ_runtime, or read it from the diagnostic panel (live simulation)."
            )
        else:
            data_min = float(langton_series.min())
            data_max = float(langton_series.max())
            default_lo = float(np.clip(data_min, 0.0, 1.0))
            default_hi = float(np.clip(data_max, 0.0, 1.0))
            if default_lo > default_hi:
                default_lo, default_hi = default_hi, default_lo
            langton_lambda_min, langton_lambda_max = st.slider(
                "λ_runtime range",
                min_value=0.0,
                max_value=1.0,
                value=(default_lo, default_hi),
                step=0.01,
                help=(
                    "Mean per-step fraction of life cells that flip state. "
                    "Filter applies only to elites with stored λ_runtime."
                ),
            )

    return FilterState(
        archive_path=archive_path,
        heatmap_metric=heatmap_metric,
        min_fitness=min_fitness,
        seed=seed,
        emitter_type=emitter_type,
        resolution=bundle.resolution,
        langton_lambda_min=langton_lambda_min,
        langton_lambda_max=langton_lambda_max,
    )


def apply_collapsed_filters(
    collapsed: pd.DataFrame,
    state: FilterState,
) -> pd.DataFrame:
    """Filter collapsed elites in memory."""
    if collapsed.empty:
        return collapsed.copy()
    mask = collapsed["fitness"] >= state.min_fitness
    if state.seed is not None and "seed" in collapsed.columns:
        mask &= collapsed["seed"] == state.seed
    if state.emitter_type is not None and "emitter_type" in collapsed.columns:
        mask &= collapsed["emitter_type"] == state.emitter_type
    if (
        state.langton_lambda_min is not None
        and state.langton_lambda_max is not None
        and "langton_lambda_runtime" in collapsed.columns
    ):
        raw_langton = collapsed["langton_lambda_runtime"]
        if isinstance(raw_langton, pd.Series):
            langton = _coerce_float_series(raw_langton)
            mask &= langton.notna()
            mask &= langton >= state.langton_lambda_min
            mask &= langton <= state.langton_lambda_max
    return collapsed.loc[mask].reset_index(drop=True)


def _coerce_float_series(column: pd.Series) -> pd.Series:
    """Coerce one archive metric column to float (always a Series for type checkers)."""
    converted = pd.to_numeric(column, errors="coerce")
    if isinstance(converted, pd.Series):
        return converted
    return pd.Series(converted, index=column.index, dtype=float)


def _langton_runtime_series(collapsed: pd.DataFrame) -> pd.Series | None:
    """Return λ_runtime column when the archive has at least one finite stored value."""
    if collapsed.empty:
        return None
    enriched = add_metrics_columns(collapsed)
    if "langton_lambda_runtime" not in enriched.columns:
        return None
    raw = enriched["langton_lambda_runtime"]
    if not isinstance(raw, pd.Series):
        return None
    series = _coerce_float_series(raw)
    if series.dropna().empty:
        return None
    return series


def rebuild_pivots_from_collapsed(
    collapsed: pd.DataFrame,
    metrics: list[str],
    resolution: int,
) -> dict[str, Any]:
    """Recompute pivot grids from a filtered collapsed frame."""
    return build_pivots(collapsed, metrics, resolution)

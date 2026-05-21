"""Streamlit UI sections for the surrogate training buffer page."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.components.training_buffer_loader import (
    TARGET_COLUMN_PREFIX,
    BufferBundle,
    buffer_summary_counts,
    export_subset_jsonl,
    slice_for_display,
)
from dashboard.utils.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from worldspace.surrogate.model import TARGET_KEYS

__all__ = [
    "BufferFilterState",
    "effective_table_page_size",
    "render_buffer_export",
    "render_buffer_filters",
    "render_buffer_stats",
    "render_buffer_table",
    "target_distribution_chart_frame",
]

_DISPLAY_COLUMNS = [
    "emitter_type",
    "feature_schema_version",
    "feature_dim",
    *[f"{TARGET_COLUMN_PREFIX}{key}" for key in TARGET_KEYS],
]

_TARGET_CHART_KEYS = ("stability", "diversity", "final_density")
_MAX_TABLE_PAGE_SIZE = 500


@dataclass(frozen=True)
class BufferFilterState:
    """Sidebar filter selections for buffer rows."""

    emitter_types: list[str]
    schema_versions: list[str]


def render_buffer_stats(bundle: BufferBundle) -> None:
    """Summary metrics and breakdown tables for the training buffer."""
    st.subheader("Buffer summary")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Valid records", len(bundle.records))
    with col_b:
        st.metric("JSONL lines (non-empty)", bundle.line_count_raw)
    with col_c:
        st.metric("Skipped / invalid lines", bundle.invalid_line_count)

    counts = buffer_summary_counts(bundle.records)
    if not counts:
        st.info("No valid buffer records to summarize.")
        return

    left, right = st.columns(2)
    with left:
        st.markdown("**By emitter_type**")
        emitter = counts.get("emitter_type")
        if emitter is not None and not emitter.empty:
            st.dataframe(emitter.rename("count").to_frame(), use_container_width=True)
        else:
            st.caption("_no data_")
    with right:
        st.markdown("**By feature_schema_version**")
        schema = counts.get("feature_schema_version")
        if schema is not None and not schema.empty:
            st.dataframe(schema.rename("count").to_frame(), use_container_width=True)
        else:
            st.caption("_no data_")

    _render_target_distributions(bundle.records)


def render_buffer_filters(bundle: BufferBundle) -> BufferFilterState:
    """Sidebar filters applied in memory (no disk reload)."""
    frame = bundle.records
    emitter_options = (
        sorted(frame["emitter_type"].dropna().unique().tolist())
        if not frame.empty
        else []
    )
    schema_options = (
        sorted(frame["feature_schema_version"].dropna().unique().tolist())
        if not frame.empty
        else []
    )

    with st.sidebar:
        st.subheader("Filters")
        emitter_types = st.multiselect(
            "Emitter type",
            emitter_options,
            default=emitter_options,
            key="buffer_filter_emitter",
        )
        schema_versions = st.multiselect(
            "Feature schema version",
            schema_options,
            default=schema_options,
            key="buffer_filter_schema",
        )

    return BufferFilterState(
        emitter_types=list(emitter_types),
        schema_versions=list(schema_versions),
    )


def render_buffer_table(
    filtered: pd.DataFrame,
    *,
    table_max_rows: int,
) -> None:
    """Paginated table capped at ``table_max_rows`` rows per page."""
    st.subheader("Records")
    total = len(filtered)
    if total == 0:
        st.warning("No records match the current filters.")
        return

    page_size = effective_table_page_size(table_max_rows)
    page_count = max(1, (total + page_size - 1) // page_size)

    if total > page_size:
        st.warning(
            f"Showing at most {page_size} rows per page "
            f"({total:,} records match filters). Use pagination or export."
        )

    page = st.number_input(
        "Page",
        min_value=1,
        max_value=page_count,
        value=1,
        step=1,
        key="buffer_table_page",
    )
    page_index = int(page) - 1
    display = slice_for_display(
        filtered,
        page=page_index,
        page_size=page_size,
        max_rows=page_size,
    )
    columns = [name for name in _DISPLAY_COLUMNS if name in display.columns]
    st.caption(
        f"Rows {page_index * page_size + 1}–{page_index * page_size + len(display)} of {total:,}"
    )
    st.dataframe(display[columns], use_container_width=True, hide_index=True)


def render_buffer_export(
    filtered_raw: list[dict[str, Any]],
    *,
    source_name: str,
) -> None:
    """Download button for the filtered JSONL subset."""
    st.subheader("Export")
    if not filtered_raw:
        st.caption("No rows to export.")
        return
    payload = export_subset_jsonl(filtered_raw)
    st.download_button(
        label=f"Download filtered subset ({len(filtered_raw):,} rows)",
        data=payload,
        file_name=f"{source_name}_subset.jsonl",
        mime="application/jsonl",
    )


def _render_target_distributions(frame: pd.DataFrame) -> None:
    """Bar charts for a subset of training targets."""
    if frame.empty:
        return
    st.markdown("**Target distributions**")
    tabs = st.tabs([key.replace("_", " ").title() for key in _TARGET_CHART_KEYS])
    for tab, key in zip(tabs, _TARGET_CHART_KEYS, strict=True):
        col_name = f"{TARGET_COLUMN_PREFIX}{key}"
        with tab:
            if col_name not in frame.columns:
                st.caption("_column missing_")
                continue
            series = frame[col_name].dropna()
            if series.empty:
                st.caption("_no values_")
                continue
            chart_df = target_distribution_chart_frame(series)
            if not chart_df.empty:
                st.bar_chart(chart_df)

    with st.expander("All training targets"):
        for key in TARGET_KEYS:
            col_name = f"{TARGET_COLUMN_PREFIX}{key}"
            if col_name not in frame.columns:
                continue
            st.caption(key.replace("_", " ").title())
            series = frame[col_name].dropna()
            chart_df = target_distribution_chart_frame(series)
            if not chart_df.empty:
                st.bar_chart(chart_df)


def effective_table_page_size(table_max_rows: int) -> int:
    """Clamp config ``table_max_rows`` to a safe positive page size."""
    return max(1, min(int(table_max_rows), _MAX_TABLE_PAGE_SIZE))


def target_distribution_chart_frame(series: pd.Series) -> pd.DataFrame:
    """Build a histogram DataFrame for ``st.bar_chart`` (binned continuous targets)."""
    clean = series.dropna()
    if clean.empty:
        return pd.DataFrame()
    unique_count = int(clean.nunique())
    n_bins = min(20, max(5, len(clean) // 10))
    if unique_count <= n_bins:
        counts = clean.value_counts().sort_index()
        return counts.rename("count").to_frame()
    binned = pd.cut(clean, bins=n_bins)
    counts = binned.value_counts().sort_index()
    return counts.rename("count").to_frame()

"""KPIs, filters, and charts for SurrogateArchive acquisition logs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.surrogate_archive_loader import ArchiveLogBundle
from dashboard.utils.plotting import apply_dark_theme, default_figure_height

__all__ = [
    "acquisition_kpis",
    "plot_cumulative_skips",
    "plot_skips_per_iteration",
    "render_acquisition_charts",
    "render_acquisition_filters",
    "render_acquisition_kpis",
    "render_acquisition_table",
    "skips_by_iteration",
]


@dataclass(frozen=True)
class AcquisitionFilterState:
    """Sidebar filter selections for the acquisition log page."""

    decisions: list[str]
    acquisition_modes: list[str]
    emitter_types: list[str]
    iteration_range: tuple[int, int] | None


def acquisition_kpis(frame: pd.DataFrame) -> dict[str, float | int]:
    """Compute skip-rate and shadow vs filter metrics for a filtered frame."""
    total = len(frame)
    if total == 0:
        return {
            "total": 0,
            "skip_count": 0,
            "skip_rate_pct": 0.0,
            "shadow_would_skip": 0,
            "shadow_would_skip_pct": 0.0,
            "filter_actual_skip": 0,
            "filter_actual_skip_pct": 0.0,
        }

    skip_mask = frame["decision"] == "skip"
    skip_count = int(skip_mask.sum())
    shadow = frame[frame["acquisition_mode"] == "shadow"]
    filter_rows = frame[frame["acquisition_mode"] == "filter"]
    shadow_would = int((shadow["decision"] == "skip").sum()) if len(shadow) else 0
    filter_skip = (
        int((filter_rows["decision"] == "skip").sum()) if len(filter_rows) else 0
    )

    return {
        "total": total,
        "skip_count": skip_count,
        "skip_rate_pct": 100.0 * skip_count / total,
        "shadow_would_skip": shadow_would,
        "shadow_would_skip_pct": (
            100.0 * shadow_would / len(shadow) if len(shadow) else 0.0
        ),
        "filter_actual_skip": filter_skip,
        "filter_actual_skip_pct": (
            100.0 * filter_skip / len(filter_rows) if len(filter_rows) else 0.0
        ),
    }


def skips_by_iteration(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate skip counts per iteration (int axis for charts)."""
    if frame.empty:
        return pd.DataFrame(columns=["iteration", "skips", "evals", "cumulative_skips"])
    grouped = (
        frame.groupby("iteration", as_index=False)
        .agg(
            skips=("decision", lambda s: int((s == "skip").sum())),
            evals=("decision", lambda s: int((s == "eval").sum())),
        )
        .sort_values("iteration")
    )
    grouped["cumulative_skips"] = grouped["skips"].cumsum()
    return grouped


def plot_skips_per_iteration(stats: pd.DataFrame) -> go.Figure:
    """Bar chart of skips per scheduler iteration."""
    fig = go.Figure()
    if stats.empty:
        fig.update_layout(title="Skips per iteration (no data)")
        return apply_dark_theme(fig)
    fig.add_trace(
        go.Bar(
            x=stats["iteration"],
            y=stats["skips"],
            name="skips",
        )
    )
    fig.update_layout(
        title="Skips per iteration",
        xaxis_title="iteration",
        yaxis_title="skip count",
        height=default_figure_height(),
    )
    return apply_dark_theme(fig)


def plot_cumulative_skips(stats: pd.DataFrame) -> go.Figure:
    """Line chart of cumulative skips across iterations."""
    fig = go.Figure()
    if stats.empty:
        fig.update_layout(title="Cumulative skips (no data)")
        return apply_dark_theme(fig)
    fig.add_trace(
        go.Scatter(
            x=stats["iteration"],
            y=stats["cumulative_skips"],
            mode="lines+markers",
            name="cumulative skips",
        )
    )
    fig.update_layout(
        title="Cumulative skips",
        xaxis_title="iteration",
        yaxis_title="cumulative skip count",
        height=default_figure_height(),
    )
    return apply_dark_theme(fig)


def render_acquisition_kpis(frame: pd.DataFrame) -> None:
    """Display acquisition KPI metrics."""
    metrics = acquisition_kpis(frame)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Slots", int(metrics["total"]))
    c2.metric("Skip rate", f"{metrics['skip_rate_pct']:.1f}%")
    c3.metric(
        "Shadow would-skip",
        f"{int(metrics['shadow_would_skip'])} ({metrics['shadow_would_skip_pct']:.1f}%)",
    )
    c4.metric(
        "Filter actual skip",
        f"{int(metrics['filter_actual_skip'])} ({metrics['filter_actual_skip_pct']:.1f}%)",
    )


def render_acquisition_filters(bundle: ArchiveLogBundle) -> AcquisitionFilterState:
    """Render sidebar multiselect filters."""
    frame = bundle.records
    if frame.empty:
        return AcquisitionFilterState([], [], [], None)

    decisions = sorted(frame["decision"].dropna().unique().tolist())
    modes = sorted(frame["acquisition_mode"].dropna().unique().tolist())
    emitters = sorted(frame["emitter_type"].dropna().unique().tolist())
    iter_min = int(frame["iteration"].min())
    iter_max = int(frame["iteration"].max())

    selected_decisions = st.sidebar.multiselect(
        "Decision",
        decisions,
        default=decisions,
        key="acq_filter_decision",
    )
    selected_modes = st.sidebar.multiselect(
        "Acquisition mode",
        modes,
        default=modes,
        key="acq_filter_mode",
    )
    selected_emitters = st.sidebar.multiselect(
        "Emitter type",
        emitters,
        default=emitters,
        key="acq_filter_emitter",
    )
    use_iter_range = st.sidebar.checkbox(
        "Limit iteration range",
        value=False,
        key="acq_filter_iter_toggle",
    )
    iteration_range: tuple[int, int] | None = None
    if use_iter_range:
        low, high = st.sidebar.slider(
            "Iteration",
            min_value=iter_min,
            max_value=iter_max,
            value=(iter_min, iter_max),
            key="acq_filter_iter_slider",
        )
        iteration_range = (int(low), int(high))

    return AcquisitionFilterState(
        decisions=selected_decisions,
        acquisition_modes=selected_modes,
        emitter_types=selected_emitters,
        iteration_range=iteration_range,
    )


def render_acquisition_charts(frame: pd.DataFrame) -> None:
    """Render skip aggregation charts."""
    stats = skips_by_iteration(frame)
    col_bar, col_line = st.columns(2)
    with col_bar:
        st.plotly_chart(plot_skips_per_iteration(stats), use_container_width=True)
    with col_line:
        st.plotly_chart(plot_cumulative_skips(stats), use_container_width=True)


def render_acquisition_table(frame: pd.DataFrame, *, table_max_rows: int) -> None:
    """Paginated decision table (no list-typed columns)."""
    st.subheader("Decision log")
    if frame.empty:
        st.info("No rows match the current filters.")
        return
    display = frame.head(max(1, table_max_rows))
    st.dataframe(display, use_container_width=True, hide_index=True)
    if len(frame) > len(display):
        st.caption(f"Showing {len(display)} of {len(frame)} rows.")

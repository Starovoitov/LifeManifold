"""Metrics Dashboard sections: correlations and distributions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.metrics import correlation_matrix
from dashboard.components.visualizations import (
    HISTOGRAM_METRIC_KEYS,
    create_correlation_heatmap,
    create_metric_histogram,
)

__all__ = [
    "HISTOGRAM_METRIC_KEYS",
    "render_correlation_section",
    "render_distributions_section",
]


def render_correlation_section(filtered: pd.DataFrame) -> None:
    """Show Pearson correlation heatmap over ``METRIC_KEYS``."""
    st.subheader("Metric correlations")
    corr = correlation_matrix(filtered)
    fig = create_correlation_heatmap(corr)
    st.plotly_chart(fig, width="stretch")
    if not corr.empty:
        st.caption(f"Pearson correlation on {len(filtered)} collapsed elites.")


def render_distributions_section(filtered: pd.DataFrame) -> None:
    """Show fitness, stability, and diversity histograms faceted by emitter."""
    st.subheader("Metric distributions")
    tabs = st.tabs(
        [metric.replace("_", " ").title() for metric in HISTOGRAM_METRIC_KEYS]
    )
    for tab, metric in zip(tabs, HISTOGRAM_METRIC_KEYS, strict=True):
        with tab:
            fig = create_metric_histogram(
                filtered,
                metric,
                color_by="emitter_type",
            )
            st.plotly_chart(fig, width="stretch")

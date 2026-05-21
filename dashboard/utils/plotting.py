"""Shared Plotly layout helpers for the dashboard (no Streamlit imports)."""

from __future__ import annotations

import plotly.graph_objects as go

PLOT_BG = "#0e1117"
PAPER_BG = "#0e1117"
FONT_COLOR = "#c9d1d9"

DEFAULT_CHART_HEIGHT = 520
DEFAULT_HEATMAP_HEIGHT = 620

__all__ = [
    "DEFAULT_CHART_HEIGHT",
    "DEFAULT_HEATMAP_HEIGHT",
    "apply_dark_theme",
    "default_figure_height",
]


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply the dashboard dark theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR),
    )
    return fig


def default_figure_height(*, heatmap: bool = False) -> int:
    """Return the default figure height for chart type."""
    return DEFAULT_HEATMAP_HEIGHT if heatmap else DEFAULT_CHART_HEIGHT

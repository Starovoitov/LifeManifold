"""Plotly visualization builders for the LifeManifold dashboard (no Streamlit)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.components.metrics import METRIC_HELP, metric_help_text
from dashboard.utils.bootstrap import ensure_repo_on_path
from dashboard.utils.plotting import apply_dark_theme, default_figure_height

ensure_repo_on_path()

from worldspace.metrics import METRIC_KEYS, WorldMetrics
from worldspace.simulator import SimulationResult

METRIC_COLORSCALES: dict[str, str] = {
    "fitness": "Viridis",
    "stability": "Blues",
    "diversity": "Plasma",
    "topology_interface_index": "Reds",
    "topology_window_heterogeneity": "YlOrBr",
    "compressibility_score": "Greens",
    "ecology_resource_adjacency": "Purples",
    "oscillation_score": "Cividis",
    "entropy": "Turbo",
}

RADAR_METRIC_KEYS: tuple[str, ...] = (
    "stability",
    "diversity",
    "topology_interface_index",
    "topology_window_heterogeneity",
    "compressibility_score",
    "ecology_resource_adjacency",
    "ecology_state_entropy_norm",
    "fitness",
)

BOUNDARY_COLORSCALE = [
    [0.0, "rgba(0,0,0,0)"],
    [0.3, "rgba(255,100,100,0.4)"],
    [0.6, "rgba(255,60,60,0.7)"],
    [1.0, "rgba(255,0,0,0.9)"],
]

# Base life/food field (must match ``_life_food_rgb`` and legend swatches).
_LIFE_FOOD_EMPTY = np.array([0.10, 0.11, 0.15], dtype=np.float32)
_LIFE_FOOD_LIFE_ONLY = np.array([0.29, 0.61, 0.56], dtype=np.float32)
_LIFE_FOOD_FOOD_ONLY = np.array([0.79, 0.64, 0.15], dtype=np.float32)
_LIFE_FOOD_BOTH = np.array([0.43, 0.48, 0.31], dtype=np.float32)
_BOUNDARY_WARM_MAX = np.array([1.0, 0.42, 0.12], dtype=np.float32) * 0.42
_FOOD_NEIGHBOR_BLEND = 0.48

_HETERO_LABELS: dict[float, str] = {
    0.0: "Uniform 2×2 window",
    1.0: "Mixed corners",
}

_HETERO_HELP: dict[int, str] = {
    0: ("Uniform 2×2 torus window: all four corners share the same life state."),
    1: ("Mixed 2×2 window: corners differ — local edge / blending (mesoscale proxy)."),
}

DIAGNOSTIC_PANEL_HELP: dict[str, str] = {
    "life_food": (
        "Final life/food field with warm boundary tint (life vs non-life edge strength). "
        "Hover a cell for state and boundary."
    ),
    "hetero": "Per-cell 2×2 window heterogeneity (0 = uniform, 1 = mixed corners).",
    "food_neighbor": (
        "Food density in 8 Moore neighbors; tint on live cells only (cool → warm)."
    ),
    "metrics": "All 12 world metrics on a 0–1 display scale (some rescaled for bars).",
    "radar": "Subset of metrics on the radar (values clipped to [0, 1]).",
}

DIAGNOSTIC_FIGURE_WIDTH = 1280
DIAGNOSTIC_FIGURE_HEIGHT = 920

HISTOGRAM_METRIC_KEYS: tuple[str, ...] = ("fitness", "stability", "diversity")

__all__ = [
    "BOUNDARY_COLORSCALE",
    "HISTOGRAM_METRIC_KEYS",
    "METRIC_COLORSCALES",
    "RADAR_METRIC_KEYS",
    "add_boundary_overlay",
    "create_archive_heatmap",
    "create_archive_scatter",
    "create_correlation_heatmap",
    "create_diagnostic_dashboard",
    "DIAGNOSTIC_PANEL_HELP",
    "METRIC_HELP",
    "format_diagnostic_interpretation",
    "create_metric_histogram",
    "create_metrics_radar",
    "plot_calibration_by_uncertainty",
    "plot_real_vs_predicted",
]


def create_archive_heatmap(
    pivot: np.ndarray | None = None,
    *,
    df: pd.DataFrame | None = None,
    metric: str = "fitness",
    resolution: int = 50,
    title: str | None = None,
) -> go.Figure:
    """Interactive MAP-Elites archive heatmap from a precomputed pivot grid."""
    grid = _resolve_pivot(pivot, df=df, metric=metric, resolution=resolution)
    colorscale = METRIC_COLORSCALES.get(metric, "Viridis")
    display_title = title or f"Archive — {metric.replace('_', ' ').title()}"

    fig = go.Figure(
        data=go.Heatmap(
            z=grid,
            x=list(range(resolution)),
            y=list(range(resolution)),
            colorscale=colorscale,
            colorbar=dict(title=metric),
            zmin=np.nanmin(grid) if np.any(~np.isnan(grid)) else 0.0,
            zmax=np.nanmax(grid) if np.any(~np.isnan(grid)) else 1.0,
        )
    )
    fig.update_layout(
        title=display_title,
        height=default_figure_height(heatmap=True),
        width=680,
        xaxis=dict(title="Diversity bin", ticks="outside"),
        yaxis=dict(title="Stability bin", ticks="outside", autorange="reversed"),
    )
    return apply_dark_theme(fig)


def create_archive_scatter(
    collapsed: pd.DataFrame,
    centroids: np.ndarray | None,
    metric: str = "fitness",
    *,
    title: str | None = None,
) -> go.Figure:
    """CVT archive scatter: BC niche centers colored by ``metric``; empty niches are hollow."""
    display_title = title or f"CVT archive — {metric.replace('_', ' ').title()}"
    colorscale = METRIC_COLORSCALES.get(metric, "Viridis")
    fig = go.Figure()

    if centroids is not None and centroids.ndim == 2 and centroids.shape[1] == 2:
        return _create_cvt_centroid_scatter(
            fig,
            collapsed=collapsed,
            centroids=centroids,
            metric=metric,
            title=display_title,
            colorscale=colorscale,
        )

    return _create_cvt_elite_only_scatter(
        fig,
        collapsed=collapsed,
        metric=metric,
        title=display_title,
        colorscale=colorscale,
    )


def create_correlation_heatmap(
    corr: pd.DataFrame,
    *,
    title: str = "Metric correlations",
) -> go.Figure:
    """Plotly heatmap of a Pearson correlation matrix over ``METRIC_KEYS``."""
    fig = go.Figure()
    if corr.empty or corr.shape[0] < 2 or corr.shape[1] < 2:
        fig.add_annotation(
            text="Need at least two metric columns for a correlation matrix.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(
            height=default_figure_height(), margin=dict(l=40, r=40, t=60, b=40)
        )
        return apply_dark_theme(fig)

    labels = [str(column) for column in corr.columns]
    matrix = corr.to_numpy(dtype=np.float64)
    fig.add_trace(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmin=-1.0,
            zmax=1.0,
            colorbar=dict(title="corr"),
            hovertemplate="%{y} vs %{x}<br>corr=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=default_figure_height(heatmap=True),
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),
    )
    return apply_dark_theme(fig)


def create_metric_histogram(
    frame: pd.DataFrame,
    metric: str,
    *,
    color_by: str | None = "emitter_type",
    title: str | None = None,
) -> go.Figure:
    """Overlaid histograms for one metric, optionally colored by ``emitter_type``."""
    from dashboard.components.metrics import add_metrics_columns

    fig = go.Figure()
    enriched = add_metrics_columns(frame)
    if metric not in enriched.columns:
        fig.add_annotation(
            text=f"Metric {metric!r} is not available in this archive slice.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(
            height=default_figure_height(), margin=dict(l=40, r=40, t=60, b=40)
        )
        return apply_dark_theme(fig)

    values = enriched[metric].dropna()
    if values.empty:
        fig.add_annotation(
            text="No values to plot after filtering.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(
            height=default_figure_height(), margin=dict(l=40, r=40, t=60, b=40)
        )
        return apply_dark_theme(fig)

    display_title = title or f"Distribution — {metric.replace('_', ' ').title()}"
    grouped = False
    if color_by and color_by in enriched.columns:
        groups = enriched[[metric, color_by]].dropna()
        emitter_labels = sorted({str(value) for value in groups[color_by].tolist()})
        if emitter_labels:
            grouped = True
            for emitter in emitter_labels:
                subset = groups.loc[groups[color_by].astype(str) == emitter, metric]
                fig.add_trace(
                    go.Histogram(
                        x=subset.to_numpy(dtype=np.float64),
                        name=str(emitter),
                        opacity=0.65,
                        histnorm="probability density",
                    )
                )
            fig.update_layout(barmode="overlay")

    if not grouped:
        fig.add_trace(
            go.Histogram(
                x=values.to_numpy(dtype=np.float64),
                name=metric,
                opacity=0.85,
                histnorm="probability density",
            )
        )

    fig.update_layout(
        title=display_title,
        xaxis_title=metric.replace("_", " ").title(),
        yaxis_title="Density",
        height=default_figure_height(),
    )
    return apply_dark_theme(fig)


def create_metrics_radar(metrics: dict[str, float]) -> go.Figure:
    """Radar chart for key metrics (values clipped to [0, 1] for display)."""
    categories: list[str] = []
    values: list[float] = []
    for key in RADAR_METRIC_KEYS:
        if key not in metrics:
            continue
        categories.append(key.replace("_", " ").title())
        values.append(_clip_unit(float(metrics[key])))

    fig = go.Figure()
    if len(values) < 3:
        fig.add_annotation(
            text="Need at least three metrics for radar display.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(height=340, margin=dict(l=40, r=40, t=40, b=40))
        return apply_dark_theme(fig)

    closed_values = values + [values[0]]
    closed_categories = categories + [categories[0]]
    fig.add_trace(
        go.Scatterpolar(
            r=closed_values,
            theta=closed_categories,
            fill="toself",
            line_color="#00ff9f",
            opacity=0.85,
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=340,
        margin=dict(l=60, r=60, t=30, b=30),
        showlegend=False,
    )
    return apply_dark_theme(fig)


def add_boundary_overlay(
    fig: go.Figure,
    interface_map: np.ndarray | None,
    *,
    row: int | None = None,
    col: int | None = None,
    opacity: float = 0.75,
) -> None:
    """Add a topology interface contour overlay to an existing figure."""
    if interface_map is None:
        return

    trace = go.Contour(
        z=interface_map,
        colorscale=BOUNDARY_COLORSCALE,
        opacity=opacity,
        showscale=False,
        contours=dict(
            coloring="heatmap",
            showlabels=False,
            start=0.1,
            end=1.0,
            size=0.1,
        ),
        hoverinfo="skip",
    )
    if row is not None and col is not None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace)


def plot_real_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray | None = None,
    *,
    metric_name: str = "fitness",
) -> go.Figure:
    """Scatter plot of real vs predicted values with an optional uncertainty color scale."""
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    label = metric_name.replace("_", " ").title()

    marker_kwargs: dict[str, Any] = {"size": 8, "opacity": 0.75}
    if uncertainty is not None:
        marker_kwargs["color"] = np.asarray(uncertainty, dtype=np.float64)
        marker_kwargs["colorscale"] = "Turbo"
        marker_kwargs["colorbar"] = dict(
            title="uncertainty",
            x=1.12,
            xanchor="left",
            xpad=12,
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true_arr,
            y=y_pred_arr,
            mode="markers",
            name="predictions",
            marker=marker_kwargs,
        )
    )

    combined = np.concatenate([y_true_arr, y_pred_arr])
    lo = float(np.min(combined))
    hi = float(np.max(combined))
    pad = 0.05 * (hi - lo) if hi > lo else 0.05
    axis_lo = lo - pad
    axis_hi = hi + pad
    fig.add_trace(
        go.Scatter(
            x=[axis_lo, axis_hi],
            y=[axis_lo, axis_hi],
            mode="lines",
            name="y = x",
            line=dict(color="#888888", dash="dash"),
            showlegend=True,
        )
    )
    fig.update_layout(
        title=f"Real vs predicted — {label}",
        xaxis_title=f"Real {label}",
        yaxis_title=f"Predicted {label}",
        height=default_figure_height(),
        margin=dict(r=110),
    )
    return apply_dark_theme(fig)


def plot_calibration_by_uncertainty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    *,
    n_bins: int = 8,
) -> go.Figure:
    """Bar chart of mean absolute error per uncertainty quantile bin."""
    from dashboard.utils.surrogate_analysis import calibration_table

    table = calibration_table(y_true, y_pred, uncertainty, n_bins=n_bins)
    fig = go.Figure()
    if table.empty:
        fig.add_annotation(
            text="Not enough points for calibration bins.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(
            height=default_figure_height(), margin=dict(l=40, r=40, t=40, b=40)
        )
        return apply_dark_theme(fig)

    labels = [
        f"{row['uncertainty_lo']:.2f}–{row['uncertainty_hi']:.2f}"
        for _, row in table.iterrows()
    ]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=table["mae"].to_numpy(dtype=np.float64),
            name="MAE",
            marker_color="#4a6fa5",
            hovertemplate="bin %{x}<br>MAE=%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Calibration — MAE by uncertainty bin",
        xaxis_title="Uncertainty bin",
        yaxis_title="Mean absolute error",
        height=default_figure_height(),
    )
    return apply_dark_theme(fig)


def _subplot_axis_index(row: int, col: int, *, n_cols: int = 3) -> int:
    """1-based subplot index used by Plotly ``xaxis`` / ``yaxis`` naming."""
    return (row - 1) * n_cols + col


def _subplot_domain(
    fig: go.Figure,
    row: int,
    col: int,
    *,
    n_rows: int = 2,
    n_cols: int = 3,
) -> tuple[float, float, float, float]:
    """Paper-domain bounds ``(x0, x1, y0, y1)`` for a subplot cell."""
    idx = _subplot_axis_index(row, col, n_cols=n_cols)
    x_name = "xaxis" if idx == 1 else f"xaxis{idx}"
    y_name = "yaxis" if idx == 1 else f"yaxis{idx}"
    x_dom = getattr(fig.layout, x_name).domain
    y_dom = getattr(fig.layout, y_name).domain
    return (float(x_dom[0]), float(x_dom[1]), float(y_dom[0]), float(y_dom[1]))


def _add_panel_legend(
    fig: go.Figure,
    row: int,
    col: int,
    entries: list[tuple[str, str]],
    *,
    title: str | None = None,
    anchor: str = "bottom_left",
    n_rows: int = 2,
    n_cols: int = 3,
) -> None:
    """Draw a compact color swatch legend inside a subplot using paper coordinates."""
    if not entries:
        return
    x0, x1, y0, y1 = _subplot_domain(fig, row, col, n_rows=n_rows, n_cols=n_cols)
    width = x1 - x0
    height = y1 - y0
    n_lines = len(entries) + (1 if title else 0)
    line_h = min(0.032 * height, 0.028)
    pad = 0.018 * height
    box_h = line_h * n_lines + 2.0 * pad
    box_w = min(0.48 * width, 0.19)

    if anchor == "bottom_right":
        bx1 = x1 - 0.02 * width
        bx0 = bx1 - box_w
        by0 = y0 + pad
        by1 = by0 + box_h
    else:
        bx0 = x0 + 0.02 * width
        bx1 = bx0 + box_w
        by0 = y0 + pad
        by1 = by0 + box_h

    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=bx0,
        x1=bx1,
        y0=by0,
        y1=by1,
        fillcolor="rgba(18, 20, 28, 0.88)",
        line=dict(color="rgba(255, 255, 255, 0.28)", width=1),
        layer="above",
    )

    swatch_w = min(0.022 * width, 0.012)
    text_x = bx0 + pad + swatch_w + 0.006 * width
    y_cursor = by1 - pad

    if title:
        fig.add_annotation(
            x=bx0 + pad,
            y=y_cursor,
            text=f"<b>{title}</b>",
            showarrow=False,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            font=dict(size=10, color="#e8e8e8"),
        )
        y_cursor -= line_h

    for label, color in entries:
        sw_y0 = y_cursor - 0.72 * line_h
        sw_y1 = y_cursor - 0.12 * line_h
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=bx0 + pad,
            x1=bx0 + pad + swatch_w,
            y0=sw_y0,
            y1=sw_y1,
            fillcolor=color,
            line=dict(color="rgba(255, 255, 255, 0.35)", width=0.5),
            layer="above",
        )
        fig.add_annotation(
            x=text_x,
            y=(sw_y0 + sw_y1) / 2.0,
            text=label,
            showarrow=False,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="middle",
            font=dict(size=9, color="#c8c8c8"),
        )
        y_cursor -= line_h


def _add_cell_hover_layer(
    fig: go.Figure,
    hover_text: np.ndarray,
    *,
    row: int,
    col: int,
) -> None:
    """Invisible heatmap so Image panels get per-cell hover text."""
    fig.add_trace(
        go.Heatmap(
            z=np.zeros(hover_text.shape, dtype=np.float32),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            opacity=0.0,
            showscale=False,
        ),
        row=row,
        col=col,
    )


def _life_food_hover_text(
    life: np.ndarray,
    food: np.ndarray,
    boundary: np.ndarray,
) -> np.ndarray:
    """Per-cell hover lines for the life + food · boundary panel."""
    life_on = life >= 0.5
    food_on = food >= 0.5
    both = life_on & food_on
    life_only = life_on & ~food_on
    food_only = food_on & ~life_on
    empty = ~life_on & ~food_on
    state = np.full(life.shape, "empty", dtype=object)
    state[life_only] = "life only"
    state[food_only] = "food only"
    state[both] = "life + food"
    state[empty] = "empty"
    b = boundary.astype(np.float64)
    lines = np.empty(life.shape, dtype=object)
    for i in range(life.shape[0]):
        for j in range(life.shape[1]):
            lines[i, j] = f"<b>{state[i, j]}</b><br>boundary strength={b[i, j]:.2f}"
    return lines


def _food_neighbor_hover_text(
    life: np.ndarray,
    food: np.ndarray,
    fnb: np.ndarray,
) -> np.ndarray:
    """Per-cell hover lines for the food-neighbor panel."""
    life_on = life >= 0.5
    food_on = food >= 0.5
    both = life_on & food_on
    life_only = life_on & ~food_on
    food_only = food_on & ~life_on
    empty = ~life_on & ~food_on
    state = np.full(life.shape, "empty", dtype=object)
    state[life_only] = "life only"
    state[food_only] = "food only"
    state[both] = "life + food"
    state[empty] = "empty"
    lines = np.empty(life.shape, dtype=object)
    for i in range(life.shape[0]):
        for j in range(life.shape[1]):
            if life_on[i, j]:
                lines[i, j] = (
                    f"<b>live</b> · {state[i, j]}<br>"
                    f"food in 8 neighbors={float(fnb[i, j]):.2f}"
                )
            else:
                lines[i, j] = f"<b>non-live</b> · {state[i, j]}"
    return lines


def _heterogeneity_hover_text(hetero: np.ndarray) -> np.ndarray:
    text = np.empty(hetero.shape, dtype=object)
    text[hetero < 0.5] = _HETERO_HELP[0]
    text[hetero >= 0.5] = _HETERO_HELP[1]
    return text


def format_diagnostic_interpretation(metrics: WorldMetrics) -> str:
    """Human-readable summary for the diagnostic panel (render in Streamlit, not Plotly)."""
    return _interpretation_block(metrics)


def create_diagnostic_dashboard(
    result: SimulationResult,
    *,
    surrogate_pred: dict[str, float] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Composite diagnostic figure (Plotly; panels aligned with legacy matplotlib diagnostic layout)."""
    life = result.final_life
    food = result.final_food
    if life is None or food is None:
        msg = "Diagnostic dashboard needs final_life and final_food from run_world"
        raise ValueError(msg)
    if life.shape != food.shape:
        msg = "life and food shapes must match"
        raise ValueError(msg)
    if result.metrics is None:
        msg = "Diagnostic dashboard needs result.metrics"
        raise ValueError(msg)

    from worldspace import math as ws_math

    metrics = result.metrics
    boundary = ws_math.topology_interface_strength_map(life)
    hetero = ws_math.topology_2x2_heterogeneity_map(life)
    fnb = ws_math.food_neighbor_fraction_map(food)
    base = _life_food_rgb(life, food)

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Life + food · boundary",
            "2×2 heterogeneity",
            "Food neighbor (live cells)",
            "All metrics",
            "",
            "Metrics radar",
        ),
        specs=[
            [{"type": "xy"}, {"type": "heatmap"}, {"type": "xy"}],
            [{"type": "xy", "colspan": 2}, None, {"type": "polar"}],
        ],
        column_widths=[1.15, 1.0, 1.0],
        row_heights=[0.62, 0.38],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    blended_main = _blend_boundary_on_rgb(base, boundary)
    fig.add_trace(_rgb_image_trace(blended_main), row=1, col=1)
    _add_cell_hover_layer(
        fig, _life_food_hover_text(life, food, boundary), row=1, col=1
    )
    hetero_hover = _heterogeneity_hover_text(hetero)
    fig.add_trace(
        go.Heatmap(
            z=hetero,
            text=hetero_hover,
            colorscale=_heterogeneity_colorscale(),
            zmin=0.0,
            zmax=1.0,
            zsmooth=False,
            showscale=False,
            hovertemplate=(
                "<b>2×2 heterogeneity=%{z:.0f}</b><br>%{text}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    blend_adj = _blend_food_neighbor_rgb(life, base, fnb)
    fig.add_trace(_rgb_image_trace(blend_adj), row=1, col=3)
    _add_cell_hover_layer(fig, _food_neighbor_hover_text(life, food, fnb), row=1, col=3)

    _add_panel_legend(
        fig,
        1,
        1,
        _life_food_panel_legend_entries(),
        title="Life + food · boundary",
        anchor="bottom_right",
    )
    _add_panel_legend(
        fig,
        1,
        2,
        _heterogeneity_legend_entries(hetero),
        title="2×2 heterogeneity",
        anchor="bottom_left",
    )
    _add_panel_legend(
        fig,
        1,
        3,
        _food_neighbor_legend_entries(life, food),
        title="Food neighbors",
        anchor="bottom_left",
    )

    names, bar_vals = _metrics_bar_values(metrics)
    bar_colors = ["#4a6fa5"] * 7 + ["#b85c38"] * 5
    bar_help = [metric_help_text(name) for name in names]
    fig.add_trace(
        go.Bar(
            x=bar_vals,
            y=names,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="#222222", width=0.3)),
            customdata=bar_help,
            hovertemplate=(
                "<b>%{y}</b><br>display=%{x:.3f}<br>%{customdata}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.zeros(len(names), dtype=np.float64),
            y=names,
            mode="markers",
            marker=dict(size=16, color="rgba(0,0,0,0)"),
            customdata=bar_help,
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(range=[0.0, 1.05], title_text="display scale (0–1)", row=2, col=1)
    fig.update_yaxes(autorange="reversed", row=2, col=1)

    radar_metrics = {key: float(value) for key, value in asdict(metrics).items()}
    radar = create_metrics_radar(radar_metrics)
    for trace in radar.data:
        if trace.type == "scatterpolar" and trace.theta is not None:
            thetas = (
                list(trace.theta)[:-1] if len(trace.theta) > 1 else list(trace.theta)
            )
            key_lookup = {k.replace("_", " ").title(): k for k in RADAR_METRIC_KEYS}
            hover = [
                metric_help_text(key_lookup.get(str(label), str(label)))
                for label in thetas
            ]
            trace.hovertemplate = (
                "<b>%{theta}</b><br>%{r:.3f}<br>%{text}<extra></extra>"
            )
            trace.text = hover
        fig.add_trace(trace, row=2, col=3)

    for col in (1, 2, 3):
        fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=col)
        fig.update_yaxes(
            showticklabels=False,
            showgrid=False,
            autorange="reversed",
            row=1,
            col=col,
        )

    world = result.world
    header = title
    if header is None and world is not None:
        header = (
            f"Diagnostic — seed={world.seed} grid={world.grid_size} steps={world.steps}"
        )
    if header is None:
        header = "Diagnostic dashboard"

    subtitle_parts = [
        f"mo_eoc={metrics.mo_eoc_indicator:.3f}",
        f"topo_if={metrics.topology_interface_index:.3f}",
        f"hetero2x2={metrics.topology_window_heterogeneity:.3f}",
        f"comp={metrics.compressibility_score:.3f}",
        f"eco_adj={metrics.ecology_resource_adjacency:.3f}",
    ]
    if surrogate_pred:
        fitness = surrogate_pred.get("fitness")
        uncertainty = surrogate_pred.get("uncertainty")
        if fitness is not None:
            line = f"Surrogate fitness: {float(fitness):.3f}"
            if uncertainty is not None:
                line += f" (±{float(uncertainty):.3f})"
            subtitle_parts.append(line)

    fig.update_layout(
        title=dict(
            text=(
                f"{header}<br>"
                f"<sup style='color:#ffffff'>{' · '.join(subtitle_parts)}</sup>"
            ),
            x=0.02,
            xanchor="left",
            font=dict(color="#ffffff", size=16),
        ),
        width=DIAGNOSTIC_FIGURE_WIDTH,
        height=DIAGNOSTIC_FIGURE_HEIGHT,
        showlegend=False,
        margin=dict(l=48, r=24, t=100, b=48),
        hoverlabel=dict(namelength=-1, font_size=11),
    )
    themed = apply_dark_theme(fig)
    themed.update_layout(title=dict(font=dict(color="#ffffff")))
    return themed


def _create_cvt_centroid_scatter(
    fig: go.Figure,
    *,
    collapsed: pd.DataFrame,
    centroids: np.ndarray,
    metric: str,
    title: str,
    colorscale: str,
) -> go.Figure:
    n_cells = centroids.shape[0]
    stability = centroids[:, 0]
    diversity = centroids[:, 1]
    cell_ids = np.arange(n_cells, dtype=np.int64)

    metric_values = np.full(n_cells, np.nan, dtype=np.float64)
    if not collapsed.empty and "cell_id" in collapsed.columns:
        metric_col = _scatter_metric_column(collapsed, metric)
        if metric_col is not None:
            lookup = collapsed.set_index("cell_id")[metric_col]
            for cell_id in lookup.index:
                idx = int(cell_id)
                if 0 <= idx < n_cells:
                    metric_values[idx] = float(lookup.loc[cell_id])

    filled_mask = ~np.isnan(metric_values)
    empty_mask = ~filled_mask

    if np.any(empty_mask):
        fig.add_trace(
            go.Scatter(
                x=diversity[empty_mask],
                y=stability[empty_mask],
                mode="markers",
                name="empty niche",
                marker=dict(
                    size=9,
                    color="rgba(180, 180, 180, 0.35)",
                    symbol="circle-open",
                    line=dict(width=1.5, color="rgba(200, 200, 200, 0.8)"),
                ),
                hovertemplate=(
                    "cell %{customdata}<br>"
                    "stability=%{y:.3f}<br>"
                    "diversity=%{x:.3f}<br>"
                    "empty<extra></extra>"
                ),
                customdata=cell_ids[empty_mask],
            )
        )

    if np.any(filled_mask):
        filled_values = metric_values[filled_mask]
        fig.add_trace(
            go.Scatter(
                x=diversity[filled_mask],
                y=stability[filled_mask],
                mode="markers",
                name="elite",
                marker=dict(
                    size=11,
                    color=filled_values,
                    colorscale=colorscale,
                    colorbar=_scatter_metric_colorbar(metric),
                    cmin=float(np.nanmin(filled_values)),
                    cmax=float(np.nanmax(filled_values)),
                    line=dict(width=0.5, color="rgba(255, 255, 255, 0.35)"),
                ),
                hovertemplate=(
                    "cell %{customdata}<br>"
                    "stability=%{y:.3f}<br>"
                    "diversity=%{x:.3f}<br>"
                    f"{metric}=%{{marker.color:.3f}}<extra></extra>"
                ),
                customdata=cell_ids[filled_mask],
            )
        )

    if not np.any(filled_mask) and not np.any(empty_mask):
        fig.add_annotation(
            text="No CVT niches to display.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )

    fig.update_layout(
        title=title,
        height=default_figure_height(heatmap=True),
        width=680,
        margin=dict(r=110),
        xaxis=dict(title="Diversity (niche center)", range=[0.0, 1.0]),
        yaxis=dict(title="Stability (niche center)", range=[0.0, 1.0]),
    )
    return apply_dark_theme(fig)


def _create_cvt_elite_only_scatter(
    fig: go.Figure,
    *,
    collapsed: pd.DataFrame,
    metric: str,
    title: str,
    colorscale: str,
) -> go.Figure:
    if collapsed.empty:
        fig.add_annotation(
            text="No elites to display (centroids file missing).",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(height=default_figure_height(heatmap=True), width=680)
        return apply_dark_theme(fig)

    x_col = _scatter_axis_column(collapsed, "diversity")
    y_col = _scatter_axis_column(collapsed, "stability")
    metric_col = _scatter_metric_column(collapsed, metric)
    if x_col is None or y_col is None or metric_col is None:
        fig.add_annotation(
            text="Missing stability/diversity columns for degraded CVT scatter.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(height=default_figure_height(heatmap=True), width=680)
        return apply_dark_theme(fig)

    x_values = collapsed[x_col].to_numpy(dtype=np.float64)
    y_values = collapsed[y_col].to_numpy(dtype=np.float64)
    color_values = collapsed[metric_col].to_numpy(dtype=np.float64)
    customdata = (
        collapsed["cell_id"].to_numpy(dtype=np.int64)
        if "cell_id" in collapsed.columns
        else np.arange(len(collapsed), dtype=np.int64)
    )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers",
            name="elite",
            marker=dict(
                size=11,
                color=color_values,
                colorscale=colorscale,
                colorbar=_scatter_metric_colorbar(metric),
                cmin=float(np.nanmin(color_values)),
                cmax=float(np.nanmax(color_values)),
            ),
            hovertemplate=(
                "cell %{customdata}<br>"
                "stability=%{y:.3f}<br>"
                "diversity=%{x:.3f}<br>"
                f"{metric}=%{{marker.color:.3f}}<extra></extra>"
            ),
            customdata=customdata,
        )
    )
    fig.update_layout(
        title=title,
        height=default_figure_height(heatmap=True),
        width=680,
        margin=dict(r=110),
        xaxis=dict(title="Diversity", range=[0.0, 1.0]),
        yaxis=dict(title="Stability", range=[0.0, 1.0]),
    )
    return apply_dark_theme(fig)


def _scatter_metric_colorbar(metric: str) -> dict[str, Any]:
    """Colorbar layout for CVT archive scatter plots."""
    return dict(
        title=metric,
        x=1.12,
        xanchor="left",
        xpad=12,
    )


def _scatter_metric_column(frame: pd.DataFrame, metric: str) -> str | None:
    if metric in frame.columns:
        return metric
    measure_key = f"measure_{metric}"
    if measure_key in frame.columns:
        return measure_key
    return None


def _scatter_axis_column(frame: pd.DataFrame, axis: str) -> str | None:
    if axis in frame.columns:
        return axis
    measure_key = f"measure_{axis}"
    if measure_key in frame.columns:
        return measure_key
    if axis == "stability" and "centroid_s" in frame.columns:
        return "centroid_s"
    if axis == "diversity" and "centroid_d" in frame.columns:
        return "centroid_d"
    return None


def _resolve_pivot(
    pivot: np.ndarray | None,
    *,
    df: pd.DataFrame | None,
    metric: str,
    resolution: int,
) -> np.ndarray:
    if pivot is not None:
        grid = np.asarray(pivot, dtype=np.float64)
        if grid.shape != (resolution, resolution):
            msg = f"pivot shape {grid.shape} != ({resolution}, {resolution})"
            raise ValueError(msg)
        return grid
    if df is not None:
        return _pivot_from_dataframe(df, metric=metric, resolution=resolution)
    msg = "create_archive_heatmap requires pivot or df"
    raise ValueError(msg)


def _pivot_from_dataframe(
    df: pd.DataFrame,
    *,
    metric: str,
    resolution: int,
) -> np.ndarray:
    grid = np.full((resolution, resolution), np.nan, dtype=np.float64)
    if df.empty or metric not in df.columns:
        return grid
    if "bin_x" not in df.columns or "bin_y" not in df.columns:
        msg = "df must include bin_x and bin_y columns"
        raise ValueError(msg)
    valid = df[metric].notna()
    bins_x = df.loc[valid, "bin_x"].to_numpy(dtype=np.int64)
    bins_y = df.loc[valid, "bin_y"].to_numpy(dtype=np.int64)
    values = df.loc[valid, metric].to_numpy(dtype=np.float64)
    for i, j, value in zip(bins_x, bins_y, values, strict=True):
        if 0 <= i < resolution and 0 <= j < resolution:
            current = grid[i, j]
            if np.isnan(current) or value > current:
                grid[i, j] = value
    return grid


def _clip_unit(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _rgb_image_trace(rgb: np.ndarray) -> go.Image:
    """Plotly ``Image`` trace from float RGB in [0, 1]."""
    z = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return go.Image(z=z)


def _blend_boundary_on_rgb(base: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    warm = np.stack([boundary, boundary * 0.42, boundary * 0.12], axis=-1)
    return np.clip(base + warm * 0.42, 0.0, 1.0)


def _blend_food_neighbor_rgb(
    life: np.ndarray,
    base: np.ndarray,
    fnb: np.ndarray,
) -> np.ndarray:
    live = life > 0.5
    t = fnb.astype(np.float32)
    warm = np.clip(
        np.stack(
            [0.18 + 0.75 * t, 0.32 + 0.38 * (1.0 - t), 0.48 + 0.35 * (1.0 - t)],
            axis=-1,
        ),
        0.0,
        1.0,
    )
    blend = _FOOD_NEIGHBOR_BLEND
    return np.where(live[..., None], (1.0 - blend) * base + blend * warm, base)


def _metrics_bar_values(metrics: WorldMetrics) -> tuple[list[str], np.ndarray]:
    values = asdict(metrics)
    names: list[str] = [str(key) for key in METRIC_KEYS]
    raw = np.array([float(values[key]) for key in names], dtype=np.float64)
    scaled = raw.copy()
    j_life = names.index("average_lifespan")
    scaled[j_life] = float(np.clip(scaled[j_life] / 10.0, 0.0, 1.0))
    j_mo = names.index("mo_eoc_indicator")
    scaled[j_mo] = float(np.clip(scaled[j_mo] / 3.0, 0.0, 1.0))
    return names, np.clip(scaled, 0.0, 1.0)


def _interpretation_block(metrics: WorldMetrics) -> str:
    lines: list[str] = []
    if (
        metrics.ecology_resource_adjacency > 0.35
        and metrics.topology_interface_index > 0.25
    ):
        lines.append(
            "Strong resource coupling with a complex life boundary — "
            "a potentially stable ecosystem (resource flow to consumers in a fragmented patch)."
        )
    elif metrics.ecology_resource_adjacency < 0.12 and metrics.density_mean > 0.15:
        lines.append(
            "Life is present, but food rarely neighbors live cells — possible weak trophic linkage."
        )
    if metrics.topology_window_heterogeneity > 0.45:
        lines.append(
            'Many locally "mixed" 2×2 windows — mesoscale heterogeneity (edges / pattern blending).'
        )
    elif metrics.topology_window_heterogeneity < 0.08:
        lines.append("Nearly uniform 2×2 windows — large-scale smooth or empty field.")
    if metrics.compressibility_score > 0.55:
        lines.append(
            'High compressibility — configuration is close to a "short description" (substantial order).'
        )
    elif metrics.compressibility_score < 0.15:
        lines.append(
            "Low compressibility — closer to noise or fine-grained non-repeating structure."
        )
    if metrics.ecology_state_entropy_norm > 0.75:
        lines.append(
            "High entropy of the joint (life, food) field — rich set of local ecological micro-states."
        )
    if not lines:
        lines.append(
            "Summary: use the metric bars on the left; the boundary overlay and 2×2 heatmap are "
            "topological proxies, not Betti numbers."
        )
    return "\n\n".join(lines)


def _rgb_float_to_css(rgb: np.ndarray) -> str:
    """Convert float RGB in [0, 1] to a Plotly/CSS ``rgb(...)`` string."""
    clipped = np.clip(rgb, 0.0, 1.0)
    u8 = (clipped * 255.0).astype(np.uint8)
    return f"rgb({int(u8[0])}, {int(u8[1])}, {int(u8[2])})"


def _boundary_tint_at_max(base: np.ndarray) -> np.ndarray:
    """Additive boundary warm layer at strength 1 (matches ``_blend_boundary_on_rgb``)."""
    return np.clip(base + _BOUNDARY_WARM_MAX, 0.0, 1.0)


def _life_food_panel_legend_entries() -> list[tuple[str, str]]:
    """Base life/food colors plus max-strength boundary tint on each base type."""
    return [
        ("Empty", _rgb_float_to_css(_LIFE_FOOD_EMPTY)),
        ("Life only", _rgb_float_to_css(_LIFE_FOOD_LIFE_ONLY)),
        ("Food only", _rgb_float_to_css(_LIFE_FOOD_FOOD_ONLY)),
        ("Life + food", _rgb_float_to_css(_LIFE_FOOD_BOTH)),
        ("Edge on empty", _rgb_float_to_css(_boundary_tint_at_max(_LIFE_FOOD_EMPTY))),
        (
            "Edge on life only",
            _rgb_float_to_css(_boundary_tint_at_max(_LIFE_FOOD_LIFE_ONLY)),
        ),
        (
            "Edge on food only",
            _rgb_float_to_css(_boundary_tint_at_max(_LIFE_FOOD_FOOD_ONLY)),
        ),
        (
            "Edge on life + food",
            _rgb_float_to_css(_boundary_tint_at_max(_LIFE_FOOD_BOTH)),
        ),
    ]


def _heterogeneity_colorscale() -> list[list[float | str]]:
    """Discrete Magma stops for binary 0/1 heterogeneity (no mid-scale interpolation)."""
    from plotly.colors import sample_colorscale

    low = str(sample_colorscale("Magma", [0.0])[0])
    high = str(sample_colorscale("Magma", [1.0])[0])
    return [[0.0, low], [1.0, high]]


def _heterogeneity_legend_entries(hetero: np.ndarray) -> list[tuple[str, str]]:
    """One swatch per value present in ``hetero`` (typically 0 and/or 1)."""
    from plotly.colors import sample_colorscale

    scale = _heterogeneity_colorscale()
    color_at = {float(stop[0]): str(stop[1]) for stop in scale}
    present = sorted(
        {float(v) for v in np.unique(hetero) if np.isfinite(v)},
        key=lambda x: x,
    )
    if not present:
        present = [0.0, 1.0]
    entries: list[tuple[str, str]] = []
    for value in present:
        label = _HETERO_LABELS.get(value, f"hetero={value:.2f}")
        color = color_at.get(value)
        if color is None:
            norm = float(np.clip(value, 0.0, 1.0))
            color = str(sample_colorscale("Magma", [norm])[0])
        entries.append((label, color))
    return entries


def _food_neighbor_warm(t: float) -> np.ndarray:
    return np.clip(
        np.array(
            [0.18 + 0.75 * t, 0.32 + 0.38 * (1.0 - t), 0.48 + 0.35 * (1.0 - t)],
            dtype=np.float32,
        ),
        0.0,
        1.0,
    )


def _blend_food_neighbor_on_base(base_rgb: np.ndarray, t: float) -> np.ndarray:
    weight = 1.0 - _FOOD_NEIGHBOR_BLEND
    warm = _food_neighbor_warm(t)
    return np.clip(weight * base_rgb + _FOOD_NEIGHBOR_BLEND * warm, 0.0, 1.0)


def _food_neighbor_legend_entries(
    life: np.ndarray,
    food: np.ndarray,
) -> list[tuple[str, str]]:
    """Swatches for non-live base colors and live-cell neighbor tint (per base type)."""
    life_f = life.astype(np.float32)
    food_f = food.astype(np.float32)
    entries: list[tuple[str, str]] = []

    dead = life_f < 0.5
    if np.any(dead & (food_f < 0.5)):
        entries.append(("Empty", _rgb_float_to_css(_LIFE_FOOD_EMPTY)))
    if np.any(dead & (food_f >= 0.5)):
        entries.append(("Food only", _rgb_float_to_css(_LIFE_FOOD_FOOD_ONLY)))

    live = life_f >= 0.5
    live_bases: list[tuple[str, np.ndarray]] = []
    if np.any(live & (food_f < 0.5)):
        live_bases.append(("Life only", _LIFE_FOOD_LIFE_ONLY))
    if np.any(live & (food_f >= 0.5)):
        live_bases.append(("Life + food", _LIFE_FOOD_BOTH))

    tint_steps = ((0.0, "few neighbors"), (0.5, "mid"), (1.0, "many neighbors"))
    for base_name, base_rgb in live_bases:
        for t, tint in tint_steps:
            color = _blend_food_neighbor_on_base(base_rgb, t)
            entries.append((f"{base_name} · {tint}", _rgb_float_to_css(color)))

    if not entries:
        entries.append(("Empty", _rgb_float_to_css(_LIFE_FOOD_EMPTY)))
    return entries


def _life_food_rgb(life: np.ndarray, food: np.ndarray) -> np.ndarray:
    """RGB base field for life/food grids (matches legacy diagnostic palette)."""
    life_f = life.astype(np.float32)
    food_f = food.astype(np.float32)
    rgb = np.zeros((*life.shape, 3), dtype=np.float32)
    empty = (life_f < 0.5) & (food_f < 0.5)
    life_only = (life_f >= 0.5) & (food_f < 0.5)
    food_only = (life_f < 0.5) & (food_f >= 0.5)
    both = (life_f >= 0.5) & (food_f >= 0.5)
    rgb[empty] = _LIFE_FOOD_EMPTY
    rgb[life_only] = _LIFE_FOOD_LIFE_ONLY
    rgb[food_only] = _LIFE_FOOD_FOOD_ONLY
    rgb[both] = _LIFE_FOOD_BOTH
    return np.clip(rgb, 0.0, 1.0)

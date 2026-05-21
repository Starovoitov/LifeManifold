"""Plotly visualization builders for the LifeManifold dashboard (no Streamlit)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

__all__ = [
    "BOUNDARY_COLORSCALE",
    "METRIC_COLORSCALES",
    "RADAR_METRIC_KEYS",
    "add_boundary_overlay",
    "create_archive_heatmap",
    "create_diagnostic_dashboard",
    "create_metrics_radar",
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
        marker_kwargs["colorbar"] = dict(title="uncertainty")

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
    )
    return apply_dark_theme(fig)


def create_diagnostic_dashboard(
    result: SimulationResult,
    *,
    surrogate_pred: dict[str, float] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Composite diagnostic figure aligned with ``worldspace.visualizer.diagnostics``."""
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
            "Interpretation",
            "Metrics radar",
        ),
        specs=[
            [{"type": "xy"}, {"type": "heatmap"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "polar"}],
        ],
        column_widths=[1.15, 1.0, 1.0],
        row_heights=[0.65, 0.35],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    blended_main = _blend_boundary_on_rgb(base, boundary)
    fig.add_trace(_rgb_image_trace(blended_main), row=1, col=1)
    fig.add_trace(
        go.Heatmap(
            z=hetero,
            colorscale="Magma",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="hetero", len=0.4, y=0.82),
            hovertemplate="hetero=%{z:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    blend_adj = _blend_food_neighbor_rgb(life, base, fnb)
    fig.add_trace(_rgb_image_trace(blend_adj), row=1, col=3)

    names, bar_vals = _metrics_bar_values(metrics)
    bar_colors = ["#4a6fa5"] * 7 + ["#b85c38"] * 5
    fig.add_trace(
        go.Bar(
            x=bar_vals,
            y=names,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="#222222", width=0.3)),
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(range=[0.0, 1.05], title_text="display scale (0–1)", row=2, col=1)
    fig.update_yaxes(autorange="reversed", row=2, col=1)

    interpretation = _interpretation_block(metrics)
    fig.add_trace(
        go.Scatter(
            x=[0.0],
            y=[1.0],
            mode="text",
            text=[interpretation.replace("\n\n", "<br><br>")],
            textposition="top left",
            textfont=dict(size=11),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=2,
    )
    fig.update_xaxes(visible=False, range=[0, 1], row=2, col=2)
    fig.update_yaxes(visible=False, range=[0, 1], row=2, col=2)

    radar_metrics = {key: float(value) for key, value in asdict(metrics).items()}
    radar = create_metrics_radar(radar_metrics)
    for trace in radar.data:
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
            text=f"{header}<br><sup>{' · '.join(subtitle_parts)}</sup>",
            x=0.02,
            xanchor="left",
        ),
        height=880,
        showlegend=False,
        margin=dict(l=48, r=48, t=100, b=48),
    )
    return apply_dark_theme(fig)


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
    return np.where(live[..., None], (1.0 - 0.48) * base + 0.48 * warm, base)


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


def _life_food_rgb(life: np.ndarray, food: np.ndarray) -> np.ndarray:
    """RGB base field for life/food grids (matches legacy diagnostic palette)."""
    life_f = life.astype(np.float32)
    food_f = food.astype(np.float32)
    rgb = np.zeros((*life.shape, 3), dtype=np.float32)
    empty = (life_f < 0.5) & (food_f < 0.5)
    life_only = (life_f >= 0.5) & (food_f < 0.5)
    food_only = (life_f < 0.5) & (food_f >= 0.5)
    both = (life_f >= 0.5) & (food_f >= 0.5)
    rgb[empty] = np.array([0.10, 0.11, 0.15], dtype=np.float32)
    rgb[life_only] = np.array([0.29, 0.61, 0.56], dtype=np.float32)
    rgb[food_only] = np.array([0.79, 0.64, 0.15], dtype=np.float32)
    rgb[both] = np.array([0.43, 0.48, 0.31], dtype=np.float32)
    return np.clip(rgb, 0.0, 1.0)

"""Single-world diagnostic dashboard and metric tertile galleries."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .. import math as ws_math
from ..generators import RandomWorldGenerator
from ..metrics import METRIC_KEYS, WorldMetrics
from ..simulator import SimulationResult, run_world
from ..specs.spec import WorldSpec

# --- public module API (constants) ---

GALLERY_NEW_METRICS = (
    "topology_interface_index",
    "topology_window_heterogeneity",
    "compressibility_score",
    "ecology_state_entropy_norm",
    "ecology_resource_adjacency",
)

# Defaults for ``python -m worldspace.visualizer`` (no separate gallery CLI flags).
VISUALIZER_GALLERY_SCAN_SEEDS = 56
VISUALIZER_GALLERY_GRID_SIZE = 22
VISUALIZER_GALLERY_STEPS = 72
VISUALIZER_GALLERY_SEED_OFFSET = 0

DEFAULT_DIAGNOSTIC_DASHBOARD_GRID = 28
DEFAULT_DIAGNOSTIC_DASHBOARD_STEPS = 120
DEFAULT_DIAGNOSTIC_DASHBOARD_SEED = 42


def plot_diagnostic_dashboard(
    result: SimulationResult,
    path: str | Path,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (14.0, 10.0),
    dpi: int = 120,
) -> None:
    """
    One composite figure per world: life+food field, boundary overlay, 2×2 heterogeneity map,
    food-neighbor shading on live cells, bar chart of all metrics, interpretation text.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    life = result.final_life
    food = result.final_food
    if life is None or food is None:
        raise ValueError(
            "Diagnostic dashboard needs ``final_life`` and ``final_food`` from ``run_world``."
        )
    if life.shape != food.shape:
        raise ValueError("life and food shapes must match.")

    m = result.metrics
    boundary = ws_math.topology_interface_strength_map(life)
    hetero = ws_math.topology_2x2_heterogeneity_map(life)
    fnb = ws_math.food_neighbor_fraction_map(food)

    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=False)
    gs = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[2.1, 1.0],
        width_ratios=[1.15, 1.0, 1.0],
        hspace=0.28,
        wspace=0.25,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_hetero = fig.add_subplot(gs[0, 1])
    ax_adj = fig.add_subplot(gs[0, 2])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_txt = fig.add_subplot(gs[1, 1:])
    for ax in (ax_txt,):
        ax.set_axis_off()

    base = _life_food_rgb(life, food)
    warm = np.stack([boundary, boundary * 0.42, boundary * 0.12], axis=-1)
    blended = np.clip(base + warm * 0.42, 0.0, 1.0)
    ax_main.imshow(blended, origin="lower", interpolation="nearest")
    ax_main.set_title("Life + food (base) · boundary strength overlay")
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    imh = ax_hetero.imshow(
        hetero,
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax_hetero.set_title("2×2 local heterogeneity (toroidal windows)")
    ax_hetero.set_xticks([])
    ax_hetero.set_yticks([])
    plt.colorbar(imh, ax=ax_hetero, fraction=0.046, pad=0.02)

    live = life > 0.5
    t = fnb.astype(np.float32)
    warm = np.clip(
        np.stack(
            [0.18 + 0.75 * t, 0.32 + 0.38 * (1.0 - t), 0.48 + 0.35 * (1.0 - t)], axis=-1
        ),
        0.0,
        1.0,
    )
    blend_adj = np.where(live[..., None], (1.0 - 0.48) * base + 0.48 * warm, base)
    ax_adj.imshow(blend_adj, origin="lower", interpolation="nearest")
    ax_adj.set_title("Live cells: food in 8-neighborhood (cool→warm)")
    ax_adj.set_xticks([])
    ax_adj.set_yticks([])

    names, vals = _metrics_bar_values(m)
    colors = ["#4a6fa5"] * 7 + ["#b85c38"] * 5
    y = np.arange(len(names))
    ax_bar.barh(y, vals, color=colors, edgecolor="#222", linewidth=0.3)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(names, fontsize=7)
    ax_bar.set_xlim(0.0, 1.05)
    ax_bar.set_xlabel("display scale (0–1)")
    ax_bar.set_title("All metrics (blue: core / orange: topology & ecology)")
    ax_bar.invert_yaxis()

    w = result.world
    hdr = title or (
        f"Diagnostic — seed={w.seed} grid={w.grid_size} steps={w.steps}\n"
        f"mo_eoc={m.mo_eoc_indicator:.3f}  topo_if={m.topology_interface_index:.3f}  "
        f"hetero2x2={m.topology_window_heterogeneity:.3f}  comp={m.compressibility_score:.3f}  "
        f"eco_adj={m.ecology_resource_adjacency:.3f}"
    )
    fig.suptitle(hdr, fontsize=11, y=0.98)
    ax_txt.text(
        0.02,
        0.96,
        _interpretation_block(m),
        transform=ax_txt.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        linespacing=1.35,
    )
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)


def plot_metric_tertile_gallery(
    metric_key: str,
    path: str | Path,
    *,
    scan_seeds: int = 400,
    grid_size: int = 28,
    steps: int = 120,
    seed_offset: int = 0,
    figsize: tuple[float, float] = (10.0, 9.0),
    dpi: int = 110,
) -> None:
    """
    3×3 thumbnails: columns = low (~0.15) / mid (~0.5) / high (~0.85) target for ``metric_key``;
    rows = three example worlds per band.
    """
    if metric_key not in METRIC_KEYS:
        raise ValueError(
            f"Unknown metric_key {metric_key!r}; expected one of {METRIC_KEYS}."
        )
    if scan_seeds < 1:
        raise ValueError("scan_seeds must be >= 1 for tertile gallery.")
    low, mid, high = _pick_tertile_examples(
        metric_key,
        scan_seeds=scan_seeds,
        grid_size=grid_size,
        steps=steps,
        seed_offset=seed_offset,
    )
    fig, axes = plt.subplots(3, 3, figsize=figsize, dpi=dpi, squeeze=False)
    cols = [("Low ≈0.15", low), ("Medium ≈0.5", mid), ("High ≈0.85", high)]
    for j, (col_title, col_data) in enumerate(cols):
        for i in range(3):
            ax = axes[i, j]
            w, r, v = col_data[i]
            life = r.final_life
            food = r.final_food
            assert life is not None and food is not None
            _thumb_axis(ax, life, food, v, f"{col_title}  seed={w.seed}")
    fig.suptitle(
        f"Gallery: {metric_key}\n(random worlds, grid={grid_size} steps={steps}, scan={scan_seeds})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --- private helpers ---


def _life_food_rgb(life: np.ndarray, food: np.ndarray) -> np.ndarray:
    """RGBA-ish RGB base field: empty / life / food / both — muted, low-glare palette."""
    life = life.astype(np.float32)
    food = food.astype(np.float32)
    h, w = life.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    empty = (life < 0.5) & (food < 0.5)
    l_only = (life >= 0.5) & (food < 0.5)
    f_only = (life < 0.5) & (food >= 0.5)
    both = (life >= 0.5) & (food >= 0.5)
    # #1a1d26 empty, #4a9b8e life, #c9a227 food, #6d7a50 both
    rgb[empty] = np.array([0.10, 0.11, 0.15], dtype=np.float32)
    rgb[l_only] = np.array([0.29, 0.61, 0.56], dtype=np.float32)
    rgb[f_only] = np.array([0.79, 0.64, 0.15], dtype=np.float32)
    rgb[both] = np.array([0.43, 0.48, 0.31], dtype=np.float32)
    return np.clip(rgb, 0.0, 1.0)


def _metrics_bar_values(metrics: WorldMetrics) -> tuple[list[str], np.ndarray]:
    """Scalar names and values scaled to ~[0, 1] for bar readability."""
    d = asdict(metrics)
    names: list[str] = list(METRIC_KEYS)
    raw = np.array([float(d[k]) for k in names], dtype=np.float64)
    scaled = raw.copy()
    # Rough display scaling (not used for scoring)
    j_life = names.index("average_lifespan")
    scaled[j_life] = float(np.clip(scaled[j_life] / 10.0, 0.0, 1.0))
    j_mo = names.index("mo_eoc_indicator")
    scaled[j_mo] = float(np.clip(scaled[j_mo] / 3.0, 0.0, 1.0))
    scaled = np.clip(scaled, 0.0, 1.0)
    return names, scaled


def _interpretation_block(m: WorldMetrics) -> str:
    lines: list[str] = []
    if m.ecology_resource_adjacency > 0.35 and m.topology_interface_index > 0.25:
        lines.append(
            "Strong resource coupling with a complex life boundary — "
            "a potentially stable ecosystem (resource flow to consumers in a fragmented patch)."
        )
    elif m.ecology_resource_adjacency < 0.12 and m.density_mean > 0.15:
        lines.append(
            "Life is present, but food rarely neighbors live cells — possible weak trophic linkage."
        )
    if m.topology_window_heterogeneity > 0.45:
        lines.append(
            'Many locally "mixed" 2×2 windows — mesoscale heterogeneity (edges / pattern blending).'
        )
    elif m.topology_window_heterogeneity < 0.08:
        lines.append("Nearly uniform 2×2 windows — large-scale smooth or empty field.")
    if m.compressibility_score > 0.55:
        lines.append(
            'High compressibility — configuration is close to a "short description" (substantial order).'
        )
    elif m.compressibility_score < 0.15:
        lines.append(
            "Low compressibility — closer to noise or fine-grained non-repeating structure."
        )
    if m.ecology_state_entropy_norm > 0.75:
        lines.append(
            "High entropy of the joint (life, food) field — rich set of local ecological micro-states."
        )
    if not lines:
        lines.append(
            "Summary: use the metric bars on the right; the boundary (warm overlay) and 2×2 heatmap are "
            "topological proxies, not Betti numbers."
        )
    return "\n\n".join(lines)


def _pick_tertile_examples(
    metric_key: str,
    *,
    scan_seeds: int,
    grid_size: int,
    steps: int,
    seed_offset: int,
) -> tuple[
    list[tuple[WorldSpec, SimulationResult, float]],
    list[tuple[WorldSpec, SimulationResult, float]],
    list[tuple[WorldSpec, SimulationResult, float]],
]:
    """Return three lists (low / mid / high) of (world, result, metric_value) length 3 each."""
    gen = RandomWorldGenerator(grid_size=grid_size, steps=steps)
    scored: list[tuple[WorldSpec, SimulationResult, float]] = []
    for s in range(seed_offset, seed_offset + scan_seeds):
        w = gen._make_world(seed=s)
        r = run_world(w)
        v = float(getattr(r.metrics, metric_key))
        scored.append((w, r, v))

    def pick_near(target: float) -> list[tuple[WorldSpec, SimulationResult, float]]:
        ranked = sorted(scored, key=lambda t: abs(t[2] - target))
        out: list[tuple[WorldSpec, SimulationResult, float]] = []
        seen: set[int] = set()
        for item in ranked:
            if item[0].seed in seen:
                continue
            seen.add(item[0].seed)
            out.append(item)
            if len(out) >= 3:
                return _pad_example_triplet(out)
        # One linear pass over ``scored`` (generation order); every seed is unique → terminates.
        for item in scored:
            if len(out) >= 3:
                break
            if item[0].seed in seen:
                continue
            seen.add(item[0].seed)
            out.append(item)
        return _pad_example_triplet(out)

    return pick_near(0.15), pick_near(0.5), pick_near(0.85)


def _pad_example_triplet(
    out: list[tuple[WorldSpec, SimulationResult, float]],
) -> list[tuple[WorldSpec, SimulationResult, float]]:
    """Gallery axes expect three rows; repeat the last available world if ``scan_seeds`` is small."""
    if not out:
        raise ValueError("No worlds scored; cannot build tertile gallery.")
    while len(out) < 3:
        out.append(out[-1])
    return out[:3]


def _thumb_axis(
    ax: plt.Axes, life: np.ndarray, food: np.ndarray, value: float, subtitle: str
) -> None:
    ax.imshow(_life_food_rgb(life, food), origin="lower", interpolation="nearest")
    ax.set_title(f"{subtitle}\n{value:.3f}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

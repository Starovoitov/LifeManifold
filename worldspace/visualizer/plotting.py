"""Matplotlib figures for world-space (embedding scatter, CA grids, CA-step traces).

Scatter ``embedding_2d`` matches ``pipeline.stream_world_space_to_jsonl``: **x** is the
dominant (highest-variance) metric minus batch mean; **y** is sklearn's first PC on the
other six metrics (sklearn centers those columns once in ``fit``/``transform``) —
not a 7D PCA decomposition and not PC2 of a full PCA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..metrics import METRIC_KEYS
from ..simulator import SimulationResult

if TYPE_CHECKING:
    import pandas as pd


def plot_world_embedding(
    points: list,
    path: str | Path,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 120,
) -> None:
    """Save a 2D scatter of metric embedding; ``points`` items need ``embedding_2d`` and ``cluster_id``."""
    xs = np.array([p.embedding_2d[0] for p in points], dtype=float)
    ys = np.array([p.embedding_2d[1] for p in points], dtype=float)
    clusters = np.array([p.cluster_id for p in points], dtype=int)
    x_label, y_label = _axis_labels_from_points(points)
    _scatter_embedding(
        xs,
        ys,
        clusters,
        path,
        title=title,
        figsize=figsize,
        dpi=dpi,
        x_label=x_label,
        y_label=y_label,
    )


def plot_world_embedding_from_jsonl(
    jsonl_path: str | Path,
    path: str | Path,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 120,
) -> None:
    """Read a metrics JSONL file (one JSON object per line) and plot embedding scatter."""
    xs: list[float] = []
    ys: list[float] = []
    cs: list[int] = []
    x_label = "Δ metric (unknown)"
    y_label = "PC1 of 6 metrics (excluding selected metric)"
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            emb = row["embedding_2d"]
            xs.append(float(emb[0]))
            ys.append(float(emb[1]))
            cs.append(int(row["cluster_id"]))
            axes = row.get("embedding_axes")
            if isinstance(axes, dict):
                x_metric = axes.get("x_metric")
                if x_metric:
                    x_label = f"Δ {x_metric}"
                    y_label = f"PC1 of 6 metrics (excluding {x_metric})"
                else:
                    x_label = str(axes.get("x_label", x_label))
                    y_label = str(axes.get("y_label", y_label))
    if not xs:
        _scatter_embedding(
            np.array([]),
            np.array([]),
            np.array([], dtype=int),
            path,
            title=title,
            figsize=figsize,
            dpi=dpi,
            x_label=x_label,
            y_label=y_label,
        )
        return
    _scatter_embedding(
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(cs, dtype=int),
        path,
        title=title,
        figsize=figsize,
        dpi=dpi,
        x_label=x_label,
        y_label=y_label,
    )


def plot_simulation_final_grid(
    result: SimulationResult,
    path: str | Path,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (6, 6),
    dpi: int = 120,
) -> None:
    """Save the last life grid of a simulation as a binary heatmap."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    grid = result.final_life
    if grid is None or grid.size == 0:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "no grid", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(grid, cmap="Greys_r", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        title
        or f"Final life grid (seed={result.world.seed}, steps={result.world.steps})"
    )
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)


def _scatter_embedding(
    xs: np.ndarray,
    ys: np.ndarray,
    clusters: np.ndarray,
    path: str | Path,
    *,
    title: str | None,
    figsize: tuple[float, float],
    dpi: int,
    x_label: str = "Δ metric (unknown)",
    y_label: str = "PC1 of 6 metrics (excluding selected metric)",
) -> None:
    """Render scatter of ``xs``/``ys`` colored by ``clusters``; write empty-state figure if no points."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if xs.size == 0:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "no points", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
        return

    cmap = plt.get_cmap("tab10")
    uniq = sorted(set(clusters.tolist()))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for idx, cid in enumerate(uniq):
        mask = clusters == cid
        color = cmap(idx % 10)
        ax.scatter(
            xs[mask],
            ys[mask],
            c=[color],
            label=f"cluster {cid}",
            alpha=0.9,
            edgecolors="k",
            linewidths=0.35,
            s=40,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(
        title
        or "World space: Δ selected metric vs PC₁ of other metrics (k-means color)"
    )
    ax.legend(title="k-means", loc="best", fontsize="small")
    ax.grid(True, alpha=0.25)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)


def _axis_labels_from_points(points: list) -> tuple[str, str]:
    """Try to derive axis labels from point metadata; fallback to generic labels."""
    default_x = "Δ metric (unknown)"
    default_y = "PC1 of 6 metrics (excluding selected metric)"
    if not points:
        return default_x, default_y

    first = points[0]
    axes = getattr(first, "embedding_axes", None)
    if isinstance(axes, dict):
        x_metric = axes.get("x_metric")
        if x_metric:
            return f"Δ {x_metric}", f"PC1 of 6 metrics (excluding {x_metric})"
        return (
            str(axes.get("x_label", default_x)),
            str(axes.get("y_label", default_y)),
        )

    x_metric = getattr(first, "x_metric", None)
    if x_metric:
        return f"Δ {x_metric}", f"PC1 of 6 metrics (excluding {x_metric})"
    return default_x, default_y


def load_ca_step_trace_jsonl(path: str | Path) -> "pd.DataFrame":
    """Load ``--ca-step-trace`` JSONL into a flat pandas table (one row per CA step)."""
    import pandas as pd

    p = Path(path)
    df = pd.read_json(p, lines=True)
    if df.empty:
        return df
    if "metrics" not in df.columns:
        raise ValueError("CA step trace JSONL must contain a 'metrics' object per line")
    met = pd.json_normalize(df["metrics"])
    base = df[["yield_index", "ca_step"]].reset_index(drop=True)
    return pd.concat([base, met], axis=1)


def summarize_ca_step_trace_by_world(df: "pd.DataFrame") -> "pd.DataFrame":
    """Per ``yield_index``, aggregate each metric across ``ca_step`` (mean/std/min/max)."""
    import pandas as pd

    cols = [c for c in METRIC_KEYS if c in df.columns]
    if not cols:
        return pd.DataFrame()
    g = df.groupby("yield_index", sort=True)[cols]
    return g.agg(["mean", "std", "min", "max"])


def plot_ca_step_metrics_timeseries(
    df: "pd.DataFrame",
    yield_indices: list[int],
    path: str | Path,
    *,
    metric_names: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10.0, 8.0),
    dpi: int = 120,
) -> None:
    """Line plots: ``ca_step`` vs selected metrics, one subplot per metric (selected worlds)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    metrics = metric_names or ["interestingness", "entropy", "density_mean"]
    metrics = [m for m in metrics if m in df.columns]
    present = set(int(x) for x in df["yield_index"].unique())
    yids = [int(y) for y in yield_indices if int(y) in present]
    if not metrics or not yids:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
        ax.text(0.5, 0.5, "no data for plot", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
        return

    n = len(metrics)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ax, m in zip(np.ravel(axes), metrics):
        for j, yid in enumerate(yids):
            seg = df[df["yield_index"] == yid].sort_values("ca_step")
            if seg.empty:
                continue
            c = cmap(j % 10)
            ax.plot(
                seg["ca_step"],
                seg[m],
                color=c,
                label=f"yield {yid}",
                linewidth=1.2,
                marker="o",
                markersize=2,
                alpha=0.85,
            )
        ax.set_xlabel("ca_step")
        ax.set_ylabel(m)
        ax.set_title(m)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize="x-small", loc="best")
    for k in range(len(metrics), nrows * ncols):
        np.ravel(axes)[k].set_axis_off()
    fig.suptitle(title or "CA metrics over simulation steps")
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)


def plot_ca_step_pca_trajectories(
    df: "pd.DataFrame",
    yield_indices: list[int],
    path: str | Path,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 120,
) -> None:
    """Fit 2D PCA on all metric rows for selected worlds; draw trajectories in PC space over time."""
    from sklearn.decomposition import PCA

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    metric_cols = [c for c in METRIC_KEYS if c in df.columns]
    present = set(int(x) for x in df["yield_index"].unique())
    yids = [int(y) for y in yield_indices if int(y) in present]
    sub = df[df["yield_index"].isin(yids)].copy()
    if sub.empty or len(metric_cols) < 2 or not yids:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "insufficient data for PCA", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
        return

    X = sub[metric_cols].to_numpy(dtype=np.float64)
    if X.shape[0] < 2:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "need ≥2 rows for PCA", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
        return

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    sub["_pc1"] = Z[:, 0]
    sub["_pc2"] = Z[:, 1]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = plt.get_cmap("tab10")
    for j, yid in enumerate(sorted(yids)):
        seg = sub[sub["yield_index"] == yid].sort_values("ca_step")
        if seg.empty:
            continue
        c = cmap(j % 10)
        ax.plot(
            seg["_pc1"],
            seg["_pc2"],
            color=c,
            alpha=0.55,
            linewidth=1.4,
        )
        ax.scatter(
            seg["_pc1"],
            seg["_pc2"],
            color=c,
            s=18,
            label=f"yield {yid}",
            edgecolors="k",
            linewidths=0.25,
            zorder=5,
        )
    v0, v1 = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({100.0 * float(v0):.1f}% var)")
    ax.set_ylabel(f"PC2 ({100.0 * float(v1):.1f}% var)")
    ax.set_title(
        title
        or "PCA of per-step metrics (lines connect increasing ca_step within each world)"
    )
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.25)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)

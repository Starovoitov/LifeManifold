"""Matplotlib figures for world-space (embedding scatter, CA grids, CA-step traces).

Scatter ``embedding_2d`` matches ``pipeline.stream_world_space_to_jsonl``: **x** is the
dominant (highest-variance) metric minus batch mean; **y** is sklearn's first PC on the
other six metrics (sklearn centers those columns once in ``fit``/``transform``) —
not a 7D PCA decomposition and not PC2 of a full PCA.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..metrics import METRIC_KEYS
from ..simulator import SimulationResult

_EMBED_X_DEFAULT = "Δ metric (unknown)"
_EMBED_Y_DEFAULT = "PC1 of 6 metrics (excluding selected metric)"


def plot_world_embedding(
    points: list,
    path: str | Path,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 120,
) -> None:
    """Save a 2D scatter of metric embedding; ``points`` items need ``embedding_2d`` and ``cluster_id``."""
    if not points:
        frame = pd.DataFrame(columns=["x", "y", "cluster_id"])
        x_label, y_label = _EMBED_X_DEFAULT, _EMBED_Y_DEFAULT
    else:
        frame = pd.DataFrame(
            {
                "x": [float(p.embedding_2d[0]) for p in points],
                "y": [float(p.embedding_2d[1]) for p in points],
                "cluster_id": [int(p.cluster_id) for p in points],
            }
        )
        x_label, y_label = _axis_labels_from_first_point(points[0])
    _scatter_world_embedding(
        frame,
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
    """Read a world-space JSONL file (e.g. ``--metrics-trace``) and plot embedding scatter."""
    raw = pd.read_json(Path(jsonl_path), lines=True)
    if not raw.empty and "embedding_axes" in raw.columns:
        axes_series = pd.Series(
            raw["embedding_axes"].to_numpy(),
            index=raw.index,
            dtype=object,
        )
    else:
        axes_series = pd.Series(dtype=object)
    x_label, y_label = _axis_labels_from_embedding_axes_series(axes_series)
    if raw.empty:
        frame = pd.DataFrame(columns=["x", "y", "cluster_id"])
    else:
        frame = pd.DataFrame(
            raw["embedding_2d"].tolist(),
            columns=["x", "y"],
            index=raw.index,
            dtype=float,
        )
        frame["cluster_id"] = raw["cluster_id"].astype(int)
    _scatter_world_embedding(
        frame,
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


def load_ca_step_trace_jsonl(path: str | Path) -> pd.DataFrame:
    """Load ``--ca-step-trace`` JSONL into a flat pandas table (one row per CA step)."""
    p = Path(path)
    df = pd.read_json(p, lines=True)
    if df.empty:
        return df
    if "metrics" not in df.columns:
        raise ValueError("CA step trace JSONL must contain a 'metrics' object per line")
    met = pd.json_normalize(df["metrics"])
    base = df[["yield_index", "ca_step"]].reset_index(drop=True)
    return pd.concat([base, met], axis=1)


def summarize_ca_step_trace_by_world(df: pd.DataFrame) -> pd.DataFrame:
    """Per ``yield_index``, aggregate each metric across ``ca_step`` (mean/std/min/max)."""
    cols = [c for c in METRIC_KEYS if c in df.columns]
    if not cols:
        return pd.DataFrame()
    g = df.groupby("yield_index", sort=True)[cols]
    return g.agg(["mean", "std", "min", "max"])


def plot_ca_step_metrics_timeseries(
    df: pd.DataFrame,
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
    df: pd.DataFrame,
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

    ctx = _ca_step_trace_reduction_context(df, yield_indices)
    if ctx is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "insufficient data for PCA", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
        return

    sub, X, yids, _metric_cols = ctx
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    sub = sub.copy()
    sub["_pc1"] = Z[:, 0]
    sub["_pc2"] = Z[:, 1]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = plt.get_cmap("tab10")
    for j, yid in enumerate(yids):
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


def plot_ca_step_umap_trajectories(
    df: pd.DataFrame,
    yield_indices: list[int],
    path: str | Path,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 120,
) -> None:
    """Fit 2D UMAP on all metric rows for selected worlds; draw trajectories in embedding space over time."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ImportWarning)
        import umap

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    ctx = _ca_step_trace_reduction_context(df, yield_indices)
    if ctx is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "insufficient data for UMAP", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
        return

    sub, X, yids, _metric_cols = ctx
    n_samples = X.shape[0]
    if n_samples > 2:
        n_neighbors = max(2, min(15, n_samples - 1))
    else:
        n_neighbors = max(1, n_samples - 1)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        n_jobs=1,
    )
    Z = reducer.fit_transform(X)
    sub = sub.copy()
    sub["_umap1"] = Z[:, 0]
    sub["_umap2"] = Z[:, 1]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = plt.get_cmap("tab10")
    for j, yid in enumerate(yids):
        seg = sub[sub["yield_index"] == yid].sort_values("ca_step")
        if seg.empty:
            continue
        c = cmap(j % 10)
        ax.plot(
            seg["_umap1"],
            seg["_umap2"],
            color=c,
            alpha=0.55,
            linewidth=1.4,
        )
        ax.scatter(
            seg["_umap1"],
            seg["_umap2"],
            color=c,
            s=18,
            label=f"yield {yid}",
            edgecolors="k",
            linewidths=0.25,
            zorder=5,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(
        title
        or "UMAP of per-step metrics (lines connect increasing ca_step within each world)"
    )
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.25)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)


def _ca_step_trace_reduction_context(
    df: pd.DataFrame,
    yield_indices: list[int],
) -> tuple[pd.DataFrame, np.ndarray, list[int], list[str]] | None:
    """Return ``(sub, X, sorted_yids, metric_cols)`` for PCA/UMAP trajectory plots, or ``None`` if not enough data."""
    metric_cols = [c for c in METRIC_KEYS if c in df.columns]
    present = set(int(x) for x in df["yield_index"].unique())
    yids = [int(y) for y in yield_indices if int(y) in present]
    sub = df[df["yield_index"].isin(yids)].copy()
    if sub.empty or len(metric_cols) < 2 or not yids:
        return None
    X = sub[metric_cols].to_numpy(dtype=np.float64)
    if X.shape[0] < 2:
        return None
    yids_sorted = sorted(yids)
    return sub, X, yids_sorted, metric_cols


def _labels_from_axes_dict(axes: dict | None) -> tuple[str, str]:
    if not isinstance(axes, dict):
        return _EMBED_X_DEFAULT, _EMBED_Y_DEFAULT
    x_metric = axes.get("x_metric")
    if x_metric:
        return f"Δ {x_metric}", f"PC1 of 6 metrics (excluding {x_metric})"
    return (
        str(axes.get("x_label", _EMBED_X_DEFAULT)),
        str(axes.get("y_label", _EMBED_Y_DEFAULT)),
    )


def _axis_labels_from_embedding_axes_series(axes: pd.Series) -> tuple[str, str]:
    """Labels from the last JSONL row that carries a dict ``embedding_axes`` (matches streaming order)."""
    if axes.empty:
        return _EMBED_X_DEFAULT, _EMBED_Y_DEFAULT
    for val in axes.iloc[::-1]:
        if isinstance(val, dict):
            return _labels_from_axes_dict(val)
    return _EMBED_X_DEFAULT, _EMBED_Y_DEFAULT


def _axis_labels_from_first_point(point: object) -> tuple[str, str]:
    axes = getattr(point, "embedding_axes", None)
    if isinstance(axes, dict):
        return _labels_from_axes_dict(axes)
    x_metric = getattr(point, "x_metric", None)
    if x_metric:
        return f"Δ {x_metric}", f"PC1 of 6 metrics (excluding {x_metric})"
    return _EMBED_X_DEFAULT, _EMBED_Y_DEFAULT


def _scatter_world_embedding(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    title: str | None,
    figsize: tuple[float, float],
    dpi: int,
    x_label: str,
    y_label: str,
) -> None:
    """Scatter ``x`` / ``y`` colored by ``cluster_id``; empty ``frame`` writes the empty-state figure."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if frame.empty:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "no points", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
        return

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for idx, (cid, g) in enumerate(frame.groupby("cluster_id", sort=True)):
        color = cmap(idx % 10)
        ax.scatter(
            g["x"],
            g["y"],
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

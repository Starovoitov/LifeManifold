"""Matplotlib figures for world-space (per-world metric scatters, CA grids, CA-step traces)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import math as ws_math
from ..metrics import METRIC_KEYS
from ..pipeline import dominant_metric_delta_xy_batch
from ..simulator import SimulationResult


def plot_world_metrics_pca_scatter_from_jsonl(
    jsonl_path: str | Path,
    path: str | Path,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 120,
    k_clusters: int = 4,
    standardize_metrics: bool = False,
) -> None:
    """
    2D scatter of worlds in PCA space of the seven ``METRIC_KEYS`` columns
    (from each line's ``metrics`` object).

    Point colors always use ``cluster_id`` from JSONL or Lloyd k-means on **raw**
    metrics (never on z-scored metrics). When ``standardize_metrics`` is true, PCA
    is fit on per-batch z-scored columns (``*_norm.png``).
    """
    from sklearn.decomposition import PCA

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw, X = _final_world_metrics_matrix_from_jsonl(jsonl_path)
    cluster_ids = _world_metrics_cluster_labels(X, raw, k_clusters=k_clusters)
    X_fit = _standardize_metrics_matrix(X) if standardize_metrics else X

    pca = PCA(n_components=2, svd_solver="full")
    Z = pca.fit_transform(X_fit)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _scatter_world_metrics_by_cluster(ax, Z, cluster_ids)
    v0, v1 = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({100.0 * float(v0):.1f}% var)")
    ax.set_ylabel(f"PC2 ({100.0 * float(v1):.1f}% var)")
    z_note = "z-scored 7D batch; " if standardize_metrics else ""
    ax.set_title(
        title or (f"PCA scatter ({z_note}7D → 2D; {_raw_cluster_color_note(raw)})")
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)


def plot_world_metrics_umap_scatter_from_jsonl(
    jsonl_path: str | Path,
    path: str | Path,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 120,
    k_clusters: int = 4,
    standardize_metrics: bool = False,
) -> None:
    """
    2D UMAP of the seven per-world ``metrics`` columns (same as
    :func:`plot_world_metrics_pca_scatter_from_jsonl`).

    Colors: ``cluster_id`` / k-means on **raw** metrics only. Layout uses raw or
    z-scored metrics per ``standardize_metrics``.

    Requires at least **three** rows. Uses ``init="random"`` on small batches.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ImportWarning)
        import umap

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw, X = _final_world_metrics_matrix_from_jsonl(jsonl_path)
    n = int(X.shape[0])
    if n < 3:
        raise ValueError(
            "Metrics JSONL for umap.png needs at least three lines (worlds); "
            "UMAP requires n_neighbors > 1."
        )

    cluster_ids = _world_metrics_cluster_labels(X, raw, k_clusters=k_clusters)
    X_fit = _standardize_metrics_matrix(X) if standardize_metrics else X
    n_neighbors = max(2, min(15, n - 1))

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        n_jobs=1,
        init="random",
    )
    Z = np.asarray(reducer.fit_transform(X_fit), dtype=np.float64)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _scatter_world_metrics_by_cluster(ax, Z, cluster_ids)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    z_note = "z-scored 7D batch; " if standardize_metrics else ""
    ax.set_title(title or (f"UMAP scatter ({z_note}{_raw_cluster_color_note(raw)})"))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)


def plot_dominant_metric_delta_scatter_from_jsonl(
    jsonl_path: str | Path,
    path: str | Path,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 120,
    k_clusters: int = 4,
    standardize_metrics: bool = False,
) -> None:
    """
    Scatter worlds in **dominant-metric-delta** layout.

    **x**: Δ dominant metric (highest variance in batch); **y**: PC1 of the other six.

    Raw layout (default) uses stored ``dominant_metric_delta_xy`` when present; z-scored
    layout (``standardize_metrics``) always recomputes via
    :func:`dominant_metric_delta_xy_batch` on per-batch z-scored metrics.

    Colors: ``cluster_id`` / k-means on **raw** metrics only.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw_pd = pd.read_json(Path(jsonl_path), lines=True)

    if raw_pd.empty:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "no points", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
        return

    xy_col = _dominant_metric_delta_xy_jsonl_column(raw_pd)
    labels_col = _dominant_metric_delta_axis_labels_jsonl_column(raw_pd)
    has_metrics = "metrics" in raw_pd.columns
    raw_m: pd.DataFrame | None = None

    if standardize_metrics:
        if not has_metrics:
            raise ValueError(
                "Metrics JSONL for dominant_metric_delta_norm.png needs per-line "
                "``metrics`` with all standard keys."
            )
        raw_m, X = _final_world_metrics_matrix_from_jsonl(jsonl_path)
        cluster_ids = _world_metrics_cluster_labels(X, raw_m, k_clusters=k_clusters)
        X_fit = _standardize_metrics_matrix(X)
        Z, axis_dict = dominant_metric_delta_xy_batch(X_fit)
        x_label, y_label = _axis_labels_from_labels_dict(axis_dict)
        x_label = f"z-scored {x_label}"
        y_label = f"z-scored {y_label}"
    elif xy_col is not None:
        Z = np.asarray(raw_pd[xy_col].tolist(), dtype=np.float64)
        x_label, y_label = _axis_labels_from_dominant_metric_delta_labels_series(
            raw_pd, labels_col
        )
        if has_metrics:
            raw_m, X = _final_world_metrics_matrix_from_jsonl(jsonl_path)
            cluster_ids = _world_metrics_cluster_labels(X, raw_m, k_clusters=k_clusters)
        elif "cluster_id" in raw_pd.columns:
            cluster_ids = raw_pd["cluster_id"].astype(int).to_numpy()
        else:
            raise ValueError(
                "Metrics JSONL with precomputed dominant_metric_delta_xy needs either a full "
                "``metrics`` object per line (for k-means coloring) or ``cluster_id``."
            )
    elif has_metrics:
        raw_m, X = _final_world_metrics_matrix_from_jsonl(jsonl_path)
        cluster_ids = _world_metrics_cluster_labels(X, raw_m, k_clusters=k_clusters)
        Z, axis_dict = dominant_metric_delta_xy_batch(X)
        x_label, y_label = _axis_labels_from_labels_dict(axis_dict)
    else:
        raise ValueError(
            "Metrics JSONL for dominant_metric_delta.png needs "
            "``dominant_metric_delta_xy`` or per-line "
            "``metrics`` with all standard keys."
        )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _scatter_world_metrics_by_cluster(ax, Z, cluster_ids)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    z_note = "z-scored batch; " if standardize_metrics else ""
    color_note = (
        _raw_cluster_color_note(raw_m)
        if raw_m is not None
        else "colors: cluster_id (raw)"
    )
    ax.set_title(title or (f"Dominant-metric-delta ({z_note}{color_note})"))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)


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
    met = pd.json_normalize(cast(pd.Series, df["metrics"]))
    base = df[["yield_index", "ca_step"]].reset_index(drop=True)
    return pd.concat([base, met], axis=1)


def summarize_ca_step_trace_by_world(df: pd.DataFrame) -> pd.DataFrame:
    """Per ``yield_index``, aggregate each metric across ``ca_step`` (mean/std/min/max)."""
    cols = [c for c in METRIC_KEYS if c in df.columns]
    if not cols:
        return pd.DataFrame()
    g = df.groupby("yield_index", sort=True)[cols]
    return cast(pd.DataFrame, g.agg(["mean", "std", "min", "max"]))


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
        ax.text(
            0.5,
            0.5,
            "no data for plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
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
            seg = cast(pd.DataFrame, df.loc[df["yield_index"] == yid]).sort_values(
                by="ca_step"
            )
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
    standardize_metrics: bool = False,
) -> None:
    """Fit 2D PCA on all metric rows for selected worlds; draw trajectories in PC space over time."""
    from sklearn.decomposition import PCA

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    ctx = _ca_step_trace_reduction_context(df, yield_indices)
    if ctx is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(
            0.5,
            0.5,
            "insufficient data for PCA",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
        return

    sub, X, yids, _metric_cols = ctx
    X_fit = _standardize_metrics_matrix(X) if standardize_metrics else X
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X_fit)
    sub = sub.copy()
    sub["_pc1"] = Z[:, 0]
    sub["_pc2"] = Z[:, 1]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = plt.get_cmap("tab10")
    for j, yid in enumerate(yids):
        seg = cast(pd.DataFrame, sub.loc[sub["yield_index"] == yid]).sort_values(
            by="ca_step"
        )
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
    z_note = "z-scored per-step batch; " if standardize_metrics else ""
    ax.set_title(
        title
        or (
            f"{z_note}PCA of per-step metrics "
            "(lines connect increasing ca_step within each world)"
        )
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
    standardize_metrics: bool = False,
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
        ax.text(
            0.5,
            0.5,
            "insufficient data for UMAP",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
        return

    sub, X, yids, _metric_cols = ctx
    X_fit = _standardize_metrics_matrix(X) if standardize_metrics else X
    n_samples = X_fit.shape[0]
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
    Z = np.asarray(reducer.fit_transform(X_fit), dtype=np.float64)
    sub = sub.copy()
    sub["_umap1"] = Z[:, 0]
    sub["_umap2"] = Z[:, 1]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = plt.get_cmap("tab10")
    for j, yid in enumerate(yids):
        seg = cast(pd.DataFrame, sub.loc[sub["yield_index"] == yid]).sort_values(
            by="ca_step"
        )
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
    z_note = "z-scored per-step batch; " if standardize_metrics else ""
    ax.set_title(
        title
        or (
            f"{z_note}UMAP of per-step metrics "
            "(lines connect increasing ca_step within each world)"
        )
    )
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.25)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)


def _final_world_metrics_matrix_from_jsonl(
    jsonl_path: str | Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load metrics-trace JSONL (one world per line): require ``metrics`` with all
    ``METRIC_KEYS``. Returns ``(raw, X)`` with ``X`` of shape ``(n, 7)``.
    """
    raw = pd.read_json(Path(jsonl_path), lines=True)
    if raw.empty:
        raise ValueError("Metrics JSONL is empty.")
    if "metrics" not in raw.columns:
        raise ValueError(
            "Metrics JSONL must contain a ``metrics`` object per line with numeric "
            f"fields {list(METRIC_KEYS)}."
        )
    met = pd.json_normalize(cast(Any, raw["metrics"]))
    missing = [k for k in METRIC_KEYS if k not in met.columns]
    if missing:
        raise ValueError(
            "Each ``metrics`` object must include all standard keys "
            f"{list(METRIC_KEYS)}. Missing: {missing}"
        )
    X = met[list(METRIC_KEYS)].to_numpy(dtype=np.float64)
    if np.isnan(X).any():
        raise ValueError("Metrics JSONL contains NaN in metric columns.")
    n = int(X.shape[0])
    if n < 2:
        raise ValueError(
            "Metrics JSONL needs at least two lines (worlds) for 2D PCA / UMAP scatter."
        )
    return raw, X


def _standardize_metrics_matrix(X: np.ndarray) -> np.ndarray:
    """Per-column z-score on batch ``X`` (population std; zero std → scale 1)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (X - mean) / std


def _raw_cluster_color_note(raw: pd.DataFrame) -> str:
    if "cluster_id" in raw.columns:
        return "colors: cluster_id (pipeline k-means on raw 7D)"
    return "colors: k-means on raw 7D metrics"


def _world_metrics_cluster_labels(
    X: np.ndarray,
    raw: pd.DataFrame,
    *,
    k_clusters: int = 4,
) -> np.ndarray:
    """
    Per-world labels for scatter coloring (always from **raw** metrics).

    Uses ``cluster_id`` from JSONL when present; otherwise Lloyd k-means on raw ``X``.
    Never cluster on z-scored metrics (``*_norm.png`` layouts use these labels).
    """
    if "cluster_id" in raw.columns:
        return raw["cluster_id"].astype(int).to_numpy()
    n = int(X.shape[0])
    labels = np.zeros(n, dtype=np.int32)
    rows = np.ascontiguousarray(X, dtype=np.float32)
    ws_math.kmeans_lloyd_on_memmap(
        cast(np.memmap, rows),
        cast(np.memmap, labels),
        n,
        k_clusters,
    )
    return labels.astype(int)


def _scatter_world_metrics_by_cluster(
    ax: plt.Axes,
    Z: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    point_size: float = 45,
) -> None:
    """Scatter ``Z`` with one color per k-means cluster id."""
    cmap = plt.get_cmap("tab10")
    for j, cid in enumerate(sorted(set(int(c) for c in cluster_ids))):
        mask = cluster_ids == cid
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            c=[cmap(j % 10)],
            label=f"cluster {cid}",
            alpha=0.9,
            edgecolors="k",
            linewidths=0.35,
            s=point_size,
        )
    ax.legend(title="cluster (raw)", loc="best")


def _dominant_metric_delta_xy_jsonl_column(raw: pd.DataFrame) -> str | None:
    if "dominant_metric_delta_xy" in raw.columns:
        return "dominant_metric_delta_xy"
    if "world_space_xy" in raw.columns:
        return "world_space_xy"
    if "embedding_2d" in raw.columns:
        return "embedding_2d"
    return None


def _dominant_metric_delta_axis_labels_jsonl_column(raw: pd.DataFrame) -> str | None:
    if "dominant_metric_delta_axis_labels" in raw.columns:
        return "dominant_metric_delta_axis_labels"
    if "world_space_axis_labels" in raw.columns:
        return "world_space_axis_labels"
    if "embedding_axes" in raw.columns:
        return "embedding_axes"
    return None


def _axis_labels_from_labels_dict(labels: dict | None) -> tuple[str, str]:
    if not isinstance(labels, dict):
        return "Δ metric (unknown)", "PC1 of 6 other metrics"
    x_metric = labels.get("x_metric")
    if x_metric:
        return f"Δ {x_metric}", f"PC1 of 6 metrics (excluding {x_metric})"
    return (
        str(labels.get("x_label", "Δ metric")),
        str(labels.get("y_label", "PC1 of 6 other metrics")),
    )


def _axis_labels_from_dominant_metric_delta_labels_series(
    raw: pd.DataFrame,
    labels_col: str | None,
) -> tuple[str, str]:
    if labels_col is None or labels_col not in raw.columns:
        return "Δ metric (unknown)", "PC1 of 6 other metrics"
    for val in raw[labels_col].iloc[::-1]:
        if isinstance(val, dict):
            return _axis_labels_from_labels_dict(val)
    return "Δ metric (unknown)", "PC1 of 6 other metrics"


def _ca_step_trace_reduction_context(
    df: pd.DataFrame,
    yield_indices: list[int],
) -> tuple[pd.DataFrame, np.ndarray, list[int], list[str]] | None:
    """Return ``(sub, X, sorted_yids, metric_cols)`` for PCA/UMAP trajectory plots, or ``None`` if not enough data."""
    metric_cols = [c for c in METRIC_KEYS if c in df.columns]
    present = set(int(x) for x in df["yield_index"].unique())
    yids = [int(y) for y in yield_indices if int(y) in present]
    sub = cast(pd.DataFrame, df.loc[df["yield_index"].isin(yids)].copy())
    if sub.empty or len(metric_cols) < 2 or not yids:
        return None
    X = cast(pd.DataFrame, sub[metric_cols]).to_numpy(dtype=np.float64)
    if X.shape[0] < 2:
        return None
    yids_sorted = sorted(yids)
    return sub, X, yids_sorted, metric_cols

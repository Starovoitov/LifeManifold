"""Matplotlib figures for world-space exploration (kept inside ``worldspace`` only)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .simulator import SimulationResult


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
    _scatter_embedding(xs, ys, clusters, path, title=title, figsize=figsize, dpi=dpi)


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
    if not xs:
        _scatter_embedding(
            np.array([]),
            np.array([]),
            np.array([], dtype=int),
            path,
            title=title,
            figsize=figsize,
            dpi=dpi,
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
    ax.set_xlabel("average lifespan − batch mean")
    ax.set_ylabel("max-variance axis ⊥ lifespan (other metrics)")
    ax.set_title(title or "World space (lifespan + orthogonal spread, colored by cluster)")
    ax.legend(title="k-means", loc="best", fontsize="small")
    ax.grid(True, alpha=0.25)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)

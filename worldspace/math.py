"""Numeric helpers for worldspace (k-means, metrics, neighborhood counts)."""

from __future__ import annotations

import zlib

import numpy as np

from .metrics import METRICS_VECTOR_DIM

# Trailing mean-density samples kept for oscillation autocorrelation (O(1) vs. step count).
OSCILLATION_DENSITY_WINDOW = 512


def neighbor_count(grid: np.ndarray) -> np.ndarray:
    """Compute Moore-neighborhood live-neighbor counts with wrap-around edges."""
    total = np.zeros_like(grid, dtype=np.int16)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            total += np.roll(np.roll(grid, dx, axis=0), dy, axis=1)
    return total


def kmeans_lloyd_on_memmap(
    mm: np.memmap,
    labels: np.memmap,
    n: int,
    k: int,
    max_iter: int = 30,
) -> None:
    """Lloyd k-means reading one row at a time from ``mm``; centroids stay in RAM (small ``k``)."""
    rng = np.random.default_rng(42)
    k = max(1, min(k, n))
    centroids = np.stack([mm[i].astype(np.float64).copy() for i in range(k)])

    for _it in range(max_iter):
        changed = False
        for i in range(n):
            v = mm[i].astype(np.float64)
            d2 = ((centroids - v) ** 2).sum(axis=1)
            new_lab = int(np.argmin(d2))
            if int(labels[i]) != new_lab:
                changed = True
            labels[i] = new_lab

        centroids.fill(0.0)
        cnt = np.zeros(k, dtype=np.float64)
        for i in range(n):
            lab = int(labels[i])
            centroids[lab] += mm[i].astype(np.float64)
            cnt[lab] += 1.0
        for j in range(k):
            if cnt[j] > 0:
                centroids[j] /= cnt[j]
            else:
                centroids[j] = rng.standard_normal(METRICS_VECTOR_DIM) * 0.01
        if not changed:
            break


def binary_entropy(p: float) -> float:
    """Compute binary Shannon entropy for occupancy probability `p`."""
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def oscillation(series: np.ndarray, max_lag: int = 10) -> float:
    """Estimate oscillation strength via normalized autocorrelation peaks."""
    if len(series) < 3:
        return 0.0
    centered = series - series.mean()
    denom = np.dot(centered, centered) + 1e-9
    scores = []
    for lag in range(1, min(max_lag + 1, len(series))):
        num = np.dot(centered[:-lag], centered[lag:])
        scores.append(abs(num / denom))
    return float(max(scores) if scores else 0.0)


def pattern_diversity_from_frame(
    frame: np.ndarray | None, sample_size: int = 128
) -> float:
    """Estimate local pattern diversity from random 3x3 patches on one grid."""
    if frame is None or frame.size == 0:
        return 0.0
    rng = np.random.default_rng(0)
    n = int(frame.shape[0])
    signatures: set[tuple[int, ...]] = set()
    for _ in range(sample_size):
        x = int(rng.integers(0, n))
        y = int(rng.integers(0, n))
        patch = frame.take([(x - 1) % n, x % n, (x + 1) % n], axis=0).take(
            [(y - 1) % n, y % n, (y + 1) % n], axis=1
        )
        signatures.add(tuple(int(v) for v in patch.reshape(-1)))
    return float(len(signatures) / sample_size)


def pattern_diversity(history: list[np.ndarray], sample_size: int = 128) -> float:
    """Backward-compatible wrapper: use last frame if the list is non-empty."""
    if not history:
        return 0.0
    return pattern_diversity_from_frame(history[-1], sample_size=sample_size)


def topology_interface_index(life: np.ndarray) -> float:
    """
    Toroidal Moore neighborhood: mean fraction of neighbors that differ in ``life``.

    High values indicate fragmented boundaries (many interfaces); low values near
    homogeneous all-dead or all-live regions. In ``[0, 1]``.
    """
    if life.size == 0:
        return 0.0
    g = life.astype(np.float32)
    diff_sum = np.zeros_like(g, dtype=np.float32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nb = np.roll(np.roll(g, dx, axis=0), dy, axis=1)
            diff_sum += (nb != g).astype(np.float32)
    return float(np.clip(diff_sum.mean() / 8.0, 0.0, 1.0))


def topology_window_heterogeneity(life: np.ndarray) -> float:
    """
    Fraction of toroidal 2×2 windows whose four corners are not all identical.

    Mesoscale proxy for local non-trivial topology / mixing (not full Betti numbers).
    In ``[0, 1]``.
    """
    if life.size == 0:
        return 0.0
    a = life
    b = np.roll(life, -1, axis=0)
    c = np.roll(life, -1, axis=1)
    d = np.roll(np.roll(life, -1, axis=0), -1, axis=1)
    stacked = np.stack([a, b, c, d], axis=0)
    hetero = stacked.max(axis=0) != stacked.min(axis=0)
    return float(np.clip(hetero.mean(), 0.0, 1.0))


def compressibility_score_joint(life: np.ndarray, food: np.ndarray) -> float:
    """
    zlib-based proxy for **description length** / approximate computability.

    Concatenates binary ``life`` and ``food`` row-major, compresses with zlib (level 6),
    returns ``1 - len(compressed)/len(raw)`` clipped to ``[0, 1]``. Highly ordered
    configurations compress well (score near 1); noisy near-random fields do not (near 0).
    """
    raw = life.astype(np.uint8, copy=False).tobytes() + food.astype(
        np.uint8, copy=False
    ).tobytes()
    n = len(raw)
    if n == 0:
        return 0.0
    compressed = zlib.compress(raw, level=6)
    ratio = len(compressed) / float(n)
    return float(np.clip(1.0 - ratio, 0.0, 1.0))


def ecology_state_entropy_norm(life: np.ndarray, food: np.ndarray) -> float:
    """
    Shannon entropy of the joint ``(life, food)`` state per cell, normalized to ``[0, 1]``.

    Encodes four classes: ``code = life + 2*food`` in ``{0,1,2,3}``. Normalizes by
    ``log2(k)`` where ``k`` is the number of **non-empty** classes (so sparse ecologies
    are not penalized by unused bins).
    """
    if life.size == 0:
        return 0.0
    code = life.astype(np.int32) + 2 * food.astype(np.int32)
    counts = np.bincount(code.ravel(), minlength=4).astype(np.float64)
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    nz = counts[counts > 0]
    p = nz / total
    k = int(p.size)
    if k <= 1:
        return 0.0
    h = float(-(p * np.log2(p + 1e-15)).sum())
    h_max = float(np.log2(k))
    return float(np.clip(h / h_max, 0.0, 1.0)) if h_max > 0.0 else 0.0


def ecology_resource_adjacency(life: np.ndarray, food: np.ndarray) -> float:
    """
    Mean Moore fraction of **food** around **live** cells (toroidal).

    In ``[0, 1]``; measures spatial coupling between consumers (``life==1``) and
    resources (``food==1``) on the multi-type grid.
    """
    live = life == 1
    if not np.any(live):
        return 0.0
    fsum = neighbor_count(food.astype(np.int16)).astype(np.float64)
    return float(np.clip((fsum[live] / 8.0).mean(), 0.0, 1.0))

"""Numeric helpers for worldspace (k-means, metrics, neighborhood counts)."""

from __future__ import annotations

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

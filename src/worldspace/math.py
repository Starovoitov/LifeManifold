"""Numeric helpers for worldspace (PCA, k-means, metrics, neighborhood counts)."""

from __future__ import annotations

import numpy as np


def neighbor_count(grid: np.ndarray) -> np.ndarray:
    """Compute Moore-neighborhood live-neighbor counts with wrap-around edges."""
    total = np.zeros_like(grid, dtype=np.int16)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            total += np.roll(np.roll(grid, dx, axis=0), dy, axis=1)
    return total


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Project row vectors to 2D using SVD-based PCA."""
    if matrix.shape[0] <= 1:
        return np.zeros((matrix.shape[0], 2), dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    return centered @ basis


def kmeans(matrix: np.ndarray, k: int = 4, max_iter: int = 30) -> np.ndarray:
    """Cluster row vectors using a compact in-module k-means."""
    if len(matrix) == 0:
        return np.array([], dtype=int)
    k = max(1, min(k, len(matrix)))
    rng = np.random.default_rng(42)
    centroids = matrix[rng.choice(len(matrix), size=k, replace=False)]
    labels = np.zeros(len(matrix), dtype=int)
    for _ in range(max_iter):
        distances = ((matrix[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for idx in range(k):
            members = matrix[labels == idx]
            if len(members) > 0:
                centroids[idx] = members.mean(axis=0)
    return labels


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


def pattern_diversity(history: list[np.ndarray], sample_size: int = 128) -> float:
    """Estimate local pattern diversity from random 3x3 patches."""
    if not history:
        return 0.0
    rng = np.random.default_rng(0)
    frame = history[-1]
    n = frame.shape[0]
    signatures: set[tuple[int, ...]] = set()
    for _ in range(sample_size):
        x = int(rng.integers(0, n))
        y = int(rng.integers(0, n))
        patch = frame.take([(x - 1) % n, x % n, (x + 1) % n], axis=0).take(
            [(y - 1) % n, y % n, (y + 1) % n], axis=1
        )
        signatures.add(tuple(int(v) for v in patch.reshape(-1)))
    return float(len(signatures) / sample_size)

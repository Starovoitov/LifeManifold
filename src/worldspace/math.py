"""Numeric helpers for worldspace (PCA, k-means, metrics, neighborhood counts)."""

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


def pca_mean_and_basis_2d(
    sum_x: np.ndarray, sum_xx: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA mean and first two loadings from sufficient statistics only (O(1) memory in n)."""
    d = METRICS_VECTOR_DIM
    if n <= 0:
        z = np.zeros(d, dtype=float)
        return z, np.zeros((d, 2), dtype=float)
    mean = sum_x / float(n)
    if n == 1:
        return mean, np.zeros((d, 2), dtype=float)
    cov = (sum_xx - np.outer(sum_x, sum_x) / float(n)) / float(n - 1)
    cov = 0.5 * (cov + cov.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    basis = evecs[:, order[:2]]
    return mean, basis


def project_pca_2d(
    vec: np.ndarray, mean: np.ndarray, basis: np.ndarray
) -> tuple[float, float]:
    """Project a metrics vector to 2D given batch PCA mean and basis (``METRICS_VECTOR_DIM`` × 2)."""
    xy = basis.T @ (vec - mean)
    return float(xy[0]), float(xy[1])

"""Numeric helpers for worldspace (k-means, metrics, neighborhood counts)."""

from __future__ import annotations

import zlib

import numpy as np

# Trailing mean-density samples kept for oscillation autocorrelation (O(1) vs. step count).
OSCILLATION_DENSITY_WINDOW = 512


def neighbor_count(grid: np.ndarray) -> np.ndarray:
    """Compute Moore-neighborhood live-neighbor counts with wrap-around edges."""
    total = np.zeros_like(grid, dtype=np.int16)
    for view in _moore_stencil_views(grid):
        total += view.astype(np.int16, copy=False)
    return total


def rule_count_masks(
    birth: list[int], survival: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``bool[9]`` lookup tables for Moore neighbor counts (0–8)."""
    birth_mask = np.zeros(9, dtype=bool)
    survival_mask = np.zeros(9, dtype=bool)
    birth_mask[birth] = True
    survival_mask[survival] = True
    return birth_mask, survival_mask


def langton_lambda_runtime(activity_sum: float, activity_steps: int) -> float:
    """Mean per-step life flip fraction (runtime Langton activity proxy, in [0, 1])."""
    if activity_steps <= 0:
        return 0.0
    return float(np.clip(activity_sum / activity_steps, 0.0, 1.0))


def kmeans_lloyd_on_memmap(
    mm: np.memmap,
    labels: np.memmap,
    n: int,
    k: int,
    max_iter: int = 30,
) -> None:
    """Assign k-means cluster labels in place; ``mm`` may be a memmap (sklearn mini-batches)."""
    if n <= 0:
        return

    # Lazy import: sklearn is heavy; pipeline callers often skip k-means entirely.
    from sklearn.cluster import MiniBatchKMeans

    k = max(1, min(k, n))
    model = MiniBatchKMeans(
        n_clusters=k,
        max_iter=max_iter,
        random_state=42,
        batch_size=min(256, n),
        n_init="auto",
    )
    assigned = model.fit_predict(mm[:n])
    labels[:n] = assigned.astype(np.int32, copy=False)


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
    n = len(centered)
    max_lag_eff = min(max_lag, n - 1)
    if max_lag_eff < 1:
        return 0.0
    ac = np.correlate(centered, centered, mode="full")
    # ac[n-1+k] = lag-k correlation; slice lags 1..max_lag_eff
    scores = np.abs(ac[n : n + max_lag_eff] / denom)
    return float(np.max(scores))


def pattern_diversity_from_frame(
    frame: np.ndarray | None, sample_size: int = 128
) -> float:
    """Estimate local pattern diversity from random 3x3 patches on one grid."""
    if frame is None or frame.size == 0:
        return 0.0
    rng = np.random.default_rng(0)
    n = int(frame.shape[0])
    coords = rng.integers(0, n, size=(sample_size, 2), dtype=np.intp)
    xs = coords[:, 0]
    ys = coords[:, 1]
    rows = np.stack([(xs - 1) % n, xs % n, (xs + 1) % n], axis=1)
    cols = np.stack([(ys - 1) % n, ys % n, (ys + 1) % n], axis=1)
    patches = frame[rows[:, :, None], cols[:, None, :]]
    bits = patches.reshape(sample_size, 9).astype(np.uint16)
    weights = 1 << np.arange(9, dtype=np.uint16)
    packed = (bits * weights).sum(axis=1)
    return float(len(np.unique(packed)) / sample_size)


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
    for nb in _moore_stencil_views(g):
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
    raw = (
        life.astype(np.uint8, copy=False).tobytes()
        + food.astype(np.uint8, copy=False).tobytes()
    )
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


def topology_interface_strength_map(life: np.ndarray) -> np.ndarray:
    """
    Per-cell Moore fraction of neighbors that differ in ``life`` (toroidal). Shape ``(n, n)``, values in ``[0, 1]``.
    """
    if life.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    g = life.astype(np.float32)
    diff_sum = np.zeros_like(g, dtype=np.float32)
    for nb in _moore_stencil_views(g):
        diff_sum += (nb != g).astype(np.float32)
    return np.clip(diff_sum / 8.0, 0.0, 1.0).astype(np.float64)


def topology_2x2_heterogeneity_map(life: np.ndarray) -> np.ndarray:
    """
    Toroidal 2×2 window heterogeneity: ``1.0`` if the four corners are not all equal, else ``0.0``.
    Value at ``(i, j)`` is for the window with top-left ``(i, j)``. Shape ``(n, n)``.
    """
    if life.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    a = life
    b = np.roll(life, -1, axis=0)
    c = np.roll(life, -1, axis=1)
    d = np.roll(np.roll(life, -1, axis=0), -1, axis=1)
    stacked = np.stack([a, b, c, d], axis=0)
    hetero = (stacked.max(axis=0) != stacked.min(axis=0)).astype(np.float64)
    return hetero


def food_neighbor_fraction_map(food: np.ndarray) -> np.ndarray:
    """Moore sum of ``food`` in 8 neighbors, divided by 8 (toroidal). Shape matches ``food``."""
    if food.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    return (neighbor_count(food.astype(np.int16)).astype(np.float64) / 8.0).clip(
        0.0, 1.0
    )


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


_MOORE_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _moore_stencil_views(grid: np.ndarray) -> tuple[np.ndarray, ...]:
    """Eight toroidal Moore neighbor views aligned with ``grid`` (center excluded).

    Views share the padded buffer; do not modify ``grid`` between call and consumption.
    """
    padded = np.pad(grid, 1, mode="wrap")
    n = int(grid.shape[0])
    return tuple(
        padded[1 + dx : 1 + dx + n, 1 + dy : 1 + dy + n] for dx, dy in _MOORE_OFFSETS
    )

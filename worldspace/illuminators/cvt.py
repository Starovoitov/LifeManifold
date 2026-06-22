"""CVT helpers: Lloyd centroids, BC assignment, and Voronoi adjacency in [0, 1]^2."""

from __future__ import annotations

import numpy as np

from worldspace.illuminators.archive import BC_MAX, BC_MIN

DEFAULT_LLOYD_ITERATIONS = 50
DEFAULT_CVT_SAMPLE_MULTIPLIER = 50
MIN_CVT_SAMPLES = 10_000
VORONOI_GRID_RESOLUTION = 200

__all__ = [
    "DEFAULT_CVT_SAMPLE_MULTIPLIER",
    "DEFAULT_LLOYD_ITERATIONS",
    "MIN_CVT_SAMPLES",
    "VORONOI_GRID_RESOLUTION",
    "assign_cell_id",
    "generate_centroids",
    "voronoi_neighbors",
]


def generate_centroids(
    n_centroids: int,
    *,
    seed: int = 0,
    lloyd_iterations: int = DEFAULT_LLOYD_ITERATIONS,
    bc_min: float = BC_MIN,
    bc_max: float = BC_MAX,
) -> np.ndarray:
    """Build ``n_centroids`` CVT centroids in behavioral space via Lloyd relaxation."""
    if n_centroids < 1:
        msg = f"n_centroids must be >= 1, got {n_centroids}"
        raise ValueError(msg)
    if lloyd_iterations < 1:
        msg = f"lloyd_iterations must be >= 1, got {lloyd_iterations}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    n_samples = max(MIN_CVT_SAMPLES, DEFAULT_CVT_SAMPLE_MULTIPLIER * n_centroids)
    samples = rng.uniform(bc_min, bc_max, size=(n_samples, 2)).astype(np.float64)
    init_indices = rng.choice(n_samples, size=n_centroids, replace=False)
    centroids = samples[init_indices].copy()

    for _ in range(lloyd_iterations):
        centroids = _lloyd_step(samples, centroids, rng)

    return np.clip(centroids, bc_min, bc_max)


def assign_cell_id(
    stability: float,
    diversity: float,
    centroids: np.ndarray,
) -> int:
    """Map behavioral coordinates to the nearest centroid index."""
    point = np.array(
        [_clip_unit(stability), _clip_unit(diversity)],
        dtype=np.float64,
    )
    distances_sq = ((centroids - point) ** 2).sum(axis=1)
    return int(np.argmin(distances_sq))


def voronoi_neighbors(
    centroids: np.ndarray,
    *,
    grid_resolution: int = VORONOI_GRID_RESOLUTION,
) -> dict[int, tuple[int, ...]]:
    """Return Voronoi adjacency as sorted neighbor tuples per centroid index."""
    n_centroids = int(centroids.shape[0])
    if n_centroids < 1:
        msg = "centroids must contain at least one row"
        raise ValueError(msg)
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        msg = f"centroids must have shape (n, 2), got {centroids.shape}"
        raise ValueError(msg)

    edges: list[set[int]] = [set() for _ in range(n_centroids)]
    stability_axis = np.linspace(BC_MIN, BC_MAX, grid_resolution, dtype=np.float64)
    diversity_axis = np.linspace(BC_MIN, BC_MAX, grid_resolution, dtype=np.float64)
    owners = np.zeros((grid_resolution, grid_resolution), dtype=np.int32)
    for row, stability in enumerate(stability_axis):
        for col, diversity in enumerate(diversity_axis):
            owners[row, col] = assign_cell_id(
                float(stability),
                float(diversity),
                centroids,
            )

    for row in range(grid_resolution):
        for col in range(grid_resolution):
            owner = int(owners[row, col])
            if row + 1 < grid_resolution:
                _link_neighbors(edges, owner, int(owners[row + 1, col]))
            if col + 1 < grid_resolution:
                _link_neighbors(edges, owner, int(owners[row, col + 1]))

    return {
        cell_id: tuple(sorted(neighbors)) for cell_id, neighbors in enumerate(edges)
    }


def _lloyd_step(
    samples: np.ndarray,
    centroids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """One Lloyd iteration: assign samples and recompute centroids."""
    n_centroids = centroids.shape[0]
    diff = samples[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    labels = np.argmin((diff**2).sum(axis=2), axis=1)

    updated = np.zeros_like(centroids)
    for cell_id in range(n_centroids):
        mask = labels == cell_id
        if np.any(mask):
            updated[cell_id] = samples[mask].mean(axis=0)
        else:
            updated[cell_id] = samples[int(rng.integers(0, samples.shape[0]))]
    return updated


def _link_neighbors(edges: list[set[int]], left: int, right: int) -> None:
    if left == right:
        return
    edges[left].add(right)
    edges[right].add(left)


def _clip_unit(value: float) -> float:
    return float(np.clip(value, BC_MIN, BC_MAX))

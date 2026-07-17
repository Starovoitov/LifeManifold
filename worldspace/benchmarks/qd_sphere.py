"""Fontaine-style linear-projection Sphere and Rastrigin QD benchmarks.

This is a lightweight D=20 adaptation of the pyribs ``examples/sphere.py``
benchmark. Solutions are clipped to ``[-5.12, 5.12]`` before objective and
measure evaluation. The behavior measures are sums over the first and second
halves of the solution. Objectives are maximized and normalized to ``[0, 100]``.

Sphere uses the pyribs tutorial optimum shift ``0.4 * 5.12 = 2.048``.
Rastrigin uses a deterministic analytical upper bound rather than batch
min-max scaling, so objective values are comparable across runs.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

DEFAULT_SOLUTION_DIM = 20
DEFAULT_ARCHIVE_DIMS = (100, 100)
CLIP_BOUND = 5.12
SPHERE_SHIFT = 0.4 * CLIP_BOUND

__all__ = [
    "CLIP_BOUND",
    "DEFAULT_ARCHIVE_DIMS",
    "DEFAULT_SOLUTION_DIM",
    "SPHERE_SHIFT",
    "archive_ranges",
    "clip_solution",
    "linear_projection_measures",
    "rastrigin_objective",
    "sphere_objective",
]


def clip_solution(solution: NDArray[np.floating]) -> FloatArray:
    """Return a float64 solution clipped to the benchmark search box."""
    return np.clip(
        np.asarray(solution, dtype=np.float64),
        -CLIP_BOUND,
        CLIP_BOUND,
    )


def sphere_objective(solution: NDArray[np.floating]) -> float | FloatArray:
    """Return normalized shifted-Sphere objective(s), where 100 is optimal."""
    clipped = clip_solution(solution)
    batch, squeeze = _as_batch(clipped)
    dim = batch.shape[1]
    raw = np.sum(np.square(batch - SPHERE_SHIFT), axis=1)
    worst = float(dim) * (-CLIP_BOUND - SPHERE_SHIFT) ** 2
    objective = np.clip(100.0 * (1.0 - raw / worst), 0.0, 100.0)
    return float(objective[0]) if squeeze else objective


def rastrigin_objective(solution: NDArray[np.floating]) -> float | FloatArray:
    """Return normalized Rastrigin objective(s), where 100 is optimal.

    The denominator ``D * (5.12**2 + 20)`` is a deterministic analytical
    upper bound on the raw Rastrigin value over the clipped search box.
    """
    clipped = clip_solution(solution)
    batch, squeeze = _as_batch(clipped)
    dim = batch.shape[1]
    raw = 10.0 * dim + np.sum(
        np.square(batch) - 10.0 * np.cos(2.0 * np.pi * batch),
        axis=1,
    )
    upper_bound = float(dim) * (CLIP_BOUND**2 + 20.0)
    objective = np.clip(100.0 * (1.0 - raw / upper_bound), 0.0, 100.0)
    return float(objective[0]) if squeeze else objective


def linear_projection_measures(
    solution: NDArray[np.floating],
) -> FloatArray:
    """Return sums of the clipped first and second solution halves."""
    clipped = clip_solution(solution)
    batch, squeeze = _as_batch(clipped)
    dim = batch.shape[1]
    if dim % 2:
        raise ValueError(f"solution dimension must be even, got {dim}")
    midpoint = dim // 2
    measures = np.column_stack(
        (
            np.sum(batch[:, :midpoint], axis=1),
            np.sum(batch[:, midpoint:], axis=1),
        )
    )
    return measures[0] if squeeze else measures


def archive_ranges(solution_dim: int) -> tuple[tuple[float, float], ...]:
    """Return the two linear-projection measure ranges for an even dimension."""
    if solution_dim <= 0 or solution_dim % 2:
        raise ValueError(
            f"solution_dim must be a positive even integer, got {solution_dim}"
        )
    max_measure = float(solution_dim // 2) * CLIP_BOUND
    return ((-max_measure, max_measure), (-max_measure, max_measure))


def _as_batch(solution: FloatArray) -> tuple[FloatArray, bool]:
    if solution.ndim == 1:
        if solution.size == 0:
            raise ValueError("solution must not be empty")
        return solution[np.newaxis, :], True
    if solution.ndim == 2:
        if solution.shape[0] == 0 or solution.shape[1] == 0:
            raise ValueError("solution batch must not be empty")
        return solution, False
    raise ValueError(f"solution must be 1-D or 2-D, got shape {solution.shape}")

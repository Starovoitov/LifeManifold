"""Bounded grid neighbor coordinates for MAP-Elites archive bins (no torus wrap)."""

from __future__ import annotations

__all__ = ["cardinal_neighbors_bounded", "moore_neighbors_bounded"]


def cardinal_neighbors_bounded(
    i: int, j: int, size: int
) -> tuple[tuple[int, int], ...]:
    """Return in-bounds 4-neighbors (N, S, E, W) of ``(i, j)`` on a ``size``×``size`` grid."""
    neighbors: list[tuple[int, int]] = []
    if i > 0:
        neighbors.append((i - 1, j))
    if i + 1 < size:
        neighbors.append((i + 1, j))
    if j > 0:
        neighbors.append((i, j - 1))
    if j + 1 < size:
        neighbors.append((i, j + 1))
    return tuple(neighbors)


def moore_neighbors_bounded(i: int, j: int, size: int) -> tuple[tuple[int, int], ...]:
    """Return in-bounds 8-neighbors (Moore) of ``(i, j)`` on a ``size``×``size`` grid."""
    neighbors: list[tuple[int, int]] = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < size and 0 <= nj < size:
                neighbors.append((ni, nj))
    return tuple(neighbors)

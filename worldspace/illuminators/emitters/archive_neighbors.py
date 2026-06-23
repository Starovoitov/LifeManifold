"""Archive-aware neighbor helpers for MAP-Elites emitters."""

from __future__ import annotations

import numpy as np

from worldspace.illuminators.archive import ArchiveElite, GridArchive
from worldspace.illuminators.archive_protocol import ArchiveProtocol
from worldspace.illuminators.grid_neighbors import moore_neighbors_bounded

__all__ = [
    "min_fitness_elite",
    "moore_neighbor_elites",
    "neighbor_elites",
    "occupied_neighbors",
]


def occupied_neighbors(
    archive: ArchiveProtocol,
    cell_id: int,
) -> list[ArchiveElite]:
    """Return occupied elites in ``archive.neighbors(cell_id)``."""
    elites: list[ArchiveElite] = []
    for neighbor_id in archive.neighbors(cell_id):
        elite = archive.get_cell(neighbor_id)
        if elite is not None:
            elites.append(elite)
    return elites


def min_fitness_elite(archive: ArchiveProtocol) -> ArchiveElite | None:
    """Return the lowest-fitness occupied niche; tie-break by smaller ``cell_id``."""
    best: ArchiveElite | None = None
    best_cell_id: int | None = None
    best_fitness = float("inf")
    for cell_id in range(archive.n_cells):
        elite = archive.get_cell(cell_id)
        if elite is None:
            continue
        if elite.fitness < best_fitness or (
            elite.fitness == best_fitness
            and (best_cell_id is None or cell_id < best_cell_id)
        ):
            best = elite
            best_cell_id = cell_id
            best_fitness = elite.fitness
    return best


def moore_neighbor_elites(
    archive: GridArchive,
    bin_coord: tuple[int, int],
    *,
    rng: np.random.Generator,
    max_count: int,
) -> list[ArchiveElite]:
    """Return up to ``max_count`` elites from occupied Moore neighbors of ``bin_coord``."""
    if max_count < 1:
        return []
    i, j = bin_coord
    resolution = archive.resolution
    neighbors: list[ArchiveElite] = []
    for ni, nj in moore_neighbors_bounded(i, j, resolution):
        elite = archive.get(ni, nj)
        if elite is not None:
            neighbors.append(elite)
    if len(neighbors) <= max_count:
        return neighbors
    indices = rng.choice(len(neighbors), size=max_count, replace=False)
    return [neighbors[int(idx)] for idx in sorted(indices)]


def neighbor_elites(
    archive: ArchiveProtocol,
    cell_id: int,
    *,
    rng: np.random.Generator,
    max_count: int,
) -> list[ArchiveElite]:
    """Return few-shot neighbor elites: Moore on grid, Voronoi on CVT."""
    if max_count < 1:
        return []
    if archive.archive_type == "grid":
        if not isinstance(archive, GridArchive):
            msg = "grid archive_type requires GridArchive instance"
            raise TypeError(msg)
        bin_ij = archive.bin_from_cell_id(cell_id)
        return moore_neighbor_elites(
            archive,
            bin_ij,
            rng=rng,
            max_count=max_count,
        )
    neighbors: list[ArchiveElite] = []
    for neighbor_id in archive.neighbors(cell_id):
        elite = archive.get_cell(neighbor_id)
        if elite is not None:
            neighbors.append(elite)
    if len(neighbors) <= max_count:
        return neighbors
    indices = rng.choice(len(neighbors), size=max_count, replace=False)
    return [neighbors[int(idx)] for idx in sorted(indices)]

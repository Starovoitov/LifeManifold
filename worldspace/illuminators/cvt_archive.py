"""CVT MAP-Elites archive: one elite per Voronoi niche in behavioral space."""

from __future__ import annotations

import numpy as np

from worldspace.illuminators.archive import ArchiveElite, InsertResult
from worldspace.illuminators.cvt import assign_cell_id, voronoi_neighbors

__all__ = ["CvtArchive"]


class CvtArchive:
    """In-memory archive with one elite per CVT centroid in ``[0, 1]^2`` BC space."""

    def __init__(self, centroids: np.ndarray) -> None:
        if centroids.ndim != 2 or centroids.shape[1] != 2:
            msg = f"centroids must have shape (n, 2), got {centroids.shape}"
            raise ValueError(msg)
        n_centroids = int(centroids.shape[0])
        if n_centroids < 1:
            msg = f"n_centroids must be >= 1, got {n_centroids}"
            raise ValueError(msg)

        self._centroids = np.asarray(centroids, dtype=np.float64).copy()
        self._cells: list[ArchiveElite | None] = [None] * n_centroids
        self._neighbors = voronoi_neighbors(self._centroids)

    @property
    def archive_type(self) -> str:
        return "cvt"

    @property
    def n_cells(self) -> int:
        return len(self._cells)

    @property
    def n_centroids(self) -> int:
        return len(self._cells)

    @property
    def centroids(self) -> np.ndarray:
        return self._centroids

    def get(self, cell_id: int) -> ArchiveElite | None:
        """Return the elite at ``cell_id`` or ``None`` if the niche is empty."""
        return self._cells[self._cell_index(cell_id)]

    def is_empty(self, cell_id: int) -> bool:
        return self.get(cell_id) is None

    def filled_count(self) -> int:
        return sum(1 for cell in self._cells if cell is not None)

    def empty_count(self) -> int:
        return len(self._cells) - self.filled_count()

    def cell_center(self, cell_id: int) -> tuple[float, float]:
        """Return the centroid coordinates ``(stability, diversity)`` for ``cell_id``."""
        index = self._cell_index(cell_id)
        return (float(self._centroids[index, 0]), float(self._centroids[index, 1]))

    def neighbors(self, cell_id: int) -> tuple[int, ...]:
        """Return sorted Voronoi neighbor indices for ``cell_id``."""
        index = self._cell_index(cell_id)
        return self._neighbors[index]

    def assign_cell_id(self, stability: float, diversity: float) -> int:
        """Map measured BC to the nearest archive niche."""
        return assign_cell_id(stability, diversity, self._centroids)

    def try_insert(self, elite: ArchiveElite) -> InsertResult:
        """Insert or replace at ``elite.bin[0]``; replace only on strict fitness gain."""
        cell_id = elite.bin[0]
        index = self._cell_index(cell_id)
        current = self._cells[index]
        if current is None:
            self._cells[index] = elite
            return InsertResult(accepted=True, improved=False, rejected=False)
        if elite.fitness > current.fitness:
            self._cells[index] = elite
            return InsertResult(accepted=True, improved=True, rejected=False)
        return InsertResult(accepted=False, improved=False, rejected=True)

    def _cell_index(self, cell_id: int) -> int:
        _validate_cell_id(cell_id, self.n_cells)
        return cell_id


def _validate_cell_id(cell_id: int, n_cells: int) -> None:
    if cell_id < 0 or cell_id >= n_cells:
        msg = f"cell_id {cell_id} out of range for {n_cells} niches"
        raise IndexError(msg)

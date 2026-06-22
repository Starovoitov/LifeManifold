"""Shared archive protocol for grid and CVT MAP-Elites implementations."""

from __future__ import annotations

from typing import Protocol

from worldspace.illuminators.archive import ArchiveElite, InsertResult

__all__ = ["ArchiveProtocol"]


class ArchiveProtocol(Protocol):
    """Unified in-memory archive API for illuminator loop and emitters."""

    @property
    def archive_type(self) -> str:
        """``grid`` or ``cvt``."""
        ...

    @property
    def n_cells(self) -> int:
        """Number of behavioral niches in the archive."""
        ...

    def get_cell(self, cell_id: int) -> ArchiveElite | None:
        """Return the elite at ``cell_id`` or ``None`` when the niche is empty."""
        ...

    def is_empty_cell(self, cell_id: int) -> bool:
        """True when ``cell_id`` has no stored elite."""
        ...

    def filled_count(self) -> int:
        """Number of occupied niches."""
        ...

    def empty_count(self) -> int:
        """Number of unoccupied niches."""
        ...

    def try_insert(self, elite: ArchiveElite) -> InsertResult:
        """Insert or replace using strict fitness improvement per niche."""
        ...

    def cell_center(self, cell_id: int) -> tuple[float, float]:
        """Return BC niche center ``(stability, diversity)`` for ``cell_id``."""
        ...

    def neighbors(self, cell_id: int) -> tuple[int, ...]:
        """Return sorted neighbor niche indices for emitters and target selection."""
        ...

    def assign_cell_id(self, stability: float, diversity: float) -> int:
        """Map measured BC to a flat niche index."""
        ...

    def cell_id_from_bin(self, bin_ij: tuple[int, int]) -> int:
        """Convert an elite ``bin`` tuple to a flat niche index."""
        ...

    def bin_from_cell_id(self, cell_id: int) -> tuple[int, int]:
        """Convert a flat niche index to an elite ``bin`` tuple."""
        ...

"""MAP-Elites grid archive: one elite per behavioral niche."""

from __future__ import annotations

from dataclasses import dataclass

BC_MIN = 0.0
BC_MAX = 1.0
DEFAULT_GRID_RESOLUTION = 40

__all__ = [
    "BC_MAX",
    "BC_MIN",
    "DEFAULT_GRID_RESOLUTION",
    "ArchiveElite",
    "GridArchive",
    "InsertResult",
]


@dataclass
class ArchiveElite:
    """Best candidate stored in one archive cell."""

    bin: tuple[int, int]
    fitness: float


@dataclass(frozen=True)
class InsertResult:
    """Outcome of ``GridArchive.try_insert``."""

    accepted: bool
    improved: bool
    rejected: bool


class GridArchive:
    """In-memory ``resolution x resolution`` archive with fixed BC range [0, 1]."""

    def __init__(self, resolution: int = DEFAULT_GRID_RESOLUTION) -> None:
        if resolution < 1:
            msg = f"resolution must be >= 1, got {resolution}"
            raise ValueError(msg)
        self._resolution = resolution
        self._cells: list[ArchiveElite | None] = [None] * (resolution * resolution)

    @property
    def resolution(self) -> int:
        return self._resolution

    @property
    def bc_min(self) -> float:
        return BC_MIN

    @property
    def bc_max(self) -> float:
        return BC_MAX

    def get(self, i: int, j: int) -> ArchiveElite | None:
        """Return the elite at ``(i, j)`` or ``None`` if the cell is empty."""
        return self._cells[self._cell_index(i, j)]

    def is_empty(self, i: int, j: int) -> bool:
        return self.get(i, j) is None

    def filled_count(self) -> int:
        return sum(1 for cell in self._cells if cell is not None)

    def empty_count(self) -> int:
        return len(self._cells) - self.filled_count()

    def try_insert(self, elite: ArchiveElite) -> InsertResult:
        """Insert or replace the elite at ``elite.bin`` using strict fitness improvement."""
        i, j = elite.bin
        idx = self._cell_index(i, j)
        current = self._cells[idx]
        if current is None:
            self._cells[idx] = elite
            return InsertResult(accepted=True, improved=False, rejected=False)
        if elite.fitness > current.fitness:
            self._cells[idx] = elite
            return InsertResult(accepted=True, improved=True, rejected=False)
        return InsertResult(accepted=False, improved=False, rejected=True)

    def _cell_index(self, i: int, j: int) -> int:
        _validate_bin(i, j, self._resolution)
        return i * self._resolution + j


def _validate_bin(i: int, j: int, resolution: int) -> None:
    if i < 0 or i >= resolution or j < 0 or j >= resolution:
        msg = f"bin ({i}, {j}) out of range for resolution {resolution}"
        raise IndexError(msg)

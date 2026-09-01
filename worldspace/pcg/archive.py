"""10×10 MAP-Elites archive for a PCG Benchmark task."""

from __future__ import annotations

from dataclasses import dataclass

from worldspace.illuminators.archive import InsertResult
from worldspace.illuminators.grid_neighbors import cardinal_neighbors_bounded
from worldspace.pcg.descriptors import PcgBinEdges
from worldspace.pcg.spec import PcgSpec


@dataclass(frozen=True)
class PcgElite:
    bin: tuple[int, int]
    fitness: float
    measures: tuple[float, float]
    spec: PcgSpec
    candidate_id: str
    parent_id: str | None
    emitter_type: str
    playable: bool


class PcgArchive:
    archive_type = "grid"

    def __init__(self, edges: PcgBinEdges) -> None:
        self.edges = edges
        self.resolution = edges.resolution
        self._cells: list[PcgElite | None] = [None] * (self.resolution**2)

    @property
    def n_cells(self) -> int:
        return len(self._cells)

    def filled_count(self) -> int:
        return sum(elite is not None for elite in self._cells)

    def get_cell(self, cell_id: int) -> PcgElite | None:
        return self._cells[cell_id]

    def is_empty_cell(self, cell_id: int) -> bool:
        return self.get_cell(cell_id) is None

    def cell_id_from_bin(self, bin_ij: tuple[int, int]) -> int:
        return bin_ij[0] * self.resolution + bin_ij[1]

    def bin_from_cell_id(self, cell_id: int) -> tuple[int, int]:
        return divmod(cell_id, self.resolution)

    def cell_center(self, cell_id: int) -> tuple[float, float]:
        row, column = self.bin_from_cell_id(cell_id)
        return (
            (row + 0.5) / self.resolution,
            (column + 0.5) / self.resolution,
        )

    def neighbors(self, cell_id: int) -> tuple[int, ...]:
        bin_ij = self.bin_from_cell_id(cell_id)
        return tuple(
            self.cell_id_from_bin(item)
            for item in cardinal_neighbors_bounded(
                bin_ij[0], bin_ij[1], self.resolution
            )
        )

    def try_insert(self, elite: PcgElite) -> InsertResult:
        cell_id = self.cell_id_from_bin(elite.bin)
        current = self._cells[cell_id]
        if current is None:
            self._cells[cell_id] = elite
            return InsertResult(accepted=True, improved=False, rejected=False)
        if elite.fitness > current.fitness:
            self._cells[cell_id] = elite
            return InsertResult(accepted=True, improved=True, rejected=False)
        return InsertResult(accepted=False, improved=False, rejected=True)

    def elites(self) -> list[PcgElite]:
        return [elite for elite in self._cells if elite is not None]

    def occupied_bins(self) -> frozenset[tuple[int, int]]:
        return frozenset(elite.bin for elite in self.elites())

    def clone(self) -> PcgArchive:
        copy = PcgArchive(self.edges)
        copy._cells = list(self._cells)
        return copy

    def coverage(self) -> float:
        return self.filled_count() / float(self.n_cells)

    def qd_score(self) -> float:
        return float(sum(elite.fitness for elite in self.elites()))

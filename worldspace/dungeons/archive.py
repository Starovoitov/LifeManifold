"""Dungeon-specific grid archive and JSONL persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from worldspace.dungeons.spec import DungeonSpec
from worldspace.illuminators.archive import InsertResult
from worldspace.illuminators.grid_neighbors import cardinal_neighbors_bounded


@dataclass(frozen=True)
class DungeonElite:
    bin: tuple[int, int]
    fitness: float
    measures: tuple[float, float]
    spec: DungeonSpec
    candidate_id: str
    parent_id: str | None
    emitter_type: str


class DungeonArchive:
    """Fixed [0,1]² grid archive implementing the shared archive protocol."""

    archive_type = "grid"
    bc_min = 0.0
    bc_max = 1.0

    def __init__(self, resolution: int = 30) -> None:
        if resolution < 1:
            raise ValueError("resolution must be positive")
        self.resolution = resolution
        self._cells: list[DungeonElite | None] = [None] * (resolution**2)

    @property
    def n_cells(self) -> int:
        return len(self._cells)

    def filled_count(self) -> int:
        return sum(elite is not None for elite in self._cells)

    def get_cell(self, cell_id: int) -> DungeonElite | None:
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

    def assign_cell_id(self, first: float, second: float) -> int:
        return self.cell_id_from_bin(self.bin_for_measures((first, second)))

    def bin_for_measures(self, measures: tuple[float, float]) -> tuple[int, int]:
        return tuple(
            min(self.resolution - 1, max(0, int(value * self.resolution)))
            for value in measures
        )  # type: ignore[return-value]

    def try_insert(self, elite: DungeonElite) -> InsertResult:
        cell_id = self.cell_id_from_bin(elite.bin)
        current = self._cells[cell_id]
        if current is None:
            self._cells[cell_id] = elite
            return InsertResult(accepted=True, improved=False, rejected=False)
        if elite.fitness > current.fitness:
            self._cells[cell_id] = elite
            return InsertResult(accepted=True, improved=True, rejected=False)
        return InsertResult(accepted=False, improved=False, rejected=True)

    def elites(self) -> list[DungeonElite]:
        return [elite for elite in self._cells if elite is not None]

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for elite in self.elites():
                handle.write(
                    json.dumps(
                        {
                            "schema_version": "dungeon-1.0",
                            "bin": list(elite.bin),
                            "fitness": elite.fitness,
                            "measures": list(elite.measures),
                            "dungeon_spec": elite.spec.to_json_dict(),
                            "metadata": {
                                "id": elite.candidate_id,
                                "parent_id": elite.parent_id,
                                "emitter_type": elite.emitter_type,
                            },
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
        temporary.replace(path)

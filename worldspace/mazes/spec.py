"""Validated, canonical 16×16 maze representation (walls + start/goal)."""

from __future__ import annotations

import hashlib
import json
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

WALL = "#"
FLOOR = "."
START = "S"
GOAL = "G"
TILES = frozenset((WALL, FLOOR, START, GOAL))

Position = tuple[int, int]


class MazeSpec(BaseModel):
    """Immutable tile rows with strict structural invariants."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    SIZE: ClassVar[int] = 16
    rows: tuple[str, ...]

    @field_validator("rows", mode="before")
    @classmethod
    def _rows_tuple(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("rows must be a list or tuple of strings")
        return tuple(str(row) for row in value)

    @model_validator(mode="after")
    def _validate_layout(self) -> MazeSpec:
        if len(self.rows) != self.SIZE:
            raise ValueError(f"maze must have {self.SIZE} rows")
        if any(len(row) != self.SIZE for row in self.rows):
            raise ValueError(f"every maze row must have length {self.SIZE}")
        unknown = set("".join(self.rows)) - TILES
        if unknown:
            raise ValueError(f"unknown maze tiles: {sorted(unknown)}")
        if any(tile != WALL for tile in self.rows[0] + self.rows[-1]):
            raise ValueError("top and bottom boundaries must be walls")
        if any(row[0] != WALL or row[-1] != WALL for row in self.rows):
            raise ValueError("left and right boundaries must be walls")
        counts = {tile: sum(row.count(tile) for row in self.rows) for tile in TILES}
        if counts[START] != 1 or counts[GOAL] != 1:
            raise ValueError("maze must contain exactly one start and one goal")
        return self

    def position(self, tile: str) -> Position | None:
        for row_index, row in enumerate(self.rows):
            column = row.find(tile)
            if column >= 0:
                return (row_index, column)
        return None

    def tile_at(self, position: Position) -> str:
        return self.rows[position[0]][position[1]]

    def canonical_json(self) -> str:
        return json.dumps(
            {"rows": list(self.rows)},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def candidate_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:16]

    def to_json_dict(self) -> dict[str, object]:
        return {"rows": list(self.rows)}

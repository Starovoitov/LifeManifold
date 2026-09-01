"""PCG Benchmark task pins and canonical integer-grid genotype."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

SOKOBAN_INFO_KEYS = frozenset(
    {"players", "crates", "targets", "content", "heuristic", "solution"}
)
ZELDA_INFO_KEYS = frozenset(
    {
        "regions",
        "players",
        "keys",
        "doors",
        "enemies",
        "player_key",
        "key_door",
        "pk_path",
        "kd_path",
    }
)


@dataclass(frozen=True)
class PcgTask:
    """One PCG Benchmark problem variant. Sokoban and Zelda are one family."""

    problem_name: str
    rows: int
    cols: int
    n_tiles: int
    measure_names: tuple[str, str]
    expected_info_keys: frozenset[str]


SOKOBAN_V0 = PcgTask(
    problem_name="sokoban-v0",
    rows=5,
    cols=5,
    n_tiles=5,
    measure_names=("solution_length", "crates"),
    expected_info_keys=SOKOBAN_INFO_KEYS,
)
ZELDA_V0 = PcgTask(
    problem_name="zelda-v0",
    rows=7,
    cols=11,
    n_tiles=6,
    measure_names=("player_key", "key_door"),
    expected_info_keys=ZELDA_INFO_KEYS,
)


class PcgSpec(BaseModel):
    """Immutable 2D int grid; hash is SHA-256 of compact JSON."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    problem_name: str
    rows: int
    cols: int
    n_tiles: int
    grid: tuple[tuple[int, ...], ...]

    @field_validator("grid", mode="before")
    @classmethod
    def _grid_tuples(cls, value: object) -> tuple[tuple[int, ...], ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("grid must be a 2D list or tuple of ints")
        rows: list[tuple[int, ...]] = []
        for row in value:
            if not isinstance(row, (list, tuple)):
                raise ValueError("each grid row must be a list or tuple")
            rows.append(tuple(int(tile) for tile in row))
        return tuple(rows)

    @model_validator(mode="after")
    def _shape_and_tiles(self) -> PcgSpec:
        if len(self.grid) != self.rows:
            raise ValueError(f"grid must have {self.rows} rows")
        if any(len(row) != self.cols for row in self.grid):
            raise ValueError(f"every grid row must have length {self.cols}")
        unknown = {
            tile
            for row in self.grid
            for tile in row
            if tile < 0 or tile >= self.n_tiles
        }
        if unknown:
            raise ValueError(f"unknown PCG tiles: {sorted(unknown)}")
        return self

    def to_nested_list(self) -> list[list[int]]:
        return [list(row) for row in self.grid]

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_nested_list(), ensure_ascii=True, separators=(",", ":")
        )

    def genotype_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def candidate_hash(self) -> str:
        return self.genotype_sha256()[:16]

    @classmethod
    def from_task_grid(cls, task: PcgTask, grid: object) -> PcgSpec:
        return cls(
            problem_name=task.problem_name,
            rows=task.rows,
            cols=task.cols,
            n_tiles=task.n_tiles,
            grid=grid,  # type: ignore[arg-type]
        )


def try_parse_grid(payload: object, task: PcgTask) -> PcgSpec | None:
    """Parse JSON grid without contacting the PCG evaluator."""
    if isinstance(payload, dict) and "grid" in payload:
        payload = payload["grid"]
    try:
        return PcgSpec.from_task_grid(task, payload)
    except (TypeError, ValueError):
        return None


def hamming_tiles(first: PcgSpec, second: PcgSpec) -> int:
    if first.rows != second.rows or first.cols != second.cols:
        raise ValueError("cannot compare grids of different shape")
    return sum(
        a != b
        for row_a, row_b in zip(first.grid, second.grid, strict=True)
        for a, b in zip(row_a, row_b, strict=True)
    )

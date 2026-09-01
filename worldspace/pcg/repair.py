"""Named PCG-native repair. Identity vs structural_counts.

Not maze solvability_repair. Does not run A*. Count constraints plus a
deterministic 4-connected carve so the relevant sprites share a walkable
component. A* eligibility is a side-effect of counts; solvability is not
guaranteed.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from worldspace.pcg.spec import PcgSpec, hamming_tiles

RepairKind = Literal["identity", "structural_counts"]


class RepairMeta(TypedDict):
    repair_kind: RepairKind
    tiles_changed: int
    astar_eligible: bool | None


SOKOBAN_SOLID = 0
SOKOBAN_EMPTY = 1
SOKOBAN_PLAYER = 2
SOKOBAN_CRATE = 3
SOKOBAN_TARGET = 4

ZELDA_WALL = 0
ZELDA_EMPTY = 1
ZELDA_PLAYER = 2
ZELDA_KEY = 3
ZELDA_DOOR = 4
ZELDA_ENEMY = 5

_NEIGHBORS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_MAX_SOKOBAN_CRATES = 3


def sokoban_astar_eligible(grid: list[list[int]]) -> bool:
    players = _count(grid, SOKOBAN_PLAYER)
    crates = _count(grid, SOKOBAN_CRATE)
    targets = _count(grid, SOKOBAN_TARGET)
    return players == 1 and crates > 0 and crates == targets


def apply_repair(spec: PcgSpec, kind: RepairKind) -> tuple[PcgSpec, RepairMeta]:
    if kind == "identity":
        grid = _mutable(spec)
        return spec, {
            "repair_kind": kind,
            "tiles_changed": 0,
            "astar_eligible": (
                sokoban_astar_eligible(grid)
                if spec.problem_name == "sokoban-v0"
                else None
            ),
        }
    if kind != "structural_counts":
        raise ValueError(f"unknown repair kind {kind!r}")
    grid = _mutable(spec)
    if spec.problem_name == "sokoban-v0":
        _repair_sokoban_counts(grid)
    elif spec.problem_name == "zelda-v0":
        _repair_zelda_counts(grid)
    else:
        raise ValueError(f"no structural_counts repair for {spec.problem_name}")
    repaired = PcgSpec(
        problem_name=spec.problem_name,
        rows=spec.rows,
        cols=spec.cols,
        n_tiles=spec.n_tiles,
        grid=tuple(tuple(int(cell) for cell in row) for row in grid),
    )
    return repaired, {
        "repair_kind": kind,
        "tiles_changed": hamming_tiles(spec, repaired),
        "astar_eligible": (
            sokoban_astar_eligible(grid) if spec.problem_name == "sokoban-v0" else None
        ),
    }


def _mutable(spec: PcgSpec) -> list[list[int]]:
    return [list(row) for row in spec.grid]


def _count(grid: list[list[int]], tile: int) -> int:
    return sum(cell == tile for row in grid for cell in row)


def _positions(grid: list[list[int]], tile: int) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == tile:
                found.append((r, c))
    return found


def _set_exactly(
    grid: list[list[int]],
    tile: int,
    n: int,
    *,
    filler: int,
    protected: set[tuple[int, int]],
) -> None:
    rows, cols = len(grid), len(grid[0])
    coords = _positions(grid, tile)

    def _rank(cell: tuple[int, int]) -> tuple[int, int, int, int]:
        r, c = cell
        interior = min(r, c, rows - 1 - r, cols - 1 - c)
        return (0 if cell in protected else 1, -interior, r, c)

    if len(coords) > n:
        ordered = sorted(coords, key=_rank)
        keep = ordered[:n]
        for r, c in ordered[n:]:
            grid[r][c] = filler
        protected.update(keep)
        return
    protected.update(coords)
    needed = n - len(coords)
    candidates = [
        (r, c) for r in range(rows) for c in range(cols) if (r, c) not in protected
    ]
    candidates.sort(key=_rank)
    for r, c in candidates:
        if needed == 0:
            break
        grid[r][c] = tile
        protected.add((r, c))
        needed -= 1
    if needed > 0:
        raise RuntimeError(f"could not place {n} tiles of type {tile}")


def _carve(
    grid: list[list[int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    empty: int,
    *,
    blocked: frozenset[int] = frozenset(),
) -> None:
    if start == goal:
        return
    rows, cols = len(grid), len(grid[0])
    queue = [start]
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    head = 0
    found = False
    while head < len(queue):
        r, c = queue[head]
        head += 1
        if (r, c) == goal:
            found = True
            break
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                continue
            nxt = (nr, nc)
            if nxt in parent:
                continue
            tile = grid[nr][nc]
            if nxt != goal and tile in blocked:
                continue
            parent[nxt] = (r, c)
            queue.append(nxt)
    if not found:
        return
    cur: tuple[int, int] | None = goal
    while cur is not None:
        r, c = cur
        if grid[r][c] == 0:
            grid[r][c] = empty
        cur = parent[cur]


def _repair_sokoban_counts(grid: list[list[int]]) -> None:
    protected: set[tuple[int, int]] = set()
    _set_exactly(grid, SOKOBAN_PLAYER, 1, filler=SOKOBAN_EMPTY, protected=protected)
    crates = min(max(_count(grid, SOKOBAN_CRATE), 1), _MAX_SOKOBAN_CRATES)
    _set_exactly(grid, SOKOBAN_CRATE, crates, filler=SOKOBAN_EMPTY, protected=protected)
    _set_exactly(
        grid, SOKOBAN_TARGET, crates, filler=SOKOBAN_EMPTY, protected=protected
    )
    player = _positions(grid, SOKOBAN_PLAYER)[0]
    blocked = frozenset({SOKOBAN_CRATE})
    for pos in _positions(grid, SOKOBAN_CRATE) + _positions(grid, SOKOBAN_TARGET):
        _carve(grid, player, pos, SOKOBAN_EMPTY, blocked=blocked)


def _repair_zelda_counts(grid: list[list[int]]) -> None:
    protected: set[tuple[int, int]] = set()
    _set_exactly(grid, ZELDA_PLAYER, 1, filler=ZELDA_EMPTY, protected=protected)
    _set_exactly(grid, ZELDA_KEY, 1, filler=ZELDA_EMPTY, protected=protected)
    _set_exactly(grid, ZELDA_DOOR, 1, filler=ZELDA_EMPTY, protected=protected)
    _set_exactly(grid, ZELDA_ENEMY, 3, filler=ZELDA_EMPTY, protected=protected)
    player = _positions(grid, ZELDA_PLAYER)[0]
    key = _positions(grid, ZELDA_KEY)[0]
    door = _positions(grid, ZELDA_DOOR)[0]
    _carve(grid, player, key, ZELDA_EMPTY, blocked=frozenset({ZELDA_DOOR}))
    _carve(grid, key, door, ZELDA_EMPTY)

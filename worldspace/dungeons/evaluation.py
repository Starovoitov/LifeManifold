"""Deterministic planning and objective/measure evaluation for dungeons."""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass

import numpy as np

from worldspace.dungeons.spec import (
    DOOR,
    GOAL,
    HAZARD,
    KEY,
    START,
    WALL,
    DungeonSpec,
    Position,
)

ROLLOUTS = 16
HAZARD_BLOCK_PROBABILITY = 0.35


@dataclass(frozen=True)
class DungeonEvaluation:
    fitness: float
    measures: tuple[float, float]
    solvable: bool
    shortest_path: int | None
    robustness: float
    reachable_ratio: float
    branching_density: float


def shortest_path_length(
    spec: DungeonSpec,
    *,
    blocked_hazards: frozenset[Position] = frozenset(),
) -> int | None:
    """Find the shortest stateful path; a door requires collecting the key."""
    start = spec.position(START)
    goal = spec.position(GOAL)
    assert start is not None and goal is not None
    queue: list[tuple[int, int, Position, bool]] = []
    order = 0
    heapq.heappush(queue, (0, order, start, False))
    best: dict[tuple[Position, bool], int] = {(start, False): 0}
    while queue:
        distance, _, position, has_key = heapq.heappop(queue)
        if position == goal:
            return distance
        if distance != best.get((position, has_key)):
            continue
        for neighbor in _neighbors(position):
            tile = spec.tile_at(neighbor)
            if tile == WALL or neighbor in blocked_hazards:
                continue
            if tile == DOOR and not has_key:
                continue
            next_has_key = has_key or tile == KEY
            next_distance = distance + (3 if tile == HAZARD else 1)
            state = (neighbor, next_has_key)
            if next_distance >= best.get(state, 10**9):
                continue
            best[state] = next_distance
            order += 1
            heapq.heappush(
                queue,
                (next_distance, order, neighbor, next_has_key),
            )
    return None


def evaluate_dungeon(spec: DungeonSpec, *, seed: int = 0) -> DungeonEvaluation:
    """Return deterministic fitness and two normalized behavior descriptors."""
    path = shortest_path_length(spec)
    reachable, branching = _reachable_structure(spec)
    traversable = sum(tile != WALL for row in spec.rows for tile in row)
    reachable_ratio = float(len(reachable)) / float(traversable) if traversable else 0.0
    branching_density = float(branching) / float(len(reachable)) if reachable else 0.0
    path_measure = float(np.clip((float(path or 0) - 10.0) / 40.0, 0.0, 1.0))
    branching_measure = float(np.clip((branching_density - 0.25) / 0.50, 0.0, 1.0))
    if path is None:
        return DungeonEvaluation(
            fitness=0.0,
            measures=(0.0, branching_measure),
            solvable=False,
            shortest_path=None,
            robustness=0.0,
            reachable_ratio=reachable_ratio,
            branching_density=branching_density,
        )

    hazards = [
        (row_index, column)
        for row_index, row in enumerate(spec.rows)
        for column, tile in enumerate(row)
        if tile == HAZARD
    ]
    solved = 0
    for rollout in range(ROLLOUTS):
        rng = np.random.default_rng(_rollout_seed(spec, seed, rollout))
        blocked = frozenset(
            position
            for position in hazards
            if float(rng.random()) < HAZARD_BLOCK_PROBABILITY
        )
        solved += shortest_path_length(spec, blocked_hazards=blocked) is not None
    robustness = solved / float(ROLLOUTS)
    fitness = float(
        np.clip(
            0.55 * robustness + 0.30 * path_measure + 0.15 * reachable_ratio,
            0.0,
            1.0,
        )
    )
    return DungeonEvaluation(
        fitness=fitness,
        measures=(path_measure, branching_measure),
        solvable=True,
        shortest_path=path,
        robustness=robustness,
        reachable_ratio=reachable_ratio,
        branching_density=branching_density,
    )


def _neighbors(position: Position) -> tuple[Position, ...]:
    row, column = position
    return (
        (row - 1, column),
        (row + 1, column),
        (row, column - 1),
        (row, column + 1),
    )


def _reachable_structure(spec: DungeonSpec) -> tuple[set[Position], int]:
    start = spec.position(START)
    assert start is not None
    reachable = {start}
    stack = [start]
    while stack:
        position = stack.pop()
        for neighbor in _neighbors(position):
            if neighbor in reachable or spec.tile_at(neighbor) == WALL:
                continue
            reachable.add(neighbor)
            stack.append(neighbor)
    branching = 0
    for position in reachable:
        degree = sum(
            neighbor in reachable and spec.tile_at(neighbor) != WALL
            for neighbor in _neighbors(position)
        )
        branching += degree >= 3
    return reachable, branching


def _rollout_seed(spec: DungeonSpec, seed: int, rollout: int) -> int:
    digest = hashlib.sha256(
        f"{spec.candidate_hash()}:{seed}:{rollout}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import math as ws_math
from .spec import WorldSpec


@dataclass
class SimulationResult:
    """Raw simulation outputs required for downstream metric calculation."""

    world: WorldSpec
    density_series: list[float]
    alive_series: list[int]
    death_ages: list[int]
    history: list[np.ndarray]


def run_world(world: WorldSpec) -> SimulationResult:
    """Run one world simulation and collect trajectory statistics."""
    rng = np.random.default_rng(world.seed)
    n = world.grid_size
    life = (rng.random((n, n)) < 0.2).astype(np.uint8)
    food = (rng.random((n, n)) < world.resource_regen).astype(np.uint8)
    ages = np.zeros((n, n), dtype=np.int16)

    density_series: list[float] = []
    alive_series: list[int] = []
    death_ages: list[int] = []
    history: list[np.ndarray] = []

    for _ in range(world.steps):
        neighbors = ws_math.neighbor_count(life)
        born = ((life == 0) & np.isin(neighbors, world.birth)).astype(np.uint8)
        survive = ((life == 1) & np.isin(neighbors, world.survival)).astype(np.uint8)
        next_life = np.maximum(born, survive)

        if world.noise > 0:
            flip = rng.random((n, n)) < world.noise
            next_life = np.where(flip, 1 - next_life, next_life).astype(np.uint8)

        if world.predation > 0:
            exposure = neighbors / 8.0
            predation_deaths = rng.random((n, n)) < (world.predation * exposure * next_life)
            next_life = np.where(predation_deaths, 0, next_life).astype(np.uint8)

        food = np.where(rng.random((n, n)) < world.resource_regen, 1, food).astype(np.uint8)
        feed_bonus = ((food == 1) & (next_life == 1)).astype(np.uint8)
        food = np.where(feed_bonus == 1, 0, food).astype(np.uint8)

        died_now = (life == 1) & (next_life == 0)
        death_ages.extend(ages[died_now].tolist())

        ages = np.where(next_life == 1, ages + 1 + feed_bonus, 0).astype(np.int16)
        life = next_life

        density_series.append(float(life.mean()))
        alive_series.append(int(life.sum()))
        history.append(life.copy())

    return SimulationResult(
        world=world,
        density_series=density_series,
        alive_series=alive_series,
        death_ages=death_ages,
        history=history,
    )

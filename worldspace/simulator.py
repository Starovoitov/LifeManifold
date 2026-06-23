from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from typing import TextIO

import numpy as np

from . import math as ws_math
from .metrics import (
    WorldMetrics,
    metrics_vector_to_dict,
    multi_objective_edge_of_chaos_indicator,
)
from worldspace.simulator_perf import (
    DEFAULT_SIMULATOR_PERFORMANCE,
    SimulatorPerformanceOptions,
    effective_numba_enabled,
)
from .specs.spec import WorldSpec


@dataclass
class SimulationResult:
    """Outcome of one simulation: metrics and optional final grids for plotting."""

    world: WorldSpec
    metrics: WorldMetrics
    final_life: np.ndarray | None = None
    final_food: np.ndarray | None = None
    early_extinct: bool = False


def run_world(
    world: WorldSpec,
    *,
    ca_step_trace_file: TextIO | None = None,
    ca_step_trace_yield_index: int = 0,
    early_extinction_step: int | None = None,
    performance: SimulatorPerformanceOptions | None = None,
) -> SimulationResult:
    """Run one world; metrics use online accumulators.

    If ``ca_step_trace_file`` is set, append one JSON object per CA timestep after that
    step (``yield_index``, ``ca_step``, ``metrics``). Used by the pipeline batch path only;
    other ``run_world`` callers omit it.

    When ``early_extinction_step`` is set (illuminator: ``200``), stop as soon as
    ``life.mean() == 0`` at timestep ``t`` with ``0 <= t < early_extinction_step``
    (``t = 0`` is post-init, before the first CA step). ``None`` keeps legacy full runs.

    ``performance`` selects optional numba / verify paths (scheduler YAML). Default is
    standard numpy. Per-step trace forces numpy regardless of ``numba_simulator``.
    """
    perf = performance if performance is not None else DEFAULT_SIMULATOR_PERFORMANCE
    _ = effective_numba_enabled(perf, ca_step_trace=ca_step_trace_file is not None)
    if perf.verify_against_reference:
        pass  # dual-run numpy vs numba when numba path is enabled
    rng = np.random.default_rng(world.seed)
    life, food, ages = _initial_grids(rng, world)

    density_mean = 0.0
    density_m2 = 0.0
    density_n = 0
    death_age_sum = 0
    death_count = 0
    density_tail: deque[float] = deque(maxlen=ws_math.OSCILLATION_DENSITY_WINDOW)
    early_extinct = False

    run_ca_loop = True
    if early_extinction_step is not None and float(life.mean()) == 0.0:
        run_ca_loop = False
        early_extinct = True
        d = float(life.mean())
        density_tail.append(d)
        density_mean, density_m2, density_n = _welford_append(
            d, density_mean, density_m2, density_n
        )

    if run_ca_loop:
        birth_mask, survival_mask = ws_math.rule_count_masks(
            world.birth, world.survival
        )
        for step in range(world.steps):
            neighbors = ws_math.neighbor_count(life)
            next_life = _next_life_from_rules(
                rng, life, neighbors, birth_mask, survival_mask, world
            )
            food, feed_bonus = _tick_food(rng, food, next_life, world.resource_regen)

            died_now = (life == 1) & (next_life == 0)
            death_age_sum, death_count = _accumulate_deaths(
                died_now, ages, death_age_sum, death_count
            )

            ages = np.where(next_life == 1, ages + 1 + feed_bonus, 0).astype(np.int16)
            life = next_life

            d = float(life.mean())
            density_tail.append(d)
            density_mean, density_m2, density_n = _welford_append(
                d, density_mean, density_m2, density_n
            )

            if ca_step_trace_file is not None:
                snap = _metrics_from_final_state(
                    life,
                    food,
                    density_mean,
                    density_m2,
                    density_n,
                    density_tail,
                    death_age_sum,
                    death_count,
                )
                row = {
                    "yield_index": ca_step_trace_yield_index,
                    "ca_step": step,
                    "metrics": metrics_vector_to_dict(snap.as_vector()),
                }
                ca_step_trace_file.write(json.dumps(row, ensure_ascii=True) + "\n")
                ca_step_trace_file.flush()

            t = step + 1
            if (
                early_extinction_step is not None
                and t < early_extinction_step
                and float(life.mean()) == 0.0
            ):
                early_extinct = True
                break

    metrics = _metrics_from_final_state(
        life,
        food,
        density_mean,
        density_m2,
        density_n,
        density_tail,
        death_age_sum,
        death_count,
    )
    return SimulationResult(
        world=world,
        metrics=metrics,
        final_life=life.copy(),
        final_food=food.copy(),
        early_extinct=early_extinct,
    )


def _initial_grids(
    rng: np.random.Generator, world: WorldSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build initial life, food, and age grids from ``world`` using ``rng``."""
    n = world.grid_size
    life = (rng.random((n, n)) < 0.2).astype(np.uint8)
    food = (rng.random((n, n)) < world.resource_regen).astype(np.uint8)
    ages = np.zeros((n, n), dtype=np.int16)
    return life, food, ages


def _next_life_from_rules(
    rng: np.random.Generator,
    life: np.ndarray,
    neighbors: np.ndarray,
    birth_mask: np.ndarray,
    survival_mask: np.ndarray,
    world: WorldSpec,
) -> np.ndarray:
    """Apply birth/survival rules, stochastic noise, and predation for one sub-step."""
    born = ((life == 0) & birth_mask[neighbors]).astype(np.uint8)
    survive = ((life == 1) & survival_mask[neighbors]).astype(np.uint8)
    next_life = np.maximum(born, survive)
    if world.noise > 0:
        flip = rng.random(life.shape) < world.noise
        next_life = np.where(flip, 1 - next_life, next_life).astype(np.uint8)
    if world.predation > 0:
        exposure = neighbors / 8.0
        predation_deaths = rng.random(life.shape) < (
            world.predation * exposure * next_life
        )
        next_life = np.where(predation_deaths, 0, next_life).astype(np.uint8)
    return next_life


def _tick_food(
    rng: np.random.Generator,
    food: np.ndarray,
    next_life: np.ndarray,
    resource_regen: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Regenerate food where random hits succeed; return updated food and feed-on-cell mask."""
    food = np.where(rng.random(food.shape) < resource_regen, 1, food).astype(np.uint8)
    feed_bonus = ((food == 1) & (next_life == 1)).astype(np.uint8)
    food = np.where(feed_bonus == 1, 0, food).astype(np.uint8)
    return food, feed_bonus


def _accumulate_deaths(
    died_now: np.ndarray,
    ages: np.ndarray,
    death_age_sum: int,
    death_count: int,
) -> tuple[int, int]:
    """Add sum of ages at death and death count for cells that were alive and now dead."""
    if not np.any(died_now):
        return death_age_sum, death_count
    return death_age_sum + int(ages[died_now].sum()), death_count + int(died_now.sum())


def _welford_append(
    value: float,
    mean: float,
    m2: float,
    n: int,
) -> tuple[float, float, int]:
    """One Welford step for running mean and sum of squared deviations (``m2``)."""
    n += 1
    delta = value - mean
    mean += delta / n
    m2 += delta * (value - mean)
    return mean, m2, n


def _metrics_from_final_state(
    life: np.ndarray,
    food: np.ndarray,
    density_mean: float,
    density_m2: float,
    density_n: int,
    density_tail: deque[float],
    death_age_sum: int,
    death_count: int,
) -> WorldMetrics:
    """Aggregate scalar metrics from the final grids and online statistics."""
    density_std = (
        float(np.sqrt(density_m2 / max(density_n - 1, 1))) if density_n > 1 else 0.0
    )
    density_mean_val = density_mean
    entropy = ws_math.binary_entropy(density_mean_val)
    stability = float(
        np.clip(1.0 - (density_std / (density_mean_val + 1e-6)), 0.0, 1.0)
    )
    avg_lifespan = float(death_age_sum / death_count) if death_count > 0 else 0.0
    osc_series = np.asarray(density_tail, dtype=float)
    oscillation = ws_math.oscillation(osc_series)
    diversity = ws_math.pattern_diversity_from_frame(life.copy())
    final_density = float(life.mean())
    extinction_penalty = float(np.clip(1.0 - final_density, 0.0, 1.0))
    mo_eoc = multi_objective_edge_of_chaos_indicator(
        entropy=entropy,
        stability=stability,
        diversity=diversity,
        oscillation_score=oscillation,
        average_lifespan=avg_lifespan,
        extinction_penalty=extinction_penalty,
    )
    topo_if = ws_math.topology_interface_index(life)
    topo_win = ws_math.topology_window_heterogeneity(life)
    comp = ws_math.compressibility_score_joint(life, food)
    eco_h = ws_math.ecology_state_entropy_norm(life, food)
    eco_adj = ws_math.ecology_resource_adjacency(life, food)
    return WorldMetrics(
        entropy=entropy,
        stability=stability,
        average_lifespan=avg_lifespan,
        density_mean=density_mean_val,
        oscillation_score=oscillation,
        diversity=diversity,
        mo_eoc_indicator=mo_eoc,
        topology_interface_index=topo_if,
        topology_window_heterogeneity=topo_win,
        compressibility_score=comp,
        ecology_state_entropy_norm=eco_h,
        ecology_resource_adjacency=eco_adj,
    )

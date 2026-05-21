"""Cached world simulation for the dashboard (sole entry point for ``run_world``)."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import streamlit as st

from dashboard.utils.bootstrap import ensure_repo_on_path
from dashboard.utils.config import load_config
from dashboard.utils.data_processing import (
    canonical_world_spec_hash,
    world_spec_from_dict,
)

ensure_repo_on_path()

from worldspace import math as ws_math
from worldspace.illuminators.evaluation import (
    ILLUMINATOR_MIN_STEPS,
    apply_canonical_seed,
)
from worldspace.simulator import SimulationResult, run_world
from worldspace.specs.spec import WorldSpec

MAP_ELITES_EARLY_EXTINCTION_STEP = 200

__all__ = [
    "WorldMaps",
    "maps_from_result",
    "prepare_world_spec_for_run",
    "run_and_cache_world",
    "run_and_cache_world_from_dict",
    "run_world_for_spec_dict",
]


@dataclass(frozen=True)
class WorldMaps:
    """Spatial maps derived from a simulation result."""

    boundary: np.ndarray
    heterogeneity: np.ndarray
    food_neighbor: np.ndarray


def prepare_world_spec_for_run(spec: WorldSpec) -> WorldSpec:
    """Apply illuminator step floor before canonical seed and simulation."""
    prepared = replace(spec)
    prepared.steps = max(prepared.steps, ILLUMINATOR_MIN_STEPS)
    return prepared


def run_world_for_spec_dict(world_spec: dict[str, Any]) -> SimulationResult:
    """Run a world simulation without Streamlit cache (tests and helpers)."""
    spec = world_spec_from_dict(world_spec)
    prepared = prepare_world_spec_for_run(spec)
    apply_canonical_seed(prepared)
    return run_world(prepared, early_extinction_step=MAP_ELITES_EARLY_EXTINCTION_STEP)


def run_and_cache_world_from_dict(
    world_spec: dict[str, Any],
    *,
    max_cache_size: int = 32,
) -> SimulationResult:
    """Serialize ``world_spec``, hash it, and return a cached ``run_world`` result."""
    del max_cache_size
    spec_hash = canonical_world_spec_hash(world_spec)
    payload = json.dumps(world_spec, sort_keys=True)
    return run_and_cache_world(spec_hash, payload)


def run_and_cache_world(
    world_spec_hash: str,
    world_spec_json: str,
) -> SimulationResult:
    """Cached ``run_world`` keyed by canonical spec hash and JSON payload."""
    spec = world_spec_from_dict(json.loads(world_spec_json))
    prepared = prepare_world_spec_for_run(spec)
    apply_canonical_seed(prepared)
    return run_world(prepared, early_extinction_step=MAP_ELITES_EARLY_EXTINCTION_STEP)


def maps_from_result(result: SimulationResult) -> WorldMaps:
    """Build topology and ecology maps from final life/food grids."""
    life = result.final_life
    food = result.final_food
    if life is None or food is None:
        msg = "SimulationResult must include final_life and final_food for maps"
        raise ValueError(msg)
    if life.shape != food.shape:
        msg = "life and food shapes must match"
        raise ValueError(msg)
    return WorldMaps(
        boundary=ws_math.topology_interface_strength_map(life),
        heterogeneity=ws_math.topology_2x2_heterogeneity_map(life),
        food_neighbor=ws_math.food_neighbor_fraction_map(food),
    )


def _simulation_cache_size() -> int:
    cfg = load_config()
    performance = cfg.get("performance")
    if isinstance(performance, dict):
        return int(performance.get("simulation_cache_size", 32))
    return 32


_SIM_CACHE_SIZE = _simulation_cache_size()

run_and_cache_world = st.cache_data(
    max_entries=_SIM_CACHE_SIZE,
    show_spinner=False,
)(run_and_cache_world)

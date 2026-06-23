"""MAP-Elites candidate evaluation: seed, simulation, measures, fitness, binning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import numpy as np

from worldspace.illuminators.archive_protocol import ArchiveProtocol
from worldspace.metrics import WorldMetrics
from worldspace.simulator import run_world
from worldspace.simulator_perf import SimulatorPerformanceOptions
from worldspace.specs.spec import WorldSpec

__all__ = [
    "MEASURE_KEYS",
    "EvalResult",
    "ILLUMINATOR_MIN_STEPS",
    "SimulationOutcome",
    "apply_canonical_seed",
    "bin_center",
    "bin_edges",
    "bin_index",
    "bin_index_from_measures",
    "canonical_seed",
    "compute_fitness",
    "eval_result_from_simulation",
    "evaluate_candidate",
    "extinction_probability",
    "measures_from_metrics",
    "assign_cell_for_archive",
    "simulate_candidate",
    "topology_complexity",
]

MEASURE_KEYS: tuple[str, ...] = ("stability", "diversity")
ILLUMINATOR_MIN_STEPS = 200

_CANONICAL_JSON_KWARGS = {"sort_keys": True, "separators": (",", ":")}


@dataclass
class EvalResult:
    """Outcome of evaluating one illuminator candidate."""

    world_spec: WorldSpec
    metrics: WorldMetrics
    measures: dict[str, float]
    fitness: float
    bin: tuple[int, int]
    early_extinct: bool


@dataclass
class SimulationOutcome:
    """Result of ``run_world`` for one candidate before archive binning."""

    world_spec: WorldSpec
    metrics: WorldMetrics
    measures: dict[str, float]
    fitness: float
    early_extinct: bool


def canonical_seed(world_spec: WorldSpec) -> int:
    """Derive a deterministic 32-bit seed from the canonical world spec."""
    digest = hashlib.sha256(_canonical_payload(world_spec).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**32)


def apply_canonical_seed(world_spec: WorldSpec) -> int:
    """Set ``world_spec.seed`` from the canonical hash and return it."""
    seed = canonical_seed(world_spec)
    world_spec.seed = seed
    return seed


def measures_from_metrics(metrics: WorldMetrics) -> dict[str, float]:
    """Behavioral coordinates for binning and archive JSONL ``measures``."""
    return {
        "stability": _clip_unit(metrics.stability),
        "diversity": _clip_unit(metrics.diversity),
    }


def topology_complexity(metrics: WorldMetrics) -> float:
    """Topology proxy used only in the fitness sum; not a behavioral axis."""
    raw = (
        0.5 * metrics.topology_interface_index
        + 0.5 * metrics.topology_window_heterogeneity
    )
    return _clip_unit(raw)


def extinction_probability(final_density: float) -> float:
    """``clip(1.0 - final_density, 0, 1)`` from final life grid."""
    return _clip_unit(1.0 - final_density)


def compute_fitness(
    metrics: WorldMetrics,
    measures: dict[str, float],
    *,
    early_extinct: bool,
    final_density: float,
) -> float:
    """Illuminator interestingness; ``0.0`` when ``early_extinct``."""
    if early_extinct:
        return 0.0
    ext_p = extinction_probability(final_density)
    return _clip_unit(
        0.45 * measures["diversity"]
        + 0.25 * (1.0 - ext_p)
        + 0.20 * _clip_unit(metrics.oscillation_score)
        + 0.10 * topology_complexity(metrics)
    )


def bin_edges(resolution: int) -> np.ndarray:
    """BC bin boundaries on ``[0.0, 1.0]`` (length ``resolution + 1``)."""
    return np.linspace(0.0, 1.0, resolution + 1)


def bin_center(i: int, j: int, resolution: int) -> tuple[float, float]:
    """Midpoint of archive cell ``(i, j)`` in stability / diversity coordinates."""
    edges = bin_edges(resolution)
    return (
        float((edges[i] + edges[i + 1]) / 2.0),
        float((edges[j] + edges[j + 1]) / 2.0),
    )


def bin_index(stability: float, diversity: float, resolution: int) -> tuple[int, int]:
    """Map BC values to archive cell indices."""
    edges = bin_edges(resolution)
    s = _clip_unit(stability)
    d = _clip_unit(diversity)
    i = int(np.minimum(np.searchsorted(edges, s, side="right") - 1, resolution - 1))
    j = int(np.minimum(np.searchsorted(edges, d, side="right") - 1, resolution - 1))
    return (i, j)


def bin_index_from_measures(
    measures: dict[str, float], resolution: int
) -> tuple[int, int]:
    """Bin from JSONL-style ``measures`` dict."""
    return bin_index(measures["stability"], measures["diversity"], resolution)


def assign_cell_for_archive(
    measures: dict[str, float],
    archive: ArchiveProtocol,
) -> int:
    """Map measured BC to a flat niche index via archive-specific assignment."""
    return archive.assign_cell_id(measures["stability"], measures["diversity"])


def evaluate_candidate(
    world_spec: WorldSpec,
    *,
    resolution: int = 50,
    archive: ArchiveProtocol | None = None,
    early_extinction_step: int = 200,
    enforce_min_steps: bool = True,
    performance: SimulatorPerformanceOptions | None = None,
) -> EvalResult:
    """Run one candidate: canonical seed, simulation, measures, fitness, and bin."""
    simulation = simulate_candidate(
        world_spec,
        early_extinction_step=early_extinction_step,
        enforce_min_steps=enforce_min_steps,
        performance=performance,
    )
    return eval_result_from_simulation(
        simulation,
        resolution=resolution,
        archive=archive,
    )


def simulate_candidate(
    world_spec: WorldSpec,
    *,
    early_extinction_step: int = 200,
    enforce_min_steps: bool = True,
    performance: SimulatorPerformanceOptions | None = None,
) -> SimulationOutcome:
    """Run simulation only (no archive binning); safe for worker processes."""
    spec = replace(world_spec)
    if enforce_min_steps:
        spec.steps = max(spec.steps, ILLUMINATOR_MIN_STEPS)
    apply_canonical_seed(spec)
    simulation = run_world(
        spec,
        early_extinction_step=early_extinction_step,
        performance=performance,
    )
    if simulation.final_life is None:
        msg = "run_world did not return final_life"
        raise RuntimeError(msg)
    measures = measures_from_metrics(simulation.metrics)
    final_density = float(simulation.final_life.mean())
    fitness = compute_fitness(
        simulation.metrics,
        measures,
        early_extinct=simulation.early_extinct,
        final_density=final_density,
    )
    return SimulationOutcome(
        world_spec=spec,
        metrics=simulation.metrics,
        measures=measures,
        fitness=fitness,
        early_extinct=simulation.early_extinct,
    )


def eval_result_from_simulation(
    simulation: SimulationOutcome,
    *,
    resolution: int,
    archive: ArchiveProtocol | None = None,
) -> EvalResult:
    """Attach archive/grid bin indices to a simulation outcome."""
    if archive is not None:
        cell_id = assign_cell_for_archive(simulation.measures, archive)
        bin_ij = archive.bin_from_cell_id(cell_id)
    else:
        bin_ij = bin_index_from_measures(simulation.measures, resolution)
    return EvalResult(
        world_spec=simulation.world_spec,
        metrics=simulation.metrics,
        measures=simulation.measures,
        fitness=simulation.fitness,
        bin=bin_ij,
        early_extinct=simulation.early_extinct,
    )


def _canonical_payload(world_spec: WorldSpec) -> str:
    return json.dumps(world_spec.to_canonical_dict(), **_CANONICAL_JSON_KWARGS)


def _clip_unit(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))

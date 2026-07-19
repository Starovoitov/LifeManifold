"""Bridge LifeManifold genomes / evaluation to pyribs CMA-ME/MAE (B2 / RQ4).

T0 locks: genetic 21-D genotype, BC ``(stability, diversity)`` on ``[0, 1]^2``,
grid ``50×50``, illuminator fitness via ``evaluate_candidate`` (surrogate off).

CMA proposes continuous ``theta``; ``decode_genome`` **rints** the 18 rule bits
(documented continuous relaxation of the genetic genotype).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from worldspace.illuminators.emitters.genetics import (
    GENOME_SIZE,
    DecodeMode,
    decode_genome,
    encode_world,
)
from worldspace.illuminators.evaluation import (
    ILLUMINATOR_MIN_STEPS,
    EvalResult,
    bin_index,
    evaluate_candidate,
    eval_result_from_simulation,
    simulate_candidate,
)
from worldspace.illuminators.parallel_eval import (
    ParallelEvalPool,
    evaluate_batch_parallel,
    parallel_eval_context,
)
from worldspace.simulator_perf import (
    DEFAULT_SIMULATOR_PERFORMANCE,
    SimulatorPerformanceOptions,
    effective_parallel_workers,
)
from worldspace.specs.spec import WorldSpec
from worldspace.specs.world_param_bounds import (
    NOISE_MAX,
    NOISE_MIN,
    PREDATION_MAX,
    PREDATION_MIN,
    RESOURCE_REGEN_MAX,
    RESOURCE_REGEN_MIN,
    clip_genome_float_params,
)

__all__ = [
    "ARCHIVE_DIMS",
    "ARCHIVE_RANGES",
    "GENOME_SIZE",
    "DecodeMode",
    "MEASURE_ORDER",
    "PyribsEvalBatch",
    "PyribsEvalKnobs",
    "coverage_pct",
    "evaluate_solution",
    "evaluate_solutions_batch",
    "flat_cell_index",
    "mean_best_fitness",
    "qd_score",
    "measures_vector",
    "mid_bounds_x0",
    "solution_to_world_spec",
    "world_spec_to_solution",
]

ARCHIVE_DIMS: tuple[int, int] = (50, 50)
ARCHIVE_RANGES: tuple[tuple[float, float], tuple[float, float]] = (
    (0.0, 1.0),
    (0.0, 1.0),
)
MEASURE_ORDER: tuple[str, str] = ("stability", "diversity")
_N_CELLS = ARCHIVE_DIMS[0] * ARCHIVE_DIMS[1]


@dataclass(frozen=True)
class PyribsEvalKnobs:
    """Illuminator-matching simulation knobs for pyribs asks."""

    grid_size: int = 50
    steps: int = ILLUMINATOR_MIN_STEPS
    resolution: int = 50
    early_extinction_step: int = 200
    enforce_min_steps: bool = True
    performance: SimulatorPerformanceOptions = DEFAULT_SIMULATOR_PERFORMANCE
    decode_mode: DecodeMode = "rint"
    eval_seed: int = 0


@dataclass(frozen=True)
class PyribsEvalBatch:
    """Batch evaluation outputs for pyribs ``tell``."""

    objectives: np.ndarray
    measures: np.ndarray
    results: tuple[EvalResult, ...]


def mid_bounds_x0() -> np.ndarray:
    """T0-locked CMA ``x0``: mid rule bits + mid float param bounds."""
    x0 = np.empty(GENOME_SIZE, dtype=np.float64)
    x0[:18] = 0.5
    x0[18] = 0.5 * (NOISE_MIN + NOISE_MAX)
    x0[19] = 0.5 * (RESOURCE_REGEN_MIN + RESOURCE_REGEN_MAX)
    x0[20] = 0.5 * (PREDATION_MIN + PREDATION_MAX)
    return x0


def world_spec_to_solution(spec: WorldSpec) -> np.ndarray:
    """Encode ``WorldSpec`` as genetic 21-D solution vector."""
    return encode_world(spec).astype(np.float64, copy=False)


def solution_to_world_spec(
    theta: np.ndarray | Sequence[float],
    *,
    grid_size: int = 50,
    steps: int = ILLUMINATOR_MIN_STEPS,
    decode_mode: DecodeMode = "rint",
    rng: np.random.Generator | None = None,
) -> WorldSpec:
    """Decode CMA ``theta`` to ``WorldSpec`` (rule bits per ``decode_mode``; floats clipped)."""
    genes = np.asarray(theta, dtype=np.float64).reshape(GENOME_SIZE).copy()
    clip_genome_float_params(genes, start_index=18)
    return decode_genome(
        genes,
        grid_size=grid_size,
        steps=steps,
        decode_mode=decode_mode,
        rng=rng,
    )


def measures_vector(measures: dict[str, float]) -> np.ndarray:
    """Return ``[stability, diversity]`` for pyribs ``ranges`` order."""
    return np.asarray(
        [float(measures["stability"]), float(measures["diversity"])],
        dtype=np.float64,
    )


def flat_cell_index(
    stability: float,
    diversity: float,
    *,
    resolution: int = 50,
) -> int:
    """LifeManifold flat niche index ``i * resolution + j`` (matches pyribs layout)."""
    i, j = bin_index(stability, diversity, resolution)
    return int(i * resolution + j)


def evaluate_solution(
    theta: np.ndarray | Sequence[float],
    *,
    knobs: PyribsEvalKnobs | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, np.ndarray, EvalResult]:
    """Decode one solution and evaluate with illuminator ``evaluate_candidate``."""
    cfg = knobs or PyribsEvalKnobs()
    spec = solution_to_world_spec(
        theta,
        grid_size=cfg.grid_size,
        steps=cfg.steps,
        decode_mode=cfg.decode_mode,
        rng=rng,
    )
    result = evaluate_candidate(
        spec,
        resolution=cfg.resolution,
        early_extinction_step=cfg.early_extinction_step,
        enforce_min_steps=cfg.enforce_min_steps,
        performance=cfg.performance,
    )
    return float(result.fitness), measures_vector(result.measures), result


def evaluate_solutions_batch(
    thetas: np.ndarray,
    *,
    knobs: PyribsEvalKnobs | None = None,
    eval_pool: ParallelEvalPool | None = None,
    batch_index: int = 0,
) -> PyribsEvalBatch:
    """Evaluate a batch of solutions; optional persistent parallel pool."""
    cfg = knobs or PyribsEvalKnobs()
    arr = np.asarray(thetas, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != GENOME_SIZE:
        msg = f"thetas must have shape (n, {GENOME_SIZE}), got {arr.shape}"
        raise ValueError(msg)
    specs: list[WorldSpec] = []
    for row_index, row in enumerate(arr):
        rng = None
        if cfg.decode_mode == "bernoulli":
            rng = np.random.default_rng(
                cfg.eval_seed + batch_index * 10_000 + row_index
            )
        specs.append(
            solution_to_world_spec(
                row,
                grid_size=cfg.grid_size,
                steps=cfg.steps,
                decode_mode=cfg.decode_mode,
                rng=rng,
            )
        )
    own_pool = False
    pool = eval_pool
    if pool is None:
        pool = parallel_eval_context(cfg.performance, batch_size=max(len(specs), 1))
        own_pool = pool is not None
    try:
        if pool is None:
            sims = [
                simulate_candidate(
                    spec,
                    early_extinction_step=cfg.early_extinction_step,
                    enforce_min_steps=cfg.enforce_min_steps,
                    performance=cfg.performance,
                )
                for spec in specs
            ]
        else:
            workers = effective_parallel_workers(
                cfg.performance, batch_size=max(len(specs), 1)
            )
            sims = evaluate_batch_parallel(
                specs,
                early_extinction_step=cfg.early_extinction_step,
                enforce_min_steps=cfg.enforce_min_steps,
                performance=cfg.performance,
                workers=workers,
                eval_pool=pool,
            )
    finally:
        if own_pool and pool is not None:
            pool.close()
            pool.join()

    results = tuple(
        eval_result_from_simulation(sim, resolution=cfg.resolution, archive=None)
        for sim in sims
    )
    objectives = np.asarray([r.fitness for r in results], dtype=np.float64)
    measures = np.vstack([measures_vector(r.measures) for r in results])
    return PyribsEvalBatch(objectives=objectives, measures=measures, results=results)


def coverage_pct(archive: Any, *, n_cells: int = _N_CELLS) -> float:
    """Nightly-style coverage percent: ``100 * num_elites / n_cells``."""
    if n_cells <= 0:
        return 0.0
    filled = int(archive.stats.num_elites)
    return 100.0 * float(filled) / float(n_cells)


def mean_best_fitness(archive: Any) -> float | None:
    """Mean objective over filled elites; ``None`` if archive empty."""
    if int(archive.stats.num_elites) == 0:
        return None
    mean = archive.stats.obj_mean
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        data = archive.data()
        objectives = np.asarray(data["objective"], dtype=np.float64)
        if objectives.size == 0:
            return None
        return float(np.mean(objectives))
    return float(mean)


def qd_score(archive: Any) -> float:
    """Canonical QD-score: sum of objectives over filled pyribs archive elites."""
    if int(archive.stats.num_elites) == 0:
        return 0.0
    data = archive.data()
    objectives = np.asarray(data["objective"], dtype=np.float64)
    if objectives.size == 0:
        return 0.0
    return float(np.sum(objectives))

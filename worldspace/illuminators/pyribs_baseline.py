"""CMA-ME / CMA-MAE (pyribs) baseline runner for Q1 v3 B2 / RQ4.

Uses T0-locked hyperparameters and T1 ``pyribs_adapter`` for genome + illuminator
evaluation. Surrogate/LLM stay off.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from worldspace.illuminators.archive import ARCHIVE_SCHEMA_VERSION
from worldspace.illuminators.archive_trace import (
    ARCHIVE_TRACE_FILENAME,
    write_archive_trace_line,
)
from worldspace.illuminators.evaluation import ILLUMINATOR_MIN_STEPS, bin_index
from worldspace.illuminators.emitters.genetics import DecodeMode
from worldspace.illuminators.pyribs_adapter import (
    ARCHIVE_DIMS,
    ARCHIVE_RANGES,
    GENOME_SIZE,
    PyribsEvalKnobs,
    coverage_pct,
    evaluate_solutions_batch,
    mean_best_fitness,
    measures_vector,
    mid_bounds_x0,
    qd_score,
    solution_to_world_spec,
    world_spec_to_solution,
)
from worldspace.simulator_perf import SimulatorPerformanceOptions
from worldspace.specs.spec import WorldSpec
from worldspace.specs.world_param_bounds import (
    FLOAT_PARAM_BOUNDS,
    RULE_BIT_MAX,
    RULE_BIT_MIN,
)

logger = logging.getLogger(__name__)

AlgoName = Literal["cma_me", "cma_mae"]

DEFAULT_NUM_EMITTERS = 5
DEFAULT_EMITTER_BATCH_SIZE = 50
DEFAULT_SIGMA0 = 0.2
DEFAULT_EVALUATIONS = 32_500
PYRIBS_VERSION = "0.11.0"
DEFAULT_BASELINE_ARCHIVE = Path(
    "artifacts/map_elites_nightly/baseline/map_elites_archive.jsonl"
)

__all__ = [
    "DEFAULT_BASELINE_ARCHIVE",
    "DEFAULT_EMITTER_BATCH_SIZE",
    "DEFAULT_EVALUATIONS",
    "DEFAULT_NUM_EMITTERS",
    "DEFAULT_SIGMA0",
    "PYRIBS_VERSION",
    "PyribsBaselineConfig",
    "PyribsBaselineResult",
    "build_scheduler",
    "export_archive_jsonl",
    "genome_bounds",
    "load_baseline_into_archives",
    "run_pyribs_baseline",
    "pyribs_hyperparams",
    "write_run_summary",
]


@dataclass(frozen=True)
class PyribsBaselineConfig:
    """Runnable B2 configuration (defaults = T0 lock)."""

    algo: AlgoName
    seed: int
    evaluations: int = DEFAULT_EVALUATIONS
    num_emitters: int = DEFAULT_NUM_EMITTERS
    emitter_batch_size: int = DEFAULT_EMITTER_BATCH_SIZE
    sigma0: float = DEFAULT_SIGMA0
    grid_size: int = 50
    steps: int = ILLUMINATOR_MIN_STEPS
    early_extinction_step: int = 200
    load_archive: Path | None = DEFAULT_BASELINE_ARCHIVE
    parallel_eval: bool = True
    parallel_workers: int = 0
    decode_mode: DecodeMode = "rint"
    condition_label: str | None = None


@dataclass(frozen=True)
class PyribsBaselineResult:
    """Artifacts from one pyribs baseline run."""

    algo: AlgoName
    seed: int
    evaluations: int
    asks: int
    ask_size: int
    filled_cells: int
    coverage: float
    mean_best_fitness: float | None
    elapsed_seconds: float
    warm_start_elites: int
    report_archive: GridArchive


def genome_bounds() -> list[tuple[float, float]]:
    """Box bounds for CMA on the genetic 21-D genome."""
    bounds: list[tuple[float, float]] = [
        (RULE_BIT_MIN, RULE_BIT_MAX) for _ in range(18)
    ]
    bounds.extend(FLOAT_PARAM_BOUNDS)
    return bounds


def pyribs_hyperparams(config: PyribsBaselineConfig) -> dict[str, Any]:
    """T0-locked pyribs/CMA knobs for ``nightly_run_summary.json`` and tables."""
    ask_size = config.num_emitters * config.emitter_batch_size
    n_asks = config.evaluations // ask_size if ask_size else 0
    x0 = mid_bounds_x0()
    common: dict[str, Any] = {
        "pyribs_version": PYRIBS_VERSION,
        "algo": config.algo,
        "solution_dim": GENOME_SIZE,
        "archive_dims": list(ARCHIVE_DIMS),
        "archive_ranges": [list(r) for r in ARCHIVE_RANGES],
        "num_emitters": config.num_emitters,
        "emitter_batch_size": config.emitter_batch_size,
        "ask_size": ask_size,
        "asks": n_asks,
        "total_evaluations": config.evaluations,
        "sigma0": config.sigma0,
        "x0": [float(v) for v in x0],
        "x0_description": (
            "18×0.5 rule bits + mid(noise, resource_regen, predation) bounds"
        ),
        "genome_bounds_note": (
            "No ES hard box bounds; clip/decode at eval (continuous relaxation)"
        ),
        "decode_mode": config.decode_mode,
        "warm_start_enabled": config.load_archive is not None,
        "warm_start_archive": (
            str(config.load_archive.resolve())
            if config.load_archive is not None
            else None
        ),
        "grid_size": config.grid_size,
        "steps": config.steps,
        "early_extinction_step": config.early_extinction_step,
    }
    if config.algo == "cma_me":
        common.update(
            {
                "learning_rate": 1.0,
                "threshold_min": None,
                "result_archive": False,
                "ranker": "2imp",
                "selection_rule": "filter",
                "restart_rule": "no_improvement",
            }
        )
    elif config.algo == "cma_mae":
        common.update(
            {
                "learning_rate": 0.01,
                "threshold_min": 0.0,
                "result_archive": True,
                "report_archive": "result",
                "ranker": "imp",
                "selection_rule": "mu",
                "restart_rule": "basic",
            }
        )
    else:
        msg = f"unknown algo {config.algo!r}"
        raise ValueError(msg)
    return common


def build_scheduler(
    config: PyribsBaselineConfig,
) -> tuple[Scheduler, GridArchive, GridArchive | None]:
    """Construct pyribs Scheduler for CMA-ME or CMA-MAE (T0 knobs)."""
    if config.algo == "cma_me":
        archive = _make_grid_archive(
            learning_rate=1.0, threshold_min=None, seed=config.seed
        )
        result_archive = None
        ranker = "2imp"
        selection_rule: Literal["mu", "filter"] = "filter"
        restart_rule: Literal["no_improvement", "basic"] = "no_improvement"
    elif config.algo == "cma_mae":
        archive = _make_grid_archive(
            learning_rate=0.01, threshold_min=0.0, seed=config.seed
        )
        result_archive = _make_grid_archive(
            learning_rate=1.0, threshold_min=None, seed=config.seed + 1
        )
        ranker = "imp"
        selection_rule = "mu"
        restart_rule = "basic"
    else:
        msg = f"unknown algo {config.algo!r}"
        raise ValueError(msg)

    x0 = mid_bounds_x0()
    # Hard box bounds + sigma0=0.2 on [0,1] causes CMA-ES resample storms.
    # Clip / rint happens in ``solution_to_world_spec`` (T0 continuous relaxation).
    emitters = [
        EvolutionStrategyEmitter(
            archive,
            x0=x0,
            sigma0=config.sigma0,
            ranker=ranker,
            selection_rule=selection_rule,
            restart_rule=restart_rule,
            batch_size=config.emitter_batch_size,
            seed=config.seed + 1000 + index,
        )
        for index in range(config.num_emitters)
    ]
    scheduler = Scheduler(archive, emitters, result_archive=result_archive)
    return scheduler, archive, result_archive


def load_baseline_into_archives(
    path: Path,
    *,
    archive: GridArchive,
    result_archive: GridArchive | None,
    grid_size: int,
    steps: int,
) -> int:
    """Warm-start pyribs archives from LifeManifold MAP-Elites JSONL."""
    if not path.is_file():
        msg = f"baseline archive not found: {path}"
        raise FileNotFoundError(msg)
    solutions: list[np.ndarray] = []
    objectives: list[float] = []
    measures: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        spec = WorldSpec.from_json_dict(record["world_spec"])
        spec.grid_size = grid_size
        spec.steps = steps
        solutions.append(world_spec_to_solution(spec))
        objectives.append(float(record["fitness"]))
        measures.append(measures_vector(record["measures"]))
    if not solutions:
        return 0
    sol_arr = np.vstack(solutions)
    obj_arr = np.asarray(objectives, dtype=np.float64)
    meas_arr = np.vstack(measures)
    archive.add(sol_arr, obj_arr, meas_arr)
    if result_archive is not None:
        result_archive.add(sol_arr, obj_arr, meas_arr)
    return int(sol_arr.shape[0])


def export_archive_jsonl(
    archive: GridArchive,
    path: Path,
    *,
    decode_mode: DecodeMode = "rint",
) -> int:
    """Write pyribs elites as minimal LifeManifold archive JSONL for aggregate."""
    data = archive.data()
    solutions = np.asarray(data["solution"], dtype=np.float64)
    objectives = np.asarray(data["objective"], dtype=np.float64)
    measures = np.asarray(data["measures"], dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for index in range(int(objectives.shape[0])):
            stability = float(measures[index, 0])
            diversity = float(measures[index, 1])
            i, j = bin_index(stability, diversity, ARCHIVE_DIMS[0])
            export_decode: DecodeMode = (
                "threshold" if decode_mode == "bernoulli" else decode_mode
            )
            spec = solution_to_world_spec(
                solutions[index],
                grid_size=50,
                steps=ILLUMINATOR_MIN_STEPS,
                decode_mode=export_decode,
            )
            record = {
                "schema_version": ARCHIVE_SCHEMA_VERSION,
                "bin": [i, j],
                "world_spec": spec.to_json_dict(),
                "fitness": float(objectives[index]),
                "measures": {
                    "stability": stability,
                    "diversity": diversity,
                },
                "metadata": {
                    "id": f"pyribs-{index}",
                    "parent_id": None,
                    "generated_by": "pyribs",
                    "emitter_type": "cma",
                    "timestamp": "",
                    "prompt_version": "",
                },
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1
    return count


def write_run_summary(
    path: Path,
    *,
    result: PyribsBaselineResult,
    config: PyribsBaselineConfig,
    archive_jsonl: Path,
    scheduler_label: str,
) -> None:
    """Write ``nightly_run_summary.json`` fields aggregate expects."""
    payload = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "scheduler": scheduler_label,
        "condition": config.condition_label or config.algo,
        "seed": result.seed,
        "iterations": result.asks,
        "evaluations": result.evaluations,
        "filled_cells": result.filled_cells,
        "grid_resolution": ARCHIVE_DIMS[0],
        "archive_type": "grid",
        "n_cells": ARCHIVE_DIMS[0] * ARCHIVE_DIMS[1],
        "coverage": result.coverage,
        "jsonl_raw_lines": result.filled_cells,
        "jsonl_collapsed_cells": result.filled_cells,
        "elapsed_seconds": result.elapsed_seconds,
        "llm_enabled": False,
        "surrogate_enabled": False,
        "archive_jsonl": str(archive_jsonl.resolve()),
        "pyribs_algo": config.algo,
        "pyribs_ask_size": result.ask_size,
        "pyribs_warm_start_elites": result.warm_start_elites,
        "pyribs_hyperparams": pyribs_hyperparams(config),
        "mean_best_fitness": result.mean_best_fitness,
        "qd_score": round(qd_score(result.report_archive), 6),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def run_pyribs_baseline(
    config: PyribsBaselineConfig,
    *,
    output_dir: Path,
) -> PyribsBaselineResult:
    """Run one B2 seed to exact ``evaluations`` count."""
    ask_size = config.num_emitters * config.emitter_batch_size
    if ask_size <= 0:
        raise ValueError("ask_size must be positive")
    if config.evaluations % ask_size != 0:
        msg = (
            f"evaluations ({config.evaluations}) must be divisible by "
            f"ask_size ({ask_size} = {config.num_emitters}×"
            f"{config.emitter_batch_size})"
        )
        raise ValueError(msg)
    n_asks = config.evaluations // ask_size

    output_dir.mkdir(parents=True, exist_ok=True)
    scheduler, archive, result_archive = build_scheduler(config)

    warm_start = 0
    if config.load_archive is not None:
        warm_start = load_baseline_into_archives(
            Path(config.load_archive),
            archive=archive,
            result_archive=result_archive,
            grid_size=config.grid_size,
            steps=config.steps,
        )
        logger.info("Warm-started %s elites from %s", warm_start, config.load_archive)

    knobs = PyribsEvalKnobs(
        grid_size=config.grid_size,
        steps=config.steps,
        resolution=ARCHIVE_DIMS[0],
        early_extinction_step=config.early_extinction_step,
        enforce_min_steps=True,
        performance=SimulatorPerformanceOptions(
            parallel_eval=config.parallel_eval,
            parallel_workers=config.parallel_workers,
        ),
        decode_mode=config.decode_mode,
        eval_seed=config.seed,
    )

    started = time.perf_counter()
    evaluated = 0
    trace_path = output_dir / ARCHIVE_TRACE_FILENAME
    trace_file = trace_path.open("w", encoding="utf-8")
    report0 = result_archive if result_archive is not None else archive
    mean0 = mean_best_fitness(report0)
    qd0 = qd_score(report0)
    write_archive_trace_line(
        trace_file,
        {
            "ask": 0,
            "asks_total": n_asks,
            "evaluations": 0,
            "filled_cells": int(report0.stats.num_elites),
            "coverage": round(coverage_pct(report0) / 100.0, 6),
            "mean_best_fitness": round(mean0, 6) if mean0 is not None else None,
            "qd_score": round(qd0, 6),
        },
    )
    try:
        for ask_index in range(n_asks):
            solutions = scheduler.ask()
            if solutions.shape[0] != ask_size:
                msg = f"expected ask size {ask_size}, got {solutions.shape[0]}"
                raise RuntimeError(msg)
            batch = evaluate_solutions_batch(
                solutions, knobs=knobs, batch_index=ask_index
            )
            scheduler.tell(batch.objectives, batch.measures)
            evaluated += int(solutions.shape[0])
            report = result_archive if result_archive is not None else archive
            mean_fit = mean_best_fitness(report)
            qd = qd_score(report)
            write_archive_trace_line(
                trace_file,
                {
                    "ask": ask_index + 1,
                    "asks_total": n_asks,
                    "evaluations": evaluated,
                    "filled_cells": int(report.stats.num_elites),
                    "coverage": round(coverage_pct(report) / 100.0, 6),
                    "mean_best_fitness": (
                        round(mean_fit, 6) if mean_fit is not None else None
                    ),
                    "qd_score": round(qd, 6),
                },
            )
            if (ask_index + 1) % 10 == 0 or ask_index + 1 == n_asks:
                logger.info(
                    "ask %s/%s evals=%s elites=%s coverage=%.4f",
                    ask_index + 1,
                    n_asks,
                    evaluated,
                    report.stats.num_elites,
                    coverage_pct(report) / 100.0,
                )
    finally:
        trace_file.close()

    if evaluated != config.evaluations:
        msg = f"expected {config.evaluations} evaluations, got {evaluated}"
        raise RuntimeError(msg)

    elapsed = time.perf_counter() - started
    report_archive = result_archive if result_archive is not None else archive
    filled = int(report_archive.stats.num_elites)
    cov_pct = coverage_pct(report_archive)
    mean_fit = mean_best_fitness(report_archive)

    archive_jsonl = output_dir / "map_elites_archive.jsonl"
    export_archive_jsonl(
        report_archive,
        archive_jsonl,
        decode_mode=config.decode_mode,
    )

    result = PyribsBaselineResult(
        algo=config.algo,
        seed=config.seed,
        evaluations=evaluated,
        asks=n_asks,
        ask_size=ask_size,
        filled_cells=filled,
        coverage=cov_pct / 100.0,
        mean_best_fitness=mean_fit,
        elapsed_seconds=round(elapsed, 3),
        warm_start_elites=warm_start,
        report_archive=report_archive,
    )
    write_run_summary(
        output_dir / "nightly_run_summary.json",
        result=result,
        config=config,
        archive_jsonl=archive_jsonl,
        scheduler_label=f"pyribs:{config.algo}",
    )
    archive_arrays = {
        str(key): np.asarray(val) for key, val in report_archive.data().items()
    }
    # Pass allow_pickle explicitly so pyright does not bind NDArray kwargs to it.
    np.savez_compressed(
        str(output_dir / "pyribs_archive.npz"),
        allow_pickle=False,
        **archive_arrays,
    )
    return result


def _make_grid_archive(
    *,
    learning_rate: float,
    threshold_min: float | None,
    seed: int,
) -> GridArchive:
    kwargs: dict[str, Any] = {
        "solution_dim": GENOME_SIZE,
        "dims": ARCHIVE_DIMS,
        "ranges": list(ARCHIVE_RANGES),
        "seed": seed,
        "learning_rate": learning_rate,
    }
    if threshold_min is not None:
        kwargs["threshold_min"] = threshold_min
    return GridArchive(**kwargs)

"""CA-independent pyribs runner for standard Sphere/Rastrigin QD benchmarks."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter, GaussianEmitter
from ribs.schedulers import Scheduler

from worldspace.benchmarks.qd_sphere import (
    DEFAULT_ARCHIVE_DIMS,
    DEFAULT_SOLUTION_DIM,
    SPHERE_SHIFT,
    archive_ranges,
    linear_projection_measures,
    rastrigin_objective,
    sphere_objective,
)
from worldspace.illuminators.archive import ARCHIVE_SCHEMA_VERSION
from worldspace.illuminators.archive_trace import (
    ARCHIVE_TRACE_FILENAME,
    write_archive_trace_line,
)

BenchmarkName = Literal["sphere", "rastrigin"]
AlgoName = Literal["cma_me", "cma_mae", "me_random"]

DEFAULT_EVALUATIONS = 32_500
DEFAULT_CMA_NUM_EMITTERS = 5
DEFAULT_CMA_BATCH_SIZE = 50
DEFAULT_RANDOM_BATCH_SIZE = 250
DEFAULT_SIGMA0 = 0.2
DEFAULT_RANDOM_SIGMA = 0.5
PYRIBS_VERSION = "0.11.0"

__all__ = [
    "AlgoName",
    "BenchmarkName",
    "DEFAULT_EVALUATIONS",
    "PYRIBS_VERSION",
    "PyribsStandardConfig",
    "PyribsStandardResult",
    "build_scheduler",
    "run_pyribs_standard",
    "standard_hyperparams",
]


@dataclass(frozen=True)
class PyribsStandardConfig:
    """Configuration for one standard-benchmark seed."""

    benchmark: BenchmarkName
    algo: AlgoName
    seed: int
    evaluations: int = DEFAULT_EVALUATIONS
    solution_dim: int = DEFAULT_SOLUTION_DIM
    archive_dims: tuple[int, int] = DEFAULT_ARCHIVE_DIMS
    num_emitters: int | None = None
    emitter_batch_size: int | None = None
    sigma0: float = DEFAULT_SIGMA0
    random_sigma: float = DEFAULT_RANDOM_SIGMA

    @property
    def effective_num_emitters(self) -> int:
        return (
            self.num_emitters
            if self.num_emitters is not None
            else (1 if self.algo == "me_random" else DEFAULT_CMA_NUM_EMITTERS)
        )

    @property
    def effective_batch_size(self) -> int:
        return (
            self.emitter_batch_size
            if self.emitter_batch_size is not None
            else (
                DEFAULT_RANDOM_BATCH_SIZE
                if self.algo == "me_random"
                else DEFAULT_CMA_BATCH_SIZE
            )
        )

    @property
    def ask_size(self) -> int:
        return self.effective_num_emitters * self.effective_batch_size

    def validate(self) -> None:
        if self.benchmark == "rastrigin" and self.algo == "me_random":
            raise ValueError("me_random is only defined for the sphere benchmark")
        if self.solution_dim <= 0 or self.solution_dim % 2:
            raise ValueError("solution_dim must be a positive even integer")
        if len(self.archive_dims) != 2 or any(dim <= 0 for dim in self.archive_dims):
            raise ValueError("archive_dims must contain two positive integers")
        if self.effective_num_emitters <= 0 or self.effective_batch_size <= 0:
            raise ValueError("emitter count and batch size must be positive")
        if self.evaluations <= 0 or self.evaluations % self.ask_size:
            raise ValueError(
                f"evaluations ({self.evaluations}) must be divisible by "
                f"ask_size ({self.ask_size})"
            )


@dataclass(frozen=True)
class PyribsStandardResult:
    """Final metrics and archive from one standard benchmark run."""

    benchmark: BenchmarkName
    algo: AlgoName
    seed: int
    evaluations: int
    asks: int
    ask_size: int
    filled_cells: int
    coverage: float
    mean_best_fitness: float | None
    qd_score: float
    elapsed_seconds: float
    report_archive: GridArchive


def standard_hyperparams(config: PyribsStandardConfig) -> dict[str, Any]:
    """Return all locked benchmark and algorithm knobs for provenance."""
    config.validate()
    params: dict[str, Any] = {
        "pyribs_version": PYRIBS_VERSION,
        "benchmark": config.benchmark,
        "algo": config.algo,
        "solution_dim": config.solution_dim,
        "clip_bounds": [-5.12, 5.12],
        "archive_dims": list(config.archive_dims),
        "archive_ranges": [list(item) for item in archive_ranges(config.solution_dim)],
        "num_emitters": config.effective_num_emitters,
        "emitter_batch_size": config.effective_batch_size,
        "ask_size": config.ask_size,
        "asks": config.evaluations // config.ask_size,
        "total_evaluations": config.evaluations,
        "x0": [SPHERE_SHIFT] * config.solution_dim,
        "warm_start_archive": None,
    }
    if config.algo == "cma_me":
        params.update(
            {
                "sigma0": config.sigma0,
                "learning_rate": 1.0,
                "threshold_min": None,
                "result_archive": False,
                "ranker": "2imp",
                "selection_rule": "filter",
                "restart_rule": "no_improvement",
            }
        )
    elif config.algo == "cma_mae":
        params.update(
            {
                "sigma0": config.sigma0,
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
        params.update(
            {
                "sigma": config.random_sigma,
                "learning_rate": 1.0,
                "threshold_min": None,
                "result_archive": False,
                "emitter": "GaussianEmitter",
            }
        )
    return params


def build_scheduler(
    config: PyribsStandardConfig,
) -> tuple[Scheduler, GridArchive, GridArchive | None]:
    """Construct the locked scheduler and its optimization/report archives."""
    config.validate()
    if config.algo == "cma_mae":
        archive = _make_archive(
            config,
            seed=config.seed,
            learning_rate=0.01,
            threshold_min=0.0,
        )
        result_archive = _make_archive(
            config,
            seed=config.seed + 1,
            learning_rate=1.0,
            threshold_min=None,
        )
    else:
        archive = _make_archive(
            config,
            seed=config.seed,
            learning_rate=1.0,
            threshold_min=None,
        )
        result_archive = None

    x0 = np.full(config.solution_dim, SPHERE_SHIFT, dtype=np.float64)
    if config.algo == "me_random":
        emitters = [
            GaussianEmitter(
                archive,
                x0=x0,
                sigma=config.random_sigma,
                batch_size=config.effective_batch_size,
                seed=config.seed + 1000,
            )
        ]
    else:
        if config.algo == "cma_me":
            ranker = "2imp"
            selection_rule: Literal["mu", "filter"] = "filter"
            restart_rule: Literal["no_improvement", "basic"] = "no_improvement"
        else:
            ranker = "imp"
            selection_rule = "mu"
            restart_rule = "basic"
        emitters = [
            EvolutionStrategyEmitter(
                archive,
                x0=x0,
                sigma0=config.sigma0,
                ranker=ranker,
                selection_rule=selection_rule,
                restart_rule=restart_rule,
                batch_size=config.effective_batch_size,
                seed=config.seed + 1000 + index,
            )
            for index in range(config.effective_num_emitters)
        ]
    return (
        Scheduler(archive, emitters, result_archive=result_archive),
        archive,
        result_archive,
    )


def run_pyribs_standard(
    config: PyribsStandardConfig,
    *,
    output_dir: Path,
) -> PyribsStandardResult:
    """Run one standard benchmark seed with an exact evaluation budget."""
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    scheduler, archive, result_archive = build_scheduler(config)
    n_asks = config.evaluations // config.ask_size
    trace_path = output_dir / ARCHIVE_TRACE_FILENAME
    started = time.perf_counter()
    evaluated = 0

    with trace_path.open("w", encoding="utf-8") as trace_file:
        _write_trace(
            trace_file,
            report_archive=result_archive or archive,
            ask=0,
            asks_total=n_asks,
            evaluations=0,
        )
        for ask_index in range(n_asks):
            solutions = np.asarray(scheduler.ask(), dtype=np.float64)
            if solutions.shape != (config.ask_size, config.solution_dim):
                raise RuntimeError(
                    f"expected solutions shape "
                    f"{(config.ask_size, config.solution_dim)}, got {solutions.shape}"
                )
            if config.benchmark == "sphere":
                objectives = np.asarray(sphere_objective(solutions))
            else:
                objectives = np.asarray(rastrigin_objective(solutions))
            measures = linear_projection_measures(solutions)
            scheduler.tell(objectives, measures)
            evaluated += solutions.shape[0]
            _write_trace(
                trace_file,
                report_archive=result_archive or archive,
                ask=ask_index + 1,
                asks_total=n_asks,
                evaluations=evaluated,
            )

    if evaluated != config.evaluations:
        raise RuntimeError(
            f"expected {config.evaluations} evaluations, got {evaluated}"
        )
    elapsed = time.perf_counter() - started
    report = result_archive or archive
    filled, coverage, mean_fitness, score = _archive_metrics(report)
    result = PyribsStandardResult(
        benchmark=config.benchmark,
        algo=config.algo,
        seed=config.seed,
        evaluations=evaluated,
        asks=n_asks,
        ask_size=config.ask_size,
        filled_cells=filled,
        coverage=coverage,
        mean_best_fitness=mean_fitness,
        qd_score=score,
        elapsed_seconds=round(elapsed, 3),
        report_archive=report,
    )
    _write_summary(
        output_dir / "nightly_run_summary.json",
        config=config,
        result=result,
    )
    archive_arrays = {
        str(key): np.asarray(value) for key, value in report.data().items()
    }
    np.savez_compressed(
        str(output_dir / "pyribs_archive.npz"),
        allow_pickle=False,
        **archive_arrays,
    )
    return result


def _make_archive(
    config: PyribsStandardConfig,
    *,
    seed: int,
    learning_rate: float,
    threshold_min: float | None,
) -> GridArchive:
    kwargs: dict[str, Any] = {
        "solution_dim": config.solution_dim,
        "dims": config.archive_dims,
        "ranges": archive_ranges(config.solution_dim),
        "seed": seed,
        "learning_rate": learning_rate,
    }
    if threshold_min is not None:
        kwargs["threshold_min"] = threshold_min
    return GridArchive(**kwargs)


def _archive_metrics(
    archive: GridArchive,
) -> tuple[int, float, float | None, float]:
    objectives = np.asarray(archive.data("objective"), dtype=np.float64)
    filled = int(objectives.size)
    n_cells = int(np.prod(archive.dims))
    coverage = float(filled) / float(n_cells)
    score = float(np.sum(objectives))
    mean_fitness = float(np.mean(objectives)) if filled else None
    return filled, coverage, mean_fitness, score


def _write_trace(
    trace_file: Any,
    *,
    report_archive: GridArchive,
    ask: int,
    asks_total: int,
    evaluations: int,
) -> None:
    filled, coverage, mean_fitness, score = _archive_metrics(report_archive)
    write_archive_trace_line(
        trace_file,
        {
            "ask": ask,
            "asks_total": asks_total,
            "evaluations": evaluations,
            "filled_cells": filled,
            "coverage": round(coverage, 6),
            "mean_best_fitness": (
                round(mean_fitness, 6) if mean_fitness is not None else None
            ),
            "qd_score": round(score, 6),
        },
    )


def _write_summary(
    path: Path,
    *,
    config: PyribsStandardConfig,
    result: PyribsStandardResult,
) -> None:
    payload = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "scheduler": f"pyribs-standard:{config.benchmark}:{config.algo}",
        "condition": config.algo,
        "benchmark": config.benchmark,
        "standard_benchmark": True,
        "seed": config.seed,
        "iterations": result.asks,
        "evaluations": result.evaluations,
        "filled_cells": result.filled_cells,
        "grid_resolution": list(config.archive_dims),
        "archive_type": "grid",
        "n_cells": int(np.prod(config.archive_dims)),
        "coverage": result.coverage,
        "mean_best_fitness": result.mean_best_fitness,
        "qd_score": result.qd_score,
        "elapsed_seconds": result.elapsed_seconds,
        "llm_enabled": False,
        "surrogate_enabled": False,
        "pyribs_algo": config.algo,
        "pyribs_version": PYRIBS_VERSION,
        "pyribs_ask_size": result.ask_size,
        "pyribs_warm_start_elites": 0,
        "benchmark_hyperparams": standard_hyperparams(config),
        "archive_npz": str((path.parent / "pyribs_archive.npz").resolve()),
        "archive_trace": str((path.parent / ARCHIVE_TRACE_FILENAME).resolve()),
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

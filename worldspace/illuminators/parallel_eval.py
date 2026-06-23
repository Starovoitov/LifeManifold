"""Parallel batch simulation for MAP-Elites (no archive IO in workers)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import pool
from typing import TYPE_CHECKING

from worldspace.illuminators.evaluation import SimulationOutcome, simulate_candidate
from worldspace.simulator_perf import (
    SimulatorPerformanceOptions,
    effective_parallel_workers,
)
from worldspace.specs.spec import WorldSpec

if TYPE_CHECKING:
    from multiprocessing.context import BaseContext

__all__ = [
    "ParallelEvalPool",
    "evaluate_batch_parallel",
    "parallel_eval_context",
]


@dataclass(frozen=True)
class _ParallelEvalJob:
    world_spec: WorldSpec
    early_extinction_step: int
    enforce_min_steps: bool
    performance: SimulatorPerformanceOptions


class ParallelEvalPool:
    """Reusable process pool for illuminator parallel eval."""

    def __init__(self, processes: int, *, mp_context: BaseContext | None = None) -> None:
        from multiprocessing import get_context

        ctx = mp_context if mp_context is not None else get_context("forkserver")
        self._pool: pool.Pool = ctx.Pool(processes=processes)

    def map_simulations(
        self,
        jobs: Sequence[_ParallelEvalJob],
    ) -> list[SimulationOutcome]:
        if not jobs:
            return []
        return self._pool.map(_parallel_eval_worker, list(jobs))

    def close(self) -> None:
        self._pool.close()

    def join(self) -> None:
        self._pool.join()

    def terminate(self) -> None:
        self._pool.terminate()


def parallel_eval_context(
    performance: SimulatorPerformanceOptions,
    *,
    batch_size: int,
) -> ParallelEvalPool | None:
    """Create a persistent pool when parallel eval is enabled."""
    if not performance.parallel_eval:
        return None
    workers = effective_parallel_workers(performance, batch_size=batch_size)
    if workers <= 1:
        return None
    return ParallelEvalPool(workers)


def evaluate_batch_parallel(
    specs: Sequence[WorldSpec],
    *,
    early_extinction_step: int,
    enforce_min_steps: bool,
    performance: SimulatorPerformanceOptions,
    workers: int,
    eval_pool: ParallelEvalPool | None = None,
) -> list[SimulationOutcome]:
    """Run simulations in parallel; binning stays in the main process."""
    if not specs:
        return []
    jobs = [
        _ParallelEvalJob(
            world_spec=spec,
            early_extinction_step=early_extinction_step,
            enforce_min_steps=enforce_min_steps,
            performance=performance,
        )
        for spec in specs
    ]
    if workers <= 1 or len(jobs) <= 1:
        return [_parallel_eval_worker(job) for job in jobs]
    if eval_pool is not None:
        return eval_pool.map_simulations(jobs)
    pool_holder = ParallelEvalPool(workers)
    try:
        return pool_holder.map_simulations(jobs)
    finally:
        pool_holder.terminate()
        pool_holder.join()


def _parallel_eval_worker(job: _ParallelEvalJob) -> SimulationOutcome:
    return simulate_candidate(
        job.world_spec,
        early_extinction_step=job.early_extinction_step,
        enforce_min_steps=job.enforce_min_steps,
        performance=job.performance,
    )

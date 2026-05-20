"""MAP-Elites iteration loop: batch evaluate, insert, optional JSONL."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from worldspace.illuminators.archive import (
    GridArchive,
    InsertResult,
    insert_and_persist,
    insert_evaluated,
)
from worldspace.illuminators.emitters.base import CandidateEmitter
from worldspace.illuminators.evaluation import (
    ILLUMINATOR_MIN_STEPS,
    EvalResult,
    evaluate_candidate,
)
from worldspace.illuminators.scheduler import (
    EmitterKind,
    RunCounters,
    SchedulerConfig,
    TargetBin,
    resolve_emitter_for_slot,
    select_target_bin,
)
from worldspace.specs.spec import WorldSpec

if TYPE_CHECKING:
    from worldspace.surrogate.buffer import SurrogateBuffer

__all__ = [
    "IterationStats",
    "SlotOutcome",
    "run_iteration",
    "run_scheduler",
]


@dataclass(frozen=True)
class IterationStats:
    """Aggregate outcomes for one scheduler iteration."""

    evaluations: int
    accepted: int
    improved: int
    rejected: int


@dataclass(frozen=True)
class SlotOutcome:
    """Per-slot trace for one iteration (``candidate_id`` ascending)."""

    candidate_id: int
    emitter_kind: EmitterKind
    target_bin: TargetBin
    eval_result: EvalResult
    insert: InsertResult


def run_iteration(
    config: SchedulerConfig,
    archive: GridArchive,
    rng: np.random.Generator,
    counters: RunCounters,
    emitter: CandidateEmitter,
    *,
    grid_size: int,
    steps: int,
    jsonl_path: str | Path | None = None,
    surrogate_buffer: SurrogateBuffer | None = None,
) -> tuple[IterationStats, list[SlotOutcome]]:
    """Run one batch: slots ``0 .. batch_size-1`` in order, evaluate, insert, count.

    Candidates are processed in ascending ``candidate_id``. Together with strict
    ``fitness_new > fitness_old`` in ``GridArchive.try_insert``, equal fitness in the
    same bin within one iteration leaves the first accepted elite in place.
    """
    outcomes: list[SlotOutcome] = []
    accepted = 0
    improved = 0
    rejected = 0

    for candidate_id in range(config.batch_size):
        emitter_kind = resolve_emitter_for_slot(
            config,
            candidate_id=candidate_id,
            candidates_evaluated=counters.candidates_evaluated,
        )
        target = select_target_bin(archive, rng)
        output = emitter.emit(
            emitter_kind=emitter_kind,
            target=target,
            archive=archive,
            rng=rng,
            grid_size=grid_size,
            steps=steps,
        )
        spec = _prepare_world_spec(output.world_spec, grid_size=grid_size, steps=steps)
        eval_result = evaluate_candidate(
            spec,
            resolution=config.grid_resolution,
            early_extinction_step=config.early_extinction_step,
            enforce_min_steps=True,
        )
        if surrogate_buffer is not None:
            from worldspace.surrogate.buffer import append_eval_to_buffer

            append_eval_to_buffer(
                surrogate_buffer,
                eval_result,
                emitter_type=output.metadata.emitter_type,
            )
        if jsonl_path is not None:
            insert = insert_and_persist(
                archive, eval_result, output.metadata, jsonl_path
            )
        else:
            insert = insert_evaluated(archive, eval_result, output.metadata)
        counters.record_evaluation()

        if insert.accepted:
            accepted += 1
        if insert.improved:
            improved += 1
        if insert.rejected:
            rejected += 1

        outcomes.append(
            SlotOutcome(
                candidate_id=candidate_id,
                emitter_kind=emitter_kind,
                target_bin=target,
                eval_result=eval_result,
                insert=insert,
            )
        )

    stats = IterationStats(
        evaluations=config.batch_size,
        accepted=accepted,
        improved=improved,
        rejected=rejected,
    )
    return stats, outcomes


def run_scheduler(
    config: SchedulerConfig,
    archive: GridArchive,
    rng: np.random.Generator,
    emitter: CandidateEmitter,
    *,
    grid_size: int,
    steps: int,
    jsonl_path: str | Path | None = None,
    counters: RunCounters | None = None,
    surrogate_buffer: SurrogateBuffer | None = None,
) -> RunCounters:
    """Run ``config.iterations`` batches and return updated global counters."""
    if counters is None:
        counters = RunCounters()
    for _ in range(config.iterations):
        run_iteration(
            config,
            archive,
            rng,
            counters,
            emitter,
            grid_size=grid_size,
            steps=steps,
            jsonl_path=jsonl_path,
            surrogate_buffer=surrogate_buffer,
        )
        if surrogate_buffer is not None:
            surrogate_buffer.flush()
    if surrogate_buffer is not None:
        surrogate_buffer.flush()
    return counters


def _prepare_world_spec(spec: WorldSpec, *, grid_size: int, steps: int) -> WorldSpec:
    prepared = replace(spec)
    prepared.grid_size = grid_size
    prepared.steps = max(steps, ILLUMINATOR_MIN_STEPS)
    return prepared

"""MAP-Elites iteration loop: batch evaluate, insert, optional JSONL."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from worldspace.illuminators.archive import (
    ARCHIVE_SCHEMA_VERSION,
    ARCHIVE_SCHEMA_VERSION_V1_3,
    GridArchive,
    InsertResult,
    insert_and_persist,
    insert_evaluated,
)
from worldspace.illuminators.archive_protocol import ArchiveProtocol
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
    select_target_cell,
)
from worldspace.specs.spec import WorldSpec
from worldspace.surrogate.acquisition import (
    AcquisitionDecision,
    decide,
    effective_action,
    policy_recommends_skip,
)
from worldspace.surrogate.canonical_hash import world_spec_canonical_hash
from worldspace.surrogate.types import SurrogatePrediction

if TYPE_CHECKING:
    from worldspace.surrogate.buffer import SurrogateBuffer
    from worldspace.surrogate.retrain import RetrainState
    from worldspace.surrogate.surrogate_archive import SurrogateArchiveWriterProtocol
    from worldspace.surrogate.types import SurrogateProtocol

logger = logging.getLogger(__name__)

__all__ = [
    "IterationStats",
    "SlotOutcome",
    "run_iteration",
    "run_scheduler",
]


@dataclass(frozen=True)
class IterationStats:
    """Aggregate outcomes for one scheduler iteration."""

    slots: int
    evaluated: int
    skipped: int
    shadow_would_skip: int
    accepted: int
    improved: int
    rejected: int

    @property
    def evaluations(self) -> int:
        """Completed real simulations in this iteration (legacy alias)."""
        return self.evaluated


@dataclass(frozen=True)
class SlotOutcome:
    """Per-slot trace for one iteration (``candidate_id`` ascending)."""

    candidate_id: int
    emitter_kind: EmitterKind
    target_bin: TargetBin
    skipped: bool
    eval_result: EvalResult | None
    insert: InsertResult | None
    prediction: SurrogatePrediction | None = None
    decision: AcquisitionDecision | None = None


def run_iteration(
    config: SchedulerConfig,
    archive: ArchiveProtocol,
    rng: np.random.Generator,
    counters: RunCounters,
    emitter: CandidateEmitter,
    *,
    iteration_index: int,
    grid_size: int,
    steps: int,
    jsonl_path: str | Path | None = None,
    surrogate_buffer: SurrogateBuffer | None = None,
    surrogate: SurrogateProtocol | None = None,
    surrogate_archive: SurrogateArchiveWriterProtocol | None = None,
) -> tuple[IterationStats, list[SlotOutcome]]:
    """Run one batch: slots ``0 .. batch_size-1`` in order, evaluate or skip, insert.

    Candidates are processed in ascending ``candidate_id``. Together with strict
    ``fitness_new > fitness_old`` in ``GridArchive.try_insert``, equal fitness in the
    same bin within one iteration leaves the first accepted elite in place.
    """
    outcomes: list[SlotOutcome] = []
    evaluated = 0
    skipped = 0
    shadow_would_skip = 0
    accepted = 0
    improved = 0
    rejected = 0
    acquisition_active = _acquisition_logging_active(config)

    for candidate_id in range(config.batch_size):
        emitter_kind = resolve_emitter_for_slot(
            config,
            candidate_id=candidate_id,
            candidates_evaluated=counters.candidates_evaluated,
        )
        target_cell = select_target_cell(archive, rng)
        target_bin = TargetBin.from_target_cell(target_cell)
        output = emitter.emit(
            emitter_kind=emitter_kind,
            target=target_cell,
            archive=archive,
            rng=rng,
            grid_size=grid_size,
            steps=steps,
        )
        spec = _prepare_world_spec(output.world_spec, grid_size=grid_size, steps=steps)

        prediction: SurrogatePrediction | None = None
        decision: AcquisitionDecision | None = None
        runtime_action = "eval"

        if config.surrogate_enabled and surrogate is not None:
            prediction = surrogate.predict(spec)
            if acquisition_active and isinstance(archive, GridArchive):
                decision = decide(config.acquisition, prediction, target_bin, archive)
                runtime_action = effective_action(
                    config.acquisition.mode,
                    decision,
                )
                if policy_recommends_skip(decision):
                    shadow_would_skip += 1

        if runtime_action == "skip":
            assert prediction is not None and decision is not None
            skipped += 1
            if surrogate_archive is not None:
                surrogate_archive.append_slot(
                    iteration=iteration_index,
                    candidate_id=candidate_id,
                    emitter_type=output.metadata.emitter_type,
                    target=target_bin,
                    world_spec_hash=world_spec_canonical_hash(spec),
                    prediction=prediction,
                    decision=decision,
                    acquisition_mode=config.acquisition.mode,
                )
            outcomes.append(
                SlotOutcome(
                    candidate_id=candidate_id,
                    emitter_kind=emitter_kind,
                    target_bin=target_bin,
                    skipped=True,
                    eval_result=None,
                    insert=None,
                    prediction=prediction,
                    decision=decision,
                )
            )
            continue

        eval_result = evaluate_candidate(
            spec,
            resolution=config.grid_resolution,
            archive=archive,
            early_extinction_step=config.early_extinction_step,
            enforce_min_steps=True,
        )
        if config.surrogate_enabled and surrogate_buffer is not None:
            from worldspace.surrogate.buffer import append_eval_to_buffer

            append_eval_to_buffer(
                surrogate_buffer,
                eval_result,
                emitter_type=output.metadata.emitter_type,
            )
        jsonl_schema_version = _jsonl_schema_version_for_archive(archive)
        if jsonl_path is not None:
            insert = insert_and_persist(
                archive,
                eval_result,
                output.metadata,
                jsonl_path,
                schema_version=jsonl_schema_version,
            )
        else:
            insert = insert_evaluated(archive, eval_result, output.metadata)
        counters.record_evaluation()
        evaluated += 1

        if insert.accepted:
            accepted += 1
        if insert.improved:
            improved += 1
        if insert.rejected:
            rejected += 1

        if (
            acquisition_active
            and surrogate_archive is not None
            and prediction is not None
            and decision is not None
        ):
            surrogate_archive.append_slot(
                iteration=iteration_index,
                candidate_id=candidate_id,
                emitter_type=output.metadata.emitter_type,
                target=target_bin,
                world_spec_hash=world_spec_canonical_hash(spec),
                prediction=prediction,
                decision=decision,
                acquisition_mode=config.acquisition.mode,
                eval_result=eval_result,
                insert=insert,
            )

        outcomes.append(
            SlotOutcome(
                candidate_id=candidate_id,
                emitter_kind=emitter_kind,
                target_bin=target_bin,
                skipped=False,
                eval_result=eval_result,
                insert=insert,
                prediction=prediction,
                decision=decision,
            )
        )

    stats = IterationStats(
        slots=config.batch_size,
        evaluated=evaluated,
        skipped=skipped,
        shadow_would_skip=shadow_would_skip,
        accepted=accepted,
        improved=improved,
        rejected=rejected,
    )
    logger.debug(
        "iteration=%s evaluated=%s skipped=%s shadow_would_skip=%s",
        iteration_index,
        evaluated,
        skipped,
        shadow_would_skip,
    )
    return stats, outcomes


def run_scheduler(
    config: SchedulerConfig,
    archive: ArchiveProtocol,
    rng: np.random.Generator,
    emitter: CandidateEmitter,
    *,
    grid_size: int,
    steps: int,
    jsonl_path: str | Path | None = None,
    counters: RunCounters | None = None,
    surrogate_buffer: SurrogateBuffer | None = None,
    surrogate: SurrogateProtocol | None = None,
    retrain_state: RetrainState | None = None,
    surrogate_archive: SurrogateArchiveWriterProtocol | None = None,
) -> RunCounters:
    """Run ``config.iterations`` batches and return updated global counters."""
    if counters is None:
        counters = RunCounters()
    for iteration_index in range(1, config.iterations + 1):
        run_iteration(
            config,
            archive,
            rng,
            counters,
            emitter,
            iteration_index=iteration_index,
            grid_size=grid_size,
            steps=steps,
            jsonl_path=jsonl_path,
            surrogate_buffer=surrogate_buffer,
            surrogate=surrogate,
            surrogate_archive=surrogate_archive,
        )
        if surrogate_buffer is not None:
            surrogate_buffer.flush()
        if surrogate_archive is not None:
            surrogate_archive.flush()
        if (
            config.retrain.enabled
            and surrogate is not None
            and retrain_state is not None
        ):
            from worldspace.surrogate.retrain import maybe_retrain_after_iteration

            maybe_retrain_after_iteration(
                config,
                iteration_index=iteration_index,
                state=retrain_state,
                surrogate=surrogate,
            )
    if surrogate_buffer is not None:
        surrogate_buffer.flush()
    if surrogate_archive is not None:
        surrogate_archive.flush()
    return counters


def _acquisition_logging_active(config: SchedulerConfig) -> bool:
    return config.surrogate_enabled and config.acquisition.mode != "off"


def _jsonl_schema_version_for_archive(archive: ArchiveProtocol) -> str:
    if archive.archive_type == "cvt":
        return ARCHIVE_SCHEMA_VERSION_V1_3
    return ARCHIVE_SCHEMA_VERSION


def _prepare_world_spec(spec: WorldSpec, *, grid_size: int, steps: int) -> WorldSpec:
    prepared = replace(spec)
    prepared.grid_size = grid_size
    prepared.steps = max(steps, ILLUMINATOR_MIN_STEPS)
    return prepared

"""MAP-Elites iteration loop: batch evaluate, insert, optional JSONL."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, TextIO

import numpy as np

from worldspace.illuminators.archive_trace import (
    ARCHIVE_TRACE_FILENAME,
    archive_trace_metrics,
    write_archive_trace_line,
)
from worldspace.illuminators.archive import (
    ARCHIVE_SCHEMA_VERSION,
    ARCHIVE_SCHEMA_VERSION_V1_3,
    EliteMetadata,
    InsertResult,
    insert_and_persist,
    insert_evaluated,
)
from worldspace.illuminators.archive_protocol import ArchiveProtocol
from worldspace.illuminators.emitters.base import (
    CandidateEmitter,
    EmitterOutput,
    MapElitesEmitter,
)
from worldspace.illuminators.emitters.llm_emitter import (
    LlmPreparedSlot,
    apply_batch_hint_placebo,
)
from worldspace.illuminators.evaluation import (
    ILLUMINATOR_MIN_STEPS,
    EvalResult,
    eval_result_from_simulation,
    evaluate_candidate,
    simulate_candidate,
)
from worldspace.illuminators.parallel_eval import (
    ParallelEvalPool,
    evaluate_batch_parallel,
    parallel_eval_context,
)
from worldspace.illuminators.parallel_llm_emit import (
    ParallelLlmPool,
    llm_slot_count_for_batch,
    parallel_llm_context,
    request_llm_batch,
    supports_parallel_llm_emit,
)
from worldspace.illuminators.scheduler import (
    EmitterKind,
    RunCounters,
    SchedulerConfig,
    TargetBin,
    TargetCell,
    resolve_emitter_for_slot,
    select_target_cell,
)
from worldspace.simulator_perf import (
    effective_llm_parallel_workers,
    effective_parallel_workers,
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
    from worldspace.illuminators.proposal_log import ProposalLogWriterProtocol
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


@dataclass(frozen=True)
class _SlotDraft:
    """Emitted candidate before acquisition / simulation."""

    candidate_id: int
    emitter_kind: EmitterKind
    target_cell: TargetCell
    target_bin: TargetBin
    spec: WorldSpec
    metadata: EliteMetadata


@dataclass(frozen=True)
class _SlotWorkItem:
    """Acquisition decision for one emitted slot."""

    draft: _SlotDraft
    prediction: SurrogatePrediction | None
    decision: AcquisitionDecision | None
    runtime_action: str


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
    proposal_log: ProposalLogWriterProtocol | None = None,
    eval_pool: ParallelEvalPool | None = None,
    llm_pool: ParallelLlmPool | None = None,
    iteration_timing_file: TextIO | None = None,
) -> tuple[IterationStats, list[SlotOutcome]]:
    """Run one batch: slots ``0 .. batch_size-1`` in order, evaluate or skip, insert.

    Candidates are processed in ascending ``candidate_id``. Together with strict
    ``fitness_new > fitness_old`` in ``GridArchive.try_insert``, equal fitness in the
    same bin within one iteration leaves the first accepted elite in place.

    Emission uses the archive state at iteration start (no intra-iteration inserts
    during the emit phase). With ``parallel_eval`` disabled, acquisition and eval run
    sequentially per slot. With ``parallel_eval`` enabled, simulations batch in parallel
    while insert/JSONL/buffer stay sequential.
    """
    acquisition_active = _acquisition_logging_active(config)

    emit_started = perf_counter()
    drafts = _emit_iteration_drafts(
        config,
        archive,
        rng,
        emitter,
        grid_size=grid_size,
        steps=steps,
        counters=counters,
        llm_pool=llm_pool,
    )
    emit_seconds = perf_counter() - emit_started
    _record_llm_emit_stats(counters, drafts)

    predictions: list[SurrogatePrediction | None]
    if config.surrogate_enabled and surrogate is not None:
        batch_predictions = surrogate.predict_batch([draft.spec for draft in drafts])
        predictions = list(batch_predictions)
    else:
        predictions = [None] * len(drafts)

    eval_started = perf_counter()
    if _use_parallel_eval_path(config):
        stats, outcomes = _process_iteration_parallel(
            config,
            archive,
            counters,
            drafts=drafts,
            predictions=predictions,
            iteration_index=iteration_index,
            jsonl_path=jsonl_path,
            surrogate_buffer=surrogate_buffer,
            surrogate_archive=surrogate_archive,
            proposal_log=proposal_log,
            acquisition_active=acquisition_active,
            eval_pool=eval_pool,
            rng=rng,
        )
    else:
        stats, outcomes = _process_iteration_sequential(
            config,
            archive,
            counters,
            drafts=drafts,
            predictions=predictions,
            iteration_index=iteration_index,
            jsonl_path=jsonl_path,
            surrogate_buffer=surrogate_buffer,
            surrogate=surrogate,
            surrogate_archive=surrogate_archive,
            proposal_log=proposal_log,
            acquisition_active=acquisition_active,
            rng=rng,
        )
    eval_seconds = perf_counter() - eval_started
    counters.emit_llm_seconds += emit_seconds
    counters.eval_seconds += eval_seconds
    if iteration_timing_file is not None:
        _write_iteration_timing(
            iteration_timing_file,
            iteration_index=iteration_index,
            emit_seconds=emit_seconds,
            eval_seconds=eval_seconds,
            drafts=drafts,
        )
    return stats, outcomes


def _use_parallel_eval_path(config: SchedulerConfig) -> bool:
    if not config.performance.parallel_eval:
        return False
    workers = effective_parallel_workers(
        config.performance,
        batch_size=config.batch_size,
    )
    return workers > 1


def _append_proposal_log(
    proposal_log: ProposalLogWriterProtocol | None,
    *,
    draft: _SlotDraft,
    iteration_index: int,
    eval_result: EvalResult,
    insert: InsertResult,
    incumbent_fitness: float | None,
    prediction: SurrogatePrediction | None,
) -> None:
    """Append one evaluated slot to the per-run proposal log (if enabled)."""
    if proposal_log is None:
        return
    proposal_log.append_evaluated(
        iteration=iteration_index,
        candidate_id=draft.candidate_id,
        emitter_type=draft.metadata.emitter_type,
        target=draft.target_bin,
        target_cell_id=draft.target_cell.cell_id,
        eval_result=eval_result,
        insert=insert,
        parent_id=draft.metadata.parent_id,
        incumbent_fitness=incumbent_fitness,
        prediction=prediction,
    )


def _process_iteration_sequential(
    config: SchedulerConfig,
    archive: ArchiveProtocol,
    counters: RunCounters,
    *,
    drafts: list[_SlotDraft],
    predictions: list[SurrogatePrediction | None],
    iteration_index: int,
    jsonl_path: str | Path | None,
    surrogate_buffer: SurrogateBuffer | None,
    surrogate: SurrogateProtocol | None,
    surrogate_archive: SurrogateArchiveWriterProtocol | None,
    proposal_log: ProposalLogWriterProtocol | None,
    acquisition_active: bool,
    rng: np.random.Generator,
) -> tuple[IterationStats, list[SlotOutcome]]:
    outcomes: list[SlotOutcome] = []
    evaluated = 0
    skipped = 0
    shadow_would_skip = 0
    accepted = 0
    improved = 0
    rejected = 0
    jsonl_schema_version = _jsonl_schema_version_for_archive(archive)

    for draft, prediction in zip(drafts, predictions):
        decision: AcquisitionDecision | None = None
        runtime_action = "eval"

        if config.surrogate_enabled and surrogate is not None:
            assert prediction is not None
            if acquisition_active:
                decision = decide(
                    config.acquisition,
                    prediction,
                    draft.target_bin,
                    archive,
                    rng=rng,
                )
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
                    candidate_id=draft.candidate_id,
                    emitter_type=draft.metadata.emitter_type,
                    target=draft.target_bin,
                    target_cell_id=draft.target_cell.cell_id,
                    world_spec_hash=world_spec_canonical_hash(draft.spec),
                    prediction=prediction,
                    decision=decision,
                    acquisition_mode=config.acquisition.mode,
                )
            outcomes.append(
                SlotOutcome(
                    candidate_id=draft.candidate_id,
                    emitter_kind=draft.emitter_kind,
                    target_bin=draft.target_bin,
                    skipped=True,
                    eval_result=None,
                    insert=None,
                    prediction=prediction,
                    decision=decision,
                )
            )
            continue

        eval_result = evaluate_candidate(
            draft.spec,
            resolution=config.grid_resolution,
            archive=archive,
            early_extinction_step=config.early_extinction_step,
            enforce_min_steps=True,
            performance=config.performance,
        )
        if config.surrogate_enabled and surrogate_buffer is not None:
            from worldspace.surrogate.buffer import append_eval_to_buffer

            append_eval_to_buffer(
                surrogate_buffer,
                eval_result,
                emitter_type=draft.metadata.emitter_type,
            )
        incumbent = archive.get_cell(archive.cell_id_from_bin(eval_result.bin))
        incumbent_fitness = float(incumbent.fitness) if incumbent is not None else None
        if jsonl_path is not None:
            insert = insert_and_persist(
                archive,
                eval_result,
                draft.metadata,
                jsonl_path,
                schema_version=jsonl_schema_version,
            )
        else:
            insert = insert_evaluated(archive, eval_result, draft.metadata)
        counters.record_evaluation()
        evaluated += 1

        if insert.accepted:
            accepted += 1
        if insert.improved:
            improved += 1
        if insert.rejected:
            rejected += 1

        _append_proposal_log(
            proposal_log,
            draft=draft,
            iteration_index=iteration_index,
            eval_result=eval_result,
            insert=insert,
            incumbent_fitness=incumbent_fitness,
            prediction=prediction,
        )

        if (
            acquisition_active
            and surrogate_archive is not None
            and prediction is not None
            and decision is not None
        ):
            surrogate_archive.append_slot(
                iteration=iteration_index,
                candidate_id=draft.candidate_id,
                emitter_type=draft.metadata.emitter_type,
                target=draft.target_bin,
                target_cell_id=draft.target_cell.cell_id,
                world_spec_hash=world_spec_canonical_hash(draft.spec),
                prediction=prediction,
                decision=decision,
                acquisition_mode=config.acquisition.mode,
                eval_result=eval_result,
                insert=insert,
            )

        outcomes.append(
            SlotOutcome(
                candidate_id=draft.candidate_id,
                emitter_kind=draft.emitter_kind,
                target_bin=draft.target_bin,
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


def _process_iteration_parallel(
    config: SchedulerConfig,
    archive: ArchiveProtocol,
    counters: RunCounters,
    *,
    drafts: list[_SlotDraft],
    predictions: list[SurrogatePrediction | None],
    iteration_index: int,
    jsonl_path: str | Path | None,
    surrogate_buffer: SurrogateBuffer | None,
    surrogate_archive: SurrogateArchiveWriterProtocol | None,
    proposal_log: ProposalLogWriterProtocol | None,
    acquisition_active: bool,
    eval_pool: ParallelEvalPool | None,
    rng: np.random.Generator,
) -> tuple[IterationStats, list[SlotOutcome]]:
    outcomes: list[SlotOutcome] = []
    evaluated = 0
    skipped = 0
    shadow_would_skip = 0
    accepted = 0
    improved = 0
    rejected = 0

    work_items = _classify_iteration_slots(
        drafts,
        predictions,
        config=config,
        archive=archive,
        acquisition_active=acquisition_active,
        rng=rng,
    )
    for item in work_items:
        if (
            acquisition_active
            and item.decision is not None
            and policy_recommends_skip(item.decision)
        ):
            shadow_would_skip += 1

    eval_results = _run_parallel_simulations(
        work_items,
        config=config,
        archive=archive,
        eval_pool=eval_pool,
    )

    jsonl_schema_version = _jsonl_schema_version_for_archive(archive)
    for item in work_items:
        draft = item.draft
        prediction = item.prediction
        decision = item.decision

        if item.runtime_action == "skip":
            assert prediction is not None and decision is not None
            skipped += 1
            if surrogate_archive is not None:
                surrogate_archive.append_slot(
                    iteration=iteration_index,
                    candidate_id=draft.candidate_id,
                    emitter_type=draft.metadata.emitter_type,
                    target=draft.target_bin,
                    target_cell_id=draft.target_cell.cell_id,
                    world_spec_hash=world_spec_canonical_hash(draft.spec),
                    prediction=prediction,
                    decision=decision,
                    acquisition_mode=config.acquisition.mode,
                )
            outcomes.append(
                SlotOutcome(
                    candidate_id=draft.candidate_id,
                    emitter_kind=draft.emitter_kind,
                    target_bin=draft.target_bin,
                    skipped=True,
                    eval_result=None,
                    insert=None,
                    prediction=prediction,
                    decision=decision,
                )
            )
            continue

        eval_result = eval_results[draft.candidate_id]
        if config.surrogate_enabled and surrogate_buffer is not None:
            from worldspace.surrogate.buffer import append_eval_to_buffer

            append_eval_to_buffer(
                surrogate_buffer,
                eval_result,
                emitter_type=draft.metadata.emitter_type,
            )
        incumbent = archive.get_cell(archive.cell_id_from_bin(eval_result.bin))
        incumbent_fitness = float(incumbent.fitness) if incumbent is not None else None
        if jsonl_path is not None:
            insert = insert_and_persist(
                archive,
                eval_result,
                draft.metadata,
                jsonl_path,
                schema_version=jsonl_schema_version,
            )
        else:
            insert = insert_evaluated(archive, eval_result, draft.metadata)
        counters.record_evaluation()
        evaluated += 1

        if insert.accepted:
            accepted += 1
        if insert.improved:
            improved += 1
        if insert.rejected:
            rejected += 1

        _append_proposal_log(
            proposal_log,
            draft=draft,
            iteration_index=iteration_index,
            eval_result=eval_result,
            insert=insert,
            incumbent_fitness=incumbent_fitness,
            prediction=prediction,
        )

        if (
            acquisition_active
            and surrogate_archive is not None
            and prediction is not None
            and decision is not None
        ):
            surrogate_archive.append_slot(
                iteration=iteration_index,
                candidate_id=draft.candidate_id,
                emitter_type=draft.metadata.emitter_type,
                target=draft.target_bin,
                target_cell_id=draft.target_cell.cell_id,
                world_spec_hash=world_spec_canonical_hash(draft.spec),
                prediction=prediction,
                decision=decision,
                acquisition_mode=config.acquisition.mode,
                eval_result=eval_result,
                insert=insert,
            )

        outcomes.append(
            SlotOutcome(
                candidate_id=draft.candidate_id,
                emitter_kind=draft.emitter_kind,
                target_bin=draft.target_bin,
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


def _classify_iteration_slots(
    drafts: list[_SlotDraft],
    predictions: list[SurrogatePrediction | None],
    *,
    config: SchedulerConfig,
    archive: ArchiveProtocol,
    acquisition_active: bool,
    rng: np.random.Generator,
) -> list[_SlotWorkItem]:
    work_items: list[_SlotWorkItem] = []
    for draft, prediction in zip(drafts, predictions):
        decision: AcquisitionDecision | None = None
        runtime_action = "eval"
        if config.surrogate_enabled and prediction is not None:
            if acquisition_active:
                decision = decide(
                    config.acquisition,
                    prediction,
                    draft.target_bin,
                    archive,
                    rng=rng,
                )
                runtime_action = effective_action(
                    config.acquisition.mode,
                    decision,
                )
        work_items.append(
            _SlotWorkItem(
                draft=draft,
                prediction=prediction,
                decision=decision,
                runtime_action=runtime_action,
            )
        )
    return work_items


def _run_parallel_simulations(
    work_items: list[_SlotWorkItem],
    *,
    config: SchedulerConfig,
    archive: ArchiveProtocol,
    eval_pool: ParallelEvalPool | None,
) -> dict[int, EvalResult]:
    to_eval = [item for item in work_items if item.runtime_action == "eval"]
    if not to_eval:
        return {}

    workers = effective_parallel_workers(
        config.performance,
        batch_size=len(to_eval),
    )
    use_parallel = config.performance.parallel_eval and len(to_eval) > 1 and workers > 1

    if use_parallel:
        if eval_pool is None:
            raise ValueError(
                "eval_pool is required for parallel eval batches; "
                "create via parallel_eval_context (see run_scheduler)."
            )
        outcomes = evaluate_batch_parallel(
            [item.draft.spec for item in to_eval],
            early_extinction_step=config.early_extinction_step,
            enforce_min_steps=True,
            performance=config.performance,
            workers=workers,
            eval_pool=eval_pool,
        )
    else:
        outcomes = [
            simulate_candidate(
                item.draft.spec,
                early_extinction_step=config.early_extinction_step,
                enforce_min_steps=True,
                performance=config.performance,
            )
            for item in to_eval
        ]

    return {
        item.draft.candidate_id: eval_result_from_simulation(
            outcome,
            resolution=config.grid_resolution,
            archive=archive,
        )
        for item, outcome in zip(to_eval, outcomes)
    }


def _emit_iteration_drafts(
    config: SchedulerConfig,
    archive: ArchiveProtocol,
    rng: np.random.Generator,
    emitter: CandidateEmitter,
    *,
    grid_size: int,
    steps: int,
    counters: RunCounters,
    llm_pool: ParallelLlmPool | None = None,
) -> list[_SlotDraft]:
    llm_slot_count = llm_slot_count_for_batch(
        config,
        candidates_evaluated=counters.candidates_evaluated,
    )
    workers = effective_llm_parallel_workers(
        config.performance,
        llm_slot_count=llm_slot_count,
    )
    if workers > 0 and supports_parallel_llm_emit(emitter, config):
        return _emit_iteration_drafts_parallel_llm(
            config,
            archive,
            rng,
            emitter,
            grid_size=grid_size,
            steps=steps,
            counters=counters,
            llm_workers=workers,
            llm_pool=llm_pool,
        )
    return _emit_iteration_drafts_sequential(
        config,
        archive,
        rng,
        emitter,
        grid_size=grid_size,
        steps=steps,
        counters=counters,
    )


def _emit_iteration_drafts_sequential(
    config: SchedulerConfig,
    archive: ArchiveProtocol,
    rng: np.random.Generator,
    emitter: CandidateEmitter,
    *,
    grid_size: int,
    steps: int,
    counters: RunCounters,
) -> list[_SlotDraft]:
    if config.llm_hint_placebo != "off":
        msg = (
            f"llm.hint_placebo={config.llm_hint_placebo!r} requires parallel LLM emit "
            "(batch prepare_emit before HTTP)"
        )
        raise ValueError(msg)
    return [
        _emit_one_slot_draft(
            config,
            archive,
            rng,
            emitter,
            candidate_id=candidate_id,
            grid_size=grid_size,
            steps=steps,
            counters=counters,
        )
        for candidate_id in range(config.batch_size)
    ]


def _emit_iteration_drafts_parallel_llm(
    config: SchedulerConfig,
    archive: ArchiveProtocol,
    rng: np.random.Generator,
    emitter: CandidateEmitter,
    *,
    grid_size: int,
    steps: int,
    counters: RunCounters,
    llm_workers: int,
    llm_pool: ParallelLlmPool | None = None,
) -> list[_SlotDraft]:
    if not isinstance(emitter, MapElitesEmitter):
        msg = "parallel LLM emit requires MapElitesEmitter"
        raise TypeError(msg)

    llm = emitter.llm_emitter
    drafts: list[_SlotDraft | None] = [None] * config.batch_size
    pending: list[tuple[int, LlmPreparedSlot]] = []

    for candidate_id in range(config.batch_size):
        emitter_kind = resolve_emitter_for_slot(
            config,
            candidate_id=candidate_id,
            candidates_evaluated=counters.candidates_evaluated,
        )
        target_cell = select_target_cell(
            archive,
            rng,
            target_selection=config.target_selection,
        )
        target_bin = TargetBin.from_target_cell(target_cell)
        if emitter_kind == "llm":
            prepared = llm.prepare_emit(
                target=target_cell,
                archive=archive,
                rng=rng,
                grid_size=grid_size,
                steps=steps,
            )
            pending.append((candidate_id, prepared))
        else:
            output = emitter.emit(
                emitter_kind=emitter_kind,
                target=target_cell,
                archive=archive,
                rng=rng,
                grid_size=grid_size,
                steps=steps,
            )
            drafts[candidate_id] = _slot_draft_from_output(
                candidate_id=candidate_id,
                emitter_kind=emitter_kind,
                target_cell=target_cell,
                target_bin=target_bin,
                output=output,
                grid_size=grid_size,
                steps=steps,
            )

    if pending:
        slot_ids = [candidate_id for candidate_id, _ in pending]
        prepared_slots = [slot for _, slot in pending]
        if config.llm_hint_placebo == "shuffle_batch":
            prepared_slots = apply_batch_hint_placebo(prepared_slots, rng)
            pending = list(zip(slot_ids, prepared_slots, strict=True))
        results = request_llm_batch(
            llm,
            prepared_slots,
            max_workers=llm_workers,
            llm_pool=llm_pool,
        )
        draft_outputs: dict[int, EmitterOutput] = {}
        rewrite_pending: list[
            tuple[int, LlmPreparedSlot, EmitterOutput, LlmPreparedSlot]
        ] = []
        for (candidate_id, prepared), http_result in zip(pending, results, strict=True):
            output = llm.finalize_emit(
                prepared,
                response=http_result.response,
                rng=rng,
                request_error=http_result.request_error,
            )
            rewrite_prepared = llm.prepare_child_rewrite(
                prepared, output, archive=archive, rng=rng
            )
            if rewrite_prepared is None:
                draft_outputs[candidate_id] = output
            else:
                rewrite_pending.append(
                    (candidate_id, prepared, output, rewrite_prepared)
                )

        if rewrite_pending:
            rewrite_slots = [slot for *_, slot in rewrite_pending]
            rewrite_results = request_llm_batch(
                llm,
                rewrite_slots,
                max_workers=llm_workers,
                llm_pool=llm_pool,
            )
            for (
                candidate_id,
                _prepared,
                draft_out,
                rewrite_prepared,
            ), http_result in zip(rewrite_pending, rewrite_results, strict=True):
                draft_outputs[candidate_id] = llm.commit_child_rewrite(
                    draft=draft_out,
                    rewrite_prepared=rewrite_prepared,
                    rewrite_response=http_result.response,
                    rng=rng,
                    request_error=http_result.request_error,
                )

        for candidate_id, prepared in pending:
            target_cell = prepared.target
            target_bin = TargetBin.from_target_cell(target_cell)
            output = draft_outputs[candidate_id]
            drafts[candidate_id] = _slot_draft_from_output(
                candidate_id=candidate_id,
                emitter_kind="llm",
                target_cell=target_cell,
                target_bin=target_bin,
                output=output,
                grid_size=grid_size,
                steps=steps,
            )

    result: list[_SlotDraft] = []
    for i in range(config.batch_size):
        draft = drafts[i]
        if draft is None:
            msg = f"parallel LLM emit left unfilled slot {i}"
            raise RuntimeError(msg)
        result.append(draft)
    return result


def _emit_one_slot_draft(
    config: SchedulerConfig,
    archive: ArchiveProtocol,
    rng: np.random.Generator,
    emitter: CandidateEmitter,
    *,
    candidate_id: int,
    grid_size: int,
    steps: int,
    counters: RunCounters,
) -> _SlotDraft:
    emitter_kind = resolve_emitter_for_slot(
        config,
        candidate_id=candidate_id,
        candidates_evaluated=counters.candidates_evaluated,
    )
    target_cell = select_target_cell(
        archive,
        rng,
        target_selection=config.target_selection,
    )
    target_bin = TargetBin.from_target_cell(target_cell)
    output = emitter.emit(
        emitter_kind=emitter_kind,
        target=target_cell,
        archive=archive,
        rng=rng,
        grid_size=grid_size,
        steps=steps,
    )
    return _slot_draft_from_output(
        candidate_id=candidate_id,
        emitter_kind=emitter_kind,
        target_cell=target_cell,
        target_bin=target_bin,
        output=output,
        grid_size=grid_size,
        steps=steps,
    )


def _slot_draft_from_output(
    *,
    candidate_id: int,
    emitter_kind: EmitterKind,
    target_cell: TargetCell,
    target_bin: TargetBin,
    output: EmitterOutput,
    grid_size: int,
    steps: int,
) -> _SlotDraft:
    spec = _prepare_world_spec(output.world_spec, grid_size=grid_size, steps=steps)
    return _SlotDraft(
        candidate_id=candidate_id,
        emitter_kind=emitter_kind,
        target_cell=target_cell,
        target_bin=target_bin,
        spec=spec,
        metadata=output.metadata,
    )


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
    proposal_log: ProposalLogWriterProtocol | None = None,
) -> RunCounters:
    """Run ``config.iterations`` batches and return updated global counters."""
    if counters is None:
        counters = RunCounters()
    eval_pool = parallel_eval_context(
        config.performance,
        batch_size=config.batch_size,
    )
    max_llm_slots = sum(1 for kind in config.batch_emitters if kind == "llm")
    llm_pool = parallel_llm_context(
        config.performance,
        max_llm_slots=max_llm_slots,
    )
    timing_file: TextIO | None = None
    trace_file: TextIO | None = None
    if config.performance.log_iteration_timing and jsonl_path is not None:
        run_dir = Path(jsonl_path).parent
        run_dir.mkdir(parents=True, exist_ok=True)
        timing_path = run_dir / "iteration_timing.jsonl"
        timing_file = timing_path.open("w", encoding="utf-8")
        trace_path = run_dir / ARCHIVE_TRACE_FILENAME
        trace_file = trace_path.open("w", encoding="utf-8")
        filled, coverage, mean_fit, qd_score = archive_trace_metrics(archive)
        write_archive_trace_line(
            trace_file,
            {
                "iteration": 0,
                "evaluations": counters.candidates_evaluated,
                "filled_cells": filled,
                "coverage": round(coverage, 6),
                "mean_best_fitness": (
                    round(mean_fit, 6) if mean_fit is not None else None
                ),
                "qd_score": round(qd_score, 6),
            },
        )
    try:
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
                proposal_log=proposal_log,
                eval_pool=eval_pool,
                llm_pool=llm_pool,
                iteration_timing_file=timing_file,
            )
            if trace_file is not None:
                filled, coverage, mean_fit, qd_score = archive_trace_metrics(archive)
                write_archive_trace_line(
                    trace_file,
                    {
                        "iteration": iteration_index,
                        "evaluations": counters.candidates_evaluated,
                        "filled_cells": filled,
                        "coverage": round(coverage, 6),
                        "mean_best_fitness": (
                            round(mean_fit, 6) if mean_fit is not None else None
                        ),
                        "qd_score": round(qd_score, 6),
                    },
                )
            if surrogate_buffer is not None:
                surrogate_buffer.flush()
            if surrogate_archive is not None:
                surrogate_archive.flush()
            if proposal_log is not None:
                proposal_log.flush()
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
    finally:
        if eval_pool is not None:
            eval_pool.close()
            eval_pool.join()
        if llm_pool is not None:
            llm_pool.shutdown()
        if timing_file is not None:
            timing_file.close()
        if trace_file is not None:
            trace_file.close()
    if surrogate_buffer is not None:
        surrogate_buffer.flush()
    if surrogate_archive is not None:
        surrogate_archive.flush()
    if proposal_log is not None:
        proposal_log.flush()
    return counters


def _record_llm_emit_stats(counters: RunCounters, drafts: list[_SlotDraft]) -> None:
    for draft in drafts:
        if draft.emitter_kind == "llm":
            counters.record_llm_emit(
                fallback=draft.metadata.emitter_type == "llm_fallback",
            )


def _write_iteration_timing(
    timing_file: TextIO,
    *,
    iteration_index: int,
    emit_seconds: float,
    eval_seconds: float,
    drafts: list[_SlotDraft],
) -> None:
    llm_slots = sum(1 for draft in drafts if draft.emitter_kind == "llm")
    llm_fallbacks = sum(
        1
        for draft in drafts
        if draft.emitter_kind == "llm" and draft.metadata.emitter_type == "llm_fallback"
    )
    timing_file.write(
        json.dumps(
            {
                "iteration": iteration_index,
                "emit_s": round(emit_seconds, 3),
                "eval_s": round(eval_seconds, 3),
                "llm_slots": llm_slots,
                "llm_fallbacks": llm_fallbacks,
            },
            ensure_ascii=True,
        )
        + "\n"
    )
    timing_file.flush()


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

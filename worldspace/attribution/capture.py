"""Opt-in prospective capture of normalized proposal-slot events."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from worldspace.attribution.manifest import BudgetAxis, RunManifest
from worldspace.attribution.records import (
    ArchiveState,
    BudgetCheckpoint,
    BudgetCounters,
    ProposalEvent,
    RunSummary,
    SourceCompleteness,
)

PROSPECTIVE_EVENT_FILENAME = "attribution_events.jsonl"
BUDGET_LEDGER_FILENAME = "budget_ledger.jsonl"


@dataclass
class ProspectiveEventCapture:
    """Validate and append complete normalized slot events for one run."""

    manifest: RunManifest
    path: Path | None = None
    ledger_path: Path | None = None
    llm_applicable: bool | None = None
    _events: list[ProposalEvent] = field(default_factory=list, init=False)
    _checkpoints: list[BudgetCheckpoint] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.ledger_path is None and self.path is not None:
            self.ledger_path = self.path.with_name(BUDGET_LEDGER_FILENAME)
        if self.llm_applicable is None:
            self.llm_applicable = self.manifest.treatment.generator.kind == "llm"
        if self.path is not None and self.path.exists() and self.path.stat().st_size:
            raise FileExistsError(
                f"prospective event log already exists and is non-empty: {self.path}"
            )
        if (
            self.ledger_path is not None
            and self.ledger_path.exists()
            and self.ledger_path.stat().st_size
        ):
            raise FileExistsError(
                f"budget ledger already exists and is non-empty: {self.ledger_path}"
            )

    @property
    def events(self) -> tuple[ProposalEvent, ...]:
        return tuple(self._events)

    @property
    def checkpoints(self) -> tuple[BudgetCheckpoint, ...]:
        return tuple(self._checkpoints)

    def append_slot(
        self,
        *,
        iteration: int,
        slot: int,
        configured_operator: str,
        realized_operator: str | None,
        target_cell_id: str | None,
        parent_id: str | None,
        parent_genotype_hash: str | None,
        candidate_id: str | None,
        candidate_genotype_hash: str | None,
        before: ArchiveState,
        generation: dict[str, Any],
        gate: dict[str, Any],
        evaluation: dict[str, Any],
        resources: dict[str, Any],
        after: ArchiveState,
    ) -> ProposalEvent:
        """Append exactly one slot in deterministic proposal order."""
        event = ProposalEvent.model_validate(
            {
                "run_id": self.manifest.run_id,
                "study_id": self.manifest.study_id,
                "arm_id": self.manifest.arm_id,
                "pair_id": self.manifest.pair_id,
                "proposal_index": len(self._events) + 1,
                "iteration": iteration,
                "slot": slot,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "configured_operator": configured_operator,
                "realized_operator": realized_operator,
                "target_cell_id": target_cell_id,
                "parent_id": parent_id,
                "parent_genotype_hash": parent_genotype_hash,
                "candidate_id": candidate_id,
                "candidate_genotype_hash": candidate_genotype_hash,
                "before": before,
                "generation": generation,
                "gate": gate,
                "evaluation": evaluation,
                "resources": resources,
                "after": after,
            }
        )
        if self._events and event.before != self._events[-1].after:
            raise ValueError(
                "prospective event before-state does not match previous after-state"
            )
        self._events.append(event)
        checkpoint = _next_budget_checkpoint(
            event,
            previous=self._checkpoints[-1] if self._checkpoints else None,
            llm_applicable=bool(self.llm_applicable),
        )
        self._checkpoints.append(checkpoint)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json() + "\n")
        if self.ledger_path is not None:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(checkpoint.model_dump_json() + "\n")
        return event


def read_prospective_events(
    path: Path,
    *,
    manifest: RunManifest,
) -> tuple[ProposalEvent, ...]:
    """Load a complete event sidecar and verify run identity and order."""
    events: list[ProposalEvent] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = ProposalEvent.model_validate_json(line)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid prospective event at {path}:{line_number}: {exc}"
                ) from exc
            expected_index = len(events) + 1
            if event.proposal_index != expected_index:
                raise ValueError(
                    f"prospective event index {event.proposal_index} "
                    f"does not match expected {expected_index}"
                )
            identities = (
                ("run_id", event.run_id, manifest.run_id),
                ("study_id", event.study_id, manifest.study_id),
                ("arm_id", event.arm_id, manifest.arm_id),
                ("pair_id", event.pair_id, manifest.pair_id),
            )
            for field_name, actual, expected in identities:
                if actual != expected:
                    raise ValueError(
                        f"prospective event {field_name}={actual!r} "
                        f"does not match manifest {expected!r}"
                    )
            if events and event.before != events[-1].after:
                raise ValueError(
                    "prospective event before-state does not match previous after-state"
                )
            events.append(event)
    return tuple(events)


def read_budget_ledger(
    path: Path,
    *,
    run_id: str,
) -> tuple[BudgetCheckpoint, ...]:
    """Load cumulative checkpoints and verify identity, order, and monotonicity."""
    checkpoints: list[BudgetCheckpoint] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                checkpoint = BudgetCheckpoint.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"invalid budget checkpoint at {path}:{line_number}: {exc}"
                ) from exc
            expected_index = len(checkpoints)
            if checkpoint.checkpoint_index != expected_index:
                raise ValueError(
                    f"budget checkpoint index {checkpoint.checkpoint_index} "
                    f"does not match expected {expected_index}"
                )
            if checkpoint.run_id != run_id:
                raise ValueError(
                    f"budget checkpoint run_id={checkpoint.run_id!r} "
                    f"does not match {run_id!r}"
                )
            if checkpoints:
                _require_monotone_counters(
                    checkpoints[-1].counters,
                    checkpoint.counters,
                )
            checkpoints.append(checkpoint)
    return tuple(checkpoints)


def archive_state_from_archive(archive: Any) -> ArchiveState:
    """Snapshot common archive metrics from a native archive protocol."""
    fitnesses = [
        float(elite.fitness)
        for cell_id in range(archive.n_cells)
        if (elite := archive.get_cell(cell_id)) is not None
    ]
    occupied = len(fitnesses)
    raw_qd = sum(fitnesses)
    return ArchiveState(
        occupied_cells=occupied,
        capacity=int(archive.n_cells),
        coverage=occupied / archive.n_cells,
        raw_qd_score=raw_qd,
        normalized_qd_score=raw_qd / archive.n_cells,
        maximum_elite_quality=max(fitnesses) if fitnesses else None,
        occupied_mean_quality=raw_qd / occupied if occupied else None,
    )


def _next_budget_checkpoint(
    event: ProposalEvent,
    *,
    previous: BudgetCheckpoint | None,
    llm_applicable: bool,
) -> BudgetCheckpoint:
    prior = previous.counters if previous is not None else None
    counters = BudgetCounters(
        proposal_slots=_increment(prior, "proposal_slots", 1),
        valid_proposals=_increment(
            prior,
            "valid_proposals",
            int(event.evaluation.attempted),
        ),
        evaluator_attempts=_increment(
            prior,
            "evaluator_attempts",
            int(event.evaluation.attempted),
        ),
        evaluator_completions=_increment(
            prior,
            "evaluator_completions",
            int(event.evaluation.completed),
        ),
        llm_attempts=_increment_resource(
            prior,
            "llm_attempts",
            event.resources.llm_calls_attempted,
            applicable=llm_applicable,
        ),
        llm_completions=_increment_resource(
            prior,
            "llm_completions",
            event.resources.llm_calls_completed,
            applicable=llm_applicable,
        ),
        prompt_tokens=_increment_resource(
            prior,
            "prompt_tokens",
            event.resources.prompt_tokens,
            applicable=llm_applicable,
        ),
        completion_tokens=_increment_resource(
            prior,
            "completion_tokens",
            event.resources.completion_tokens,
            applicable=llm_applicable,
        ),
        total_tokens=_increment_resource(
            prior,
            "total_tokens",
            event.resources.total_tokens,
            applicable=llm_applicable,
        ),
        evaluator_seconds=_increment_evaluator_seconds(prior, event),
        llm_latency_seconds=_increment_resource(
            prior,
            "llm_latency_seconds",
            event.resources.llm_latency_seconds,
            applicable=llm_applicable,
        ),
        wall_seconds=_increment_resource(
            prior,
            "wall_seconds",
            event.resources.event_seconds,
            applicable=True,
        ),
        monetary_cost=_increment_resource(
            prior,
            "monetary_cost",
            event.resources.monetary_cost,
            applicable=True,
        ),
    )
    completeness: dict[BudgetAxis, SourceCompleteness] = {
        "proposal": "observed",
        "valid_proposal": "observed",
        "evaluation": "observed",
        "llm_call_attempted": _counter_completeness(counters.llm_attempts),
        "llm_call_completed": _counter_completeness(counters.llm_completions),
        "prompt_token": _counter_completeness(counters.prompt_tokens),
        "completion_token": _counter_completeness(counters.completion_tokens),
        "token": _counter_completeness(counters.total_tokens),
        "evaluator_wall_time": _counter_completeness(counters.evaluator_seconds),
        "llm_latency": _counter_completeness(counters.llm_latency_seconds),
        "wall_time": _counter_completeness(counters.wall_seconds),
        "monetary": _counter_completeness(counters.monetary_cost),
    }
    calls_used = dict(previous.calls_used_by_operator) if previous is not None else {}
    attempted = event.resources.llm_calls_attempted
    if attempted is not None and attempted:
        calls_used[event.configured_operator] = (
            calls_used.get(event.configured_operator, 0) + attempted
        )
    return BudgetCheckpoint(
        run_id=event.run_id,
        checkpoint_index=0 if previous is None else previous.checkpoint_index + 1,
        indexed_by="proposal",
        counters=counters,
        source_completeness=completeness,
        calls_allocated_by_operator={},
        calls_used_by_operator=calls_used,
        calls_remaining_by_operator={},
        calls_forfeited_by_operator={},
        archive=event.after,
    )


def _increment(
    previous: BudgetCounters | None,
    field_name: str,
    delta: int,
) -> int:
    prior = 0 if previous is None else getattr(previous, field_name)
    assert prior is not None
    return int(prior) + delta


def _increment_resource(
    previous: BudgetCounters | None,
    field_name: str,
    delta: int | float | None,
    *,
    applicable: bool,
) -> Any:
    if not applicable or delta is None:
        return None
    prior = 0 if previous is None else getattr(previous, field_name)
    if prior is None:
        return None
    return prior + delta


def _increment_evaluator_seconds(
    previous: BudgetCounters | None,
    event: ProposalEvent,
) -> float | None:
    prior = 0.0 if previous is None else previous.evaluator_seconds
    if not event.evaluation.attempted:
        return prior
    delta = event.resources.evaluator_seconds
    if prior is None or delta is None:
        return None
    return prior + delta


def _counter_completeness(value: int | float | None) -> SourceCompleteness:
    return "observed" if value is not None else "unavailable"


def _require_monotone_counters(
    previous: BudgetCounters,
    current: BudgetCounters,
) -> None:
    for field_name in (
        "proposal_slots",
        "valid_proposals",
        "evaluator_attempts",
        "evaluator_completions",
        "llm_attempts",
        "llm_completions",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "evaluator_seconds",
        "llm_latency_seconds",
        "wall_seconds",
        "monetary_cost",
    ):
        before = getattr(previous, field_name)
        after = getattr(current, field_name)
        if before is not None and after is not None and after < before:
            raise ValueError(f"budget counter {field_name} is not monotone")


def event_budget_counters(
    events: tuple[ProposalEvent, ...],
    *,
    llm_applicable: bool,
) -> BudgetCounters:
    """Derive a null-preserving cumulative ledger from proposal events."""
    evaluated = tuple(event for event in events if event.evaluation.attempted)
    return BudgetCounters(
        proposal_slots=len(events),
        valid_proposals=len(evaluated),
        evaluator_attempts=len(evaluated),
        evaluator_completions=sum(event.evaluation.completed for event in evaluated),
        llm_attempts=_sum_event_int_resource(
            events,
            "llm_calls_attempted",
            applicable=llm_applicable,
        ),
        llm_completions=_sum_event_int_resource(
            events,
            "llm_calls_completed",
            applicable=llm_applicable,
        ),
        prompt_tokens=_sum_event_int_resource(
            events,
            "prompt_tokens",
            applicable=llm_applicable,
        ),
        completion_tokens=_sum_event_int_resource(
            events,
            "completion_tokens",
            applicable=llm_applicable,
        ),
        total_tokens=_sum_event_int_resource(
            events,
            "total_tokens",
            applicable=llm_applicable,
        ),
        evaluator_seconds=_sum_evaluator_seconds(evaluated),
        llm_latency_seconds=_sum_event_float_resource(
            events,
            "llm_latency_seconds",
            applicable=llm_applicable,
        ),
        wall_seconds=_sum_event_float_resource(
            events,
            "event_seconds",
            applicable=True,
        ),
        monetary_cost=_sum_event_float_resource(
            events,
            "monetary_cost",
            applicable=True,
        ),
    )


def reconcile_event_ledger(
    events: tuple[ProposalEvent, ...],
    summary: RunSummary,
    *,
    llm_applicable: bool,
    require_complete_llm_usage: bool = False,
) -> BudgetCounters:
    """Fail closed when a prospective event ledger disagrees with its summary."""
    if summary.event_completeness != "full":
        raise ValueError(
            "exact event-ledger reconciliation requires event_completeness='full'"
        )
    if not events:
        raise ValueError("exact event-ledger reconciliation requires events")
    ledger = event_budget_counters(events, llm_applicable=llm_applicable)
    for field_name in (
        "proposal_slots",
        "valid_proposals",
        "evaluator_attempts",
        "evaluator_completions",
    ):
        observed = getattr(ledger, field_name)
        reported = getattr(summary.final_counters, field_name)
        if observed != reported:
            raise ValueError(
                f"event ledger {field_name}={observed!r} disagrees with "
                f"summary {reported!r}"
            )
    for field_name in (
        "llm_attempts",
        "llm_completions",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "llm_latency_seconds",
        "evaluator_seconds",
        "wall_seconds",
        "monetary_cost",
    ):
        observed = getattr(ledger, field_name)
        reported = getattr(summary.final_counters, field_name)
        if observed is not None and reported is not None and observed != reported:
            raise ValueError(
                f"event ledger {field_name}={observed!r} disagrees with "
                f"summary {reported!r}"
            )
    if require_complete_llm_usage and llm_applicable:
        missing = [
            field
            for field in (
                "llm_attempts",
                "llm_completions",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "llm_latency_seconds",
            )
            if getattr(ledger, field) is None
        ]
        if missing:
            raise ValueError(
                f"prospective LLM event ledger has unavailable fields {missing!r}"
            )
    if events[-1].after != summary.final_archive:
        raise ValueError(
            "event ledger terminal archive disagrees with normalized summary"
        )
    return ledger


def reconcile_budget_ledger(
    checkpoints: tuple[BudgetCheckpoint, ...],
    events: tuple[ProposalEvent, ...],
    *,
    llm_applicable: bool,
) -> BudgetCounters:
    """Verify one cumulative checkpoint per event and exact terminal totals."""
    if len(checkpoints) != len(events):
        raise ValueError(
            f"budget ledger has {len(checkpoints)} checkpoints for "
            f"{len(events)} proposal events"
        )
    if not checkpoints:
        raise ValueError("budget ledger must not be empty")
    for checkpoint, event in zip(checkpoints, events, strict=True):
        if checkpoint.indexed_by != "proposal":
            raise ValueError("prospective budget ledger must be proposal-indexed")
        if checkpoint.checkpoint_index != event.proposal_index - 1:
            raise ValueError(
                "budget checkpoint index does not match proposal event index"
            )
        if checkpoint.archive != event.after:
            raise ValueError(
                "budget checkpoint archive does not match proposal event after-state"
            )
    expected = event_budget_counters(events, llm_applicable=llm_applicable)
    if checkpoints[-1].counters != expected:
        raise ValueError(
            "budget ledger terminal counters disagree with proposal-event totals"
        )
    return expected


def _sum_event_int_resource(
    events: tuple[ProposalEvent, ...],
    field_name: str,
    *,
    applicable: bool,
) -> int | None:
    if not applicable:
        return None
    values = [getattr(event.resources, field_name) for event in events]
    if not values or any(value is None for value in values):
        return None
    return sum(int(value) for value in values if value is not None)


def _sum_event_float_resource(
    events: tuple[ProposalEvent, ...],
    field_name: str,
    *,
    applicable: bool,
) -> float | None:
    if not applicable:
        return None
    values = [getattr(event.resources, field_name) for event in events]
    if not values or any(value is None for value in values):
        return None
    return sum(float(value) for value in values if value is not None)


def _sum_evaluator_seconds(
    evaluated: tuple[ProposalEvent, ...],
) -> float | None:
    values = [event.resources.evaluator_seconds for event in evaluated]
    if not values or any(value is None for value in values):
        return None
    return sum(float(value) for value in values if value is not None)

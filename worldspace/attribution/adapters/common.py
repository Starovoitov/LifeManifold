"""Shared normalization calculations for native domain adapters."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, cast

from worldspace.attribution.adapters.base import NormalizationError
from worldspace.attribution.manifest import BudgetAxis
from worldspace.attribution.records import (
    ArchiveState,
    BudgetCheckpoint,
    BudgetCounters,
    SourceCompleteness,
)


def archive_state_from_fitnesses(
    fitnesses: Iterable[float],
    *,
    capacity: int,
) -> ArchiveState:
    """Compute canonical archive metrics from final elite fitnesses."""
    values = tuple(float(value) for value in fitnesses)
    if capacity < 1:
        raise NormalizationError(f"archive capacity must be positive, got {capacity}")
    if len(values) > capacity:
        raise NormalizationError(
            f"archive has {len(values)} elites but capacity is {capacity}"
        )
    raw_qd = sum(values)
    return ArchiveState(
        occupied_cells=len(values),
        capacity=capacity,
        coverage=len(values) / capacity,
        raw_qd_score=raw_qd,
        normalized_qd_score=raw_qd / capacity,
        maximum_elite_quality=max(values) if values else None,
        occupied_mean_quality=raw_qd / len(values) if values else None,
    )


def assert_close(
    actual: float | None,
    expected: object,
    *,
    label: str,
    tolerance: float = 1e-6,
) -> None:
    """Fail closed when a native summary and derived metric disagree."""
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise NormalizationError(
                f"{label} mismatch: derived={actual!r}, native={expected!r}"
            )
        return
    try:
        expected_float = float(cast(Any, expected))
    except (TypeError, ValueError) as exc:
        raise NormalizationError(f"{label} is not numeric: {expected!r}") from exc
    if not math.isclose(actual, expected_float, rel_tol=tolerance, abs_tol=tolerance):
        raise NormalizationError(
            f"{label} mismatch: derived={actual!r}, native={expected_float!r}"
        )


def checkpoints_from_trace(
    rows: Iterable[Mapping[str, Any]],
    *,
    run_id: str,
    capacity: int,
) -> tuple[BudgetCheckpoint, ...]:
    """Convert native trace rows into proposal/evaluation-indexed checkpoints."""
    checkpoints: list[BudgetCheckpoint] = []
    previous_proposals: int | None = None
    previous_evaluations: int | None = None
    previous_occupied = 0
    for row in rows:
        proposals = _optional_int(row.get("proposals"))
        evaluations = _optional_int(row.get("evaluations"))
        occupied = _required_int(row.get("filled_cells"), "filled_cells")
        _require_monotone(previous_proposals, proposals, "proposals")
        _require_monotone(previous_evaluations, evaluations, "evaluations")
        if occupied < previous_occupied:
            raise NormalizationError("trace filled_cells must be monotone")
        previous_proposals = proposals
        previous_evaluations = evaluations
        previous_occupied = occupied
        raw_qd = _optional_float(row.get("qd_score"))
        mean_quality = _optional_float(row.get("mean_best_fitness"))
        assert_close(
            occupied / capacity,
            row.get("coverage"),
            label="trace coverage",
        )
        state = ArchiveState(
            occupied_cells=occupied,
            capacity=capacity,
            coverage=occupied / capacity,
            raw_qd_score=raw_qd,
            normalized_qd_score=(raw_qd / capacity if raw_qd is not None else None),
            maximum_elite_quality=None,
            occupied_mean_quality=mean_quality,
        )
        counters = BudgetCounters(
            proposal_slots=proposals,
            valid_proposals=None,
            evaluator_attempts=evaluations,
            evaluator_completions=evaluations,
            llm_attempts=None,
            llm_completions=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            evaluator_seconds=None,
            llm_latency_seconds=None,
            wall_seconds=None,
            monetary_cost=None,
        )
        completeness: dict[BudgetAxis, SourceCompleteness] = {
            "proposal": "observed" if proposals is not None else "unavailable",
            "evaluation": "observed" if evaluations is not None else "unavailable",
            "valid_proposal": "unavailable",
            "llm_call_attempted": "unavailable",
            "llm_call_completed": "unavailable",
            "prompt_token": "unavailable",
            "completion_token": "unavailable",
            "token": "unavailable",
            "evaluator_wall_time": "unavailable",
            "llm_latency": "unavailable",
            "wall_time": "unavailable",
            "monetary": "unavailable",
        }
        available_axes: list[BudgetAxis] = []
        if proposals is not None:
            available_axes.append("proposal")
        if evaluations is not None:
            available_axes.append("evaluation")
        for axis in available_axes:
            checkpoints.append(
                BudgetCheckpoint(
                    run_id=run_id,
                    checkpoint_index=len(checkpoints),
                    indexed_by=axis,
                    counters=counters,
                    source_completeness=completeness,
                    calls_allocated_by_operator={},
                    calls_used_by_operator={},
                    calls_remaining_by_operator={},
                    calls_forfeited_by_operator={},
                    archive=state,
                )
            )
    return tuple(checkpoints)


def terminal_checkpoint(
    *,
    run_id: str,
    counters: BudgetCounters,
    source_completeness: dict[BudgetAxis, SourceCompleteness],
    archive: ArchiveState,
) -> BudgetCheckpoint:
    """Build one terminal fallback when no native anytime trace exists."""
    if counters.evaluator_completions is not None:
        indexed_by: BudgetAxis = "evaluation"
    elif counters.proposal_slots is not None:
        indexed_by = "proposal"
    else:
        indexed_by = "wall_time"
    return BudgetCheckpoint(
        run_id=run_id,
        checkpoint_index=0,
        indexed_by=indexed_by,
        counters=counters,
        source_completeness=source_completeness,
        calls_allocated_by_operator={},
        calls_used_by_operator={},
        calls_remaining_by_operator={},
        calls_forfeited_by_operator={},
        archive=archive,
    )


def _require_monotone(
    previous: int | None,
    current: int | None,
    label: str,
) -> None:
    if previous is not None and current is not None and current < previous:
        raise NormalizationError(f"trace {label} must be monotone")


def _required_int(value: object, label: str) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        raise NormalizationError(f"trace field {label!r} is required")
    return parsed


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise NormalizationError(f"expected integer counter, got {value!r}")
    try:
        parsed = int(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise NormalizationError(f"expected integer counter, got {value!r}") from exc
    if parsed < 0:
        raise NormalizationError(f"counter cannot be negative, got {parsed}")
    return parsed


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise NormalizationError(f"expected numeric value, got {value!r}") from exc

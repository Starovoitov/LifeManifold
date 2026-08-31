"""Normalized event, budget, summary, and artifact records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal, Self, TypeVar

from pydantic import Field, model_validator

from worldspace.attribution.manifest import (
    BUDGET_AXES,
    SCHEMA_VERSION,
    AttributionModel,
    BudgetAxis,
    EvidenceTier,
    NonEmptyStr,
    Sha256,
)

EventCompleteness = Literal["full", "partial", "summary_only"]
SourceCompleteness = Literal["observed", "derived", "unavailable"]
KeyT = TypeVar("KeyT", bound=str)
ArchiveMetric = Literal[
    "coverage",
    "raw_qd_score",
    "normalized_qd_score",
    "maximum_elite_quality",
    "occupied_mean_quality",
]
ARCHIVE_METRICS: tuple[ArchiveMetric, ...] = (
    "coverage",
    "raw_qd_score",
    "normalized_qd_score",
    "maximum_elite_quality",
    "occupied_mean_quality",
)


class ArchiveState(AttributionModel):
    """Common archive metrics at one event or ledger checkpoint."""

    occupied_cells: int = Field(ge=0)
    capacity: int = Field(ge=1)
    coverage: float = Field(ge=0.0, le=1.0)
    raw_qd_score: float | None
    normalized_qd_score: float | None
    maximum_elite_quality: float | None
    occupied_mean_quality: float | None

    @model_validator(mode="after")
    def _validate_occupancy(self) -> Self:
        if self.occupied_cells > self.capacity:
            raise ValueError("occupied_cells cannot exceed archive capacity")
        expected = self.occupied_cells / self.capacity
        if not math.isclose(self.coverage, expected, abs_tol=1e-9):
            raise ValueError(
                f"coverage mismatch: expected {expected}, got {self.coverage}"
            )
        return self


class GenerationOutcome(AttributionModel):
    """Generation, validity, repair, and fallback result for one slot."""

    status: Literal["generated", "invalid", "failed"]
    parse_valid: bool | None
    structurally_valid: bool | None
    duplicate: bool | None
    repair_attempts: int = Field(ge=0)
    repair_outcome: NonEmptyStr | None
    fallback: bool
    fallback_cause: NonEmptyStr | None
    step_metrics: dict[str, float]

    @model_validator(mode="after")
    def _validate_fallback(self) -> Self:
        if self.fallback and self.fallback_cause is None:
            raise ValueError("fallback events require fallback_cause")
        if not self.fallback and self.fallback_cause is not None:
            raise ValueError("fallback_cause requires fallback=true")
        return self


class GateOutcome(AttributionModel):
    """Acquisition or validity-gate result."""

    mode: NonEmptyStr
    decision: Literal["evaluate", "skip", "reject"]
    reason: NonEmptyStr
    policy_version: NonEmptyStr


class EvaluationOutcome(AttributionModel):
    """Evaluator and archive-insertion result for one slot."""

    attempted: bool
    completed: bool
    evaluator_seed: int | None
    fitness: float | None
    descriptors: dict[str, float] | None
    realized_cell_id: NonEmptyStr | None
    incumbent_fitness: float | None
    insertion: Literal[
        "fill_empty",
        "improve",
        "occupied_not_better",
        "not_evaluated",
        "evaluation_failed",
    ]
    delta_qd: float

    @model_validator(mode="after")
    def _validate_evaluation(self) -> Self:
        if self.completed and not self.attempted:
            raise ValueError("completed evaluation must have been attempted")
        if self.completed:
            if self.fitness is None or self.descriptors is None:
                raise ValueError(
                    "completed evaluation requires fitness and descriptors"
                )
            if self.insertion in {"not_evaluated", "evaluation_failed"}:
                raise ValueError("completed evaluation requires insertion outcome")
        else:
            if self.insertion not in {"not_evaluated", "evaluation_failed"}:
                raise ValueError("incomplete evaluation cannot have archive insertion")
            if not math.isclose(self.delta_qd, 0.0, abs_tol=1e-12):
                raise ValueError("unevaluated event must have delta_qd=0")
        return self


class ResourceDelta(AttributionModel):
    """Per-event resource increments; ``None`` means unavailable."""

    llm_calls_attempted: int | None = Field(ge=0)
    llm_calls_completed: int | None = Field(ge=0)
    prompt_tokens: int | None = Field(ge=0)
    completion_tokens: int | None = Field(ge=0)
    total_tokens: int | None = Field(ge=0)
    llm_latency_seconds: float | None = Field(ge=0.0)
    evaluator_seconds: float | None = Field(ge=0.0)
    event_seconds: float | None = Field(ge=0.0)
    monetary_cost: float | None = Field(ge=0.0)
    price_table_id: NonEmptyStr

    @model_validator(mode="after")
    def _validate_resource_relations(self) -> Self:
        _lte_if_known(
            self.llm_calls_completed,
            self.llm_calls_attempted,
            "completed LLM calls",
            "attempted LLM calls",
        )
        if (
            self.prompt_tokens is not None
            and self.completion_tokens is not None
            and self.total_tokens is not None
            and self.prompt_tokens + self.completion_tokens != self.total_tokens
        ):
            raise ValueError("total_tokens must equal prompt + completion tokens")
        return self


class ProposalEvent(AttributionModel):
    """One normalized proposal-slot record."""

    schema_version: Literal["attribution-1.0"] = SCHEMA_VERSION
    run_id: NonEmptyStr
    study_id: NonEmptyStr
    arm_id: NonEmptyStr
    pair_id: NonEmptyStr
    proposal_index: int = Field(ge=1)
    iteration: int = Field(ge=0)
    slot: int = Field(ge=0)
    timestamp_utc: NonEmptyStr
    configured_operator: NonEmptyStr
    realized_operator: NonEmptyStr | None
    target_cell_id: NonEmptyStr | None
    parent_id: NonEmptyStr | None
    parent_genotype_hash: Sha256 | None
    candidate_id: NonEmptyStr | None
    candidate_genotype_hash: Sha256 | None
    before: ArchiveState
    generation: GenerationOutcome
    gate: GateOutcome
    evaluation: EvaluationOutcome
    resources: ResourceDelta
    after: ArchiveState

    @model_validator(mode="after")
    def _validate_transition(self) -> Self:
        if self.before.raw_qd_score is None or self.after.raw_qd_score is None:
            raise ValueError("proposal events require known before/after raw QD-score")
        observed = self.after.raw_qd_score - self.before.raw_qd_score
        if not math.isclose(
            observed,
            self.evaluation.delta_qd,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"delta_qd mismatch: archive transition is {observed}, "
                f"event reports {self.evaluation.delta_qd}"
            )
        if self.after.occupied_cells < self.before.occupied_cells:
            raise ValueError("archive occupancy cannot decrease")
        if self.after.capacity != self.before.capacity:
            raise ValueError("archive capacity cannot change within a proposal")
        return self


class BudgetCounters(AttributionModel):
    """Cumulative resource counters at one ledger checkpoint."""

    proposal_slots: int | None = Field(ge=0)
    valid_proposals: int | None = Field(ge=0)
    evaluator_attempts: int | None = Field(ge=0)
    evaluator_completions: int | None = Field(ge=0)
    llm_attempts: int | None = Field(ge=0)
    llm_completions: int | None = Field(ge=0)
    prompt_tokens: int | None = Field(ge=0)
    completion_tokens: int | None = Field(ge=0)
    total_tokens: int | None = Field(ge=0)
    evaluator_seconds: float | None = Field(ge=0.0)
    llm_latency_seconds: float | None = Field(ge=0.0)
    wall_seconds: float | None = Field(ge=0.0)
    monetary_cost: float | None = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_counter_relations(self) -> Self:
        _lte_if_known(
            self.valid_proposals,
            self.proposal_slots,
            "valid proposals",
            "proposal slots",
        )
        _lte_if_known(
            self.evaluator_attempts,
            self.proposal_slots,
            "evaluator attempts",
            "proposal slots",
        )
        _lte_if_known(
            self.evaluator_completions,
            self.evaluator_attempts,
            "evaluator completions",
            "evaluator attempts",
        )
        _lte_if_known(
            self.llm_completions,
            self.llm_attempts,
            "LLM completions",
            "LLM attempts",
        )
        if (
            self.prompt_tokens is not None
            and self.completion_tokens is not None
            and self.total_tokens is not None
            and self.prompt_tokens + self.completion_tokens != self.total_tokens
        ):
            raise ValueError("total_tokens must equal prompt + completion tokens")
        return self


class BudgetCheckpoint(AttributionModel):
    """One cumulative budget-ledger row."""

    schema_version: Literal["attribution-1.0"] = SCHEMA_VERSION
    run_id: NonEmptyStr
    checkpoint_index: int = Field(ge=0)
    indexed_by: BudgetAxis
    counters: BudgetCounters
    source_completeness: dict[BudgetAxis, SourceCompleteness]
    calls_allocated_by_operator: dict[str, int]
    calls_used_by_operator: dict[str, int]
    calls_remaining_by_operator: dict[str, int]
    calls_forfeited_by_operator: dict[str, int]
    archive: ArchiveState

    @model_validator(mode="after")
    def _validate_allocations(self) -> Self:
        _require_exact_keys(
            self.source_completeness,
            set(BUDGET_AXES),
            "budget checkpoint source_completeness",
        )
        for name, values in (
            ("calls_allocated_by_operator", self.calls_allocated_by_operator),
            ("calls_used_by_operator", self.calls_used_by_operator),
            ("calls_remaining_by_operator", self.calls_remaining_by_operator),
            ("calls_forfeited_by_operator", self.calls_forfeited_by_operator),
        ):
            if any(value < 0 for value in values.values()):
                raise ValueError(f"{name} cannot contain negative counts")
        return self


class RunSummary(AttributionModel):
    """Normalized terminal result for one run."""

    schema_version: Literal["attribution-1.0"] = SCHEMA_VERSION
    run_id: NonEmptyStr
    study_id: NonEmptyStr
    arm_id: NonEmptyStr
    pair_id: NonEmptyStr
    evidence_tier: EvidenceTier
    domain_id: NonEmptyStr
    domain_version: NonEmptyStr
    adapter_id: NonEmptyStr
    adapter_version: NonEmptyStr
    evaluator_hash: Sha256
    seed: int
    domain_instance_id: NonEmptyStr
    initial_archive_hash: Sha256
    protocol_hash: Sha256
    study_manifest_hash: Sha256
    run_manifest_hash: Sha256
    treatment_hash: Sha256
    event_completeness: EventCompleteness
    final_counters: BudgetCounters
    counter_completeness: dict[BudgetAxis, SourceCompleteness]
    final_archive: ArchiveState
    archive_metric_completeness: dict[ArchiveMetric, SourceCompleteness]
    completed: bool
    failure_reason: NonEmptyStr | None

    @model_validator(mode="after")
    def _validate_completion(self) -> Self:
        _require_exact_keys(
            self.counter_completeness,
            set(BUDGET_AXES),
            "run summary counter_completeness",
        )
        _require_exact_keys(
            self.archive_metric_completeness,
            set(ARCHIVE_METRICS),
            "run summary archive_metric_completeness",
        )
        if self.completed and self.failure_reason is not None:
            raise ValueError("completed run cannot have failure_reason")
        if not self.completed and self.failure_reason is None:
            raise ValueError("incomplete run requires failure_reason")
        return self


class ArtifactEntry(AttributionModel):
    """Integrity and release metadata for one run artifact."""

    logical_name: NonEmptyStr
    path: NonEmptyStr
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    schema_version: NonEmptyStr | None
    privacy_class: Literal["public", "private", "discard"]
    producer: NonEmptyStr


class ArtifactManifest(AttributionModel):
    """Integrity manifest for all files in one run bundle."""

    schema_version: Literal["attribution-1.0"] = SCHEMA_VERSION
    run_id: NonEmptyStr
    run_manifest_hash: Sha256
    artifacts: tuple[ArtifactEntry, ...]

    @model_validator(mode="after")
    def _unique_artifacts(self) -> Self:
        if not self.artifacts:
            raise ValueError("artifact manifest must not be empty")
        names = [item.logical_name for item in self.artifacts]
        paths = [item.path for item in self.artifacts]
        if len(set(names)) != len(names):
            raise ValueError("artifact logical names must be unique")
        if len(set(paths)) != len(paths):
            raise ValueError("artifact paths must be unique")
        return self


def _lte_if_known(
    left: int | None,
    right: int | None,
    left_name: str,
    right_name: str,
) -> None:
    if left is not None and right is not None and left > right:
        raise ValueError(f"{left_name} cannot exceed {right_name}")


def _require_exact_keys(
    value: Mapping[KeyT, object],
    expected: set[KeyT],
    label: str,
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")

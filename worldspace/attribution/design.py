"""Frozen design-matrix and job-plan records for factorial expansion."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from worldspace.attribution.hashing import canonical_sha256
from worldspace.attribution.manifest import (
    SCHEMA_VERSION,
    AttributionModel,
    BudgetAxis,
    EvidenceTier,
    NonEmptyStr,
    RunManifest,
    Sha256,
    TreatmentAxis,
)


class DesignCell(AttributionModel):
    """One planned arm × seed × domain-instance × block cell."""

    run_id: NonEmptyStr
    arm_id: NonEmptyStr
    arm_label: NonEmptyStr
    pair_id: NonEmptyStr
    block_id: NonEmptyStr
    seed: int
    domain_instance_id: NonEmptyStr
    initial_archive_id: NonEmptyStr
    initial_archive_hash: Sha256
    treatment_hash: Sha256
    run_manifest_hash: Sha256
    output_dir: NonEmptyStr


class PlannedContrast(AttributionModel):
    """Exact component diff for one planned treatment/control pair."""

    estimand_id: NonEmptyStr
    treatment_arm_id: NonEmptyStr
    control_arm_id: NonEmptyStr
    differing_axes: tuple[TreatmentAxis, ...]
    controlled_axes_match: bool


class ResourceProjection(AttributionModel):
    """Projected resource totals before any job is launched."""

    run_count: int = Field(ge=0)
    proposal_slots: int | None = Field(ge=0)
    evaluator_calls: int | None = Field(ge=0)
    llm_calls_attempted: int | None = Field(ge=0)
    total_tokens: int | None = Field(ge=0)
    monetary_cost: float = Field(ge=0.0)
    approved_total_cost: float | None = Field(ge=0.0)
    within_approved_cap: bool


class DesignMatrix(AttributionModel):
    """Explicit frozen expansion used by analysis admission."""

    schema_version: Literal["attribution-1.0"] = SCHEMA_VERSION
    study_id: NonEmptyStr
    study_manifest_hash: Sha256
    evidence_tier: EvidenceTier
    domain_id: NonEmptyStr
    protocol_hash: Sha256
    cells: tuple[DesignCell, ...]
    pair_map: dict[NonEmptyStr, tuple[NonEmptyStr, ...]]
    contrasts: tuple[PlannedContrast, ...]
    design_matrix_hash: Sha256

    @model_validator(mode="after")
    def _verify_hash(self) -> DesignMatrix:
        observed = canonical_sha256(
            self,
            omit_keys=frozenset({"design_matrix_hash"}),
        )
        if observed != self.design_matrix_hash:
            raise ValueError(
                f"design_matrix_hash mismatch: expected {observed}, "
                f"got {self.design_matrix_hash}"
            )
        run_ids = [cell.run_id for cell in self.cells]
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("design matrix run_id values must be unique")
        return self


class JobPlan(AttributionModel):
    """Builder output: manifests, design matrix, projection, and order."""

    schema_version: Literal["attribution-1.0"] = SCHEMA_VERSION
    study_id: NonEmptyStr
    study_manifest_hash: Sha256
    evidence_tier: EvidenceTier
    runs: tuple[RunManifest, ...]
    design: DesignMatrix
    projection: ResourceProjection
    execution_order: dict[NonEmptyStr, tuple[NonEmptyStr, ...]]
    preflight_issues: tuple[str, ...]
    launched: Literal[False] = False

    @model_validator(mode="after")
    def _consistent_plan(self) -> JobPlan:
        if self.launched:
            raise ValueError("job builder must not mark plans as launched")
        if self.study_manifest_hash != self.design.study_manifest_hash:
            raise ValueError("job plan study hash must match design matrix")
        run_ids = {run.run_id for run in self.runs}
        design_ids = {cell.run_id for cell in self.design.cells}
        if run_ids != design_ids:
            raise ValueError("job plan runs must match design matrix cells")
        ordered = [
            run_id for block in self.execution_order.values() for run_id in block
        ]
        if set(ordered) != run_ids or len(ordered) != len(run_ids):
            raise ValueError("execution_order must list each run exactly once")
        return self


class CellMean(AttributionModel):
    """Descriptive absolute cell mean for one arm."""

    arm_id: NonEmptyStr
    endpoint: NonEmptyStr
    budget_axis: BudgetAxis
    n: int = Field(ge=0)
    mean: float | None


class PairedDifference(AttributionModel):
    """One complete pair's treatment − control endpoint difference."""

    pair_id: NonEmptyStr
    treatment_arm_id: NonEmptyStr
    control_arm_id: NonEmptyStr
    treatment_value: float
    control_value: float
    difference: float


class InteractionDifference(AttributionModel):
    """One pair's 2x2 interaction (A−B)−(C−D)."""

    pair_id: NonEmptyStr
    formula: NonEmptyStr
    minuend_treatment_arm_id: NonEmptyStr
    minuend_control_arm_id: NonEmptyStr
    subtrahend_treatment_arm_id: NonEmptyStr
    subtrahend_control_arm_id: NonEmptyStr
    minuend: float
    subtrahend: float
    difference: float


class ArmAnatomy(AttributionModel):
    """Companion validity / fallback / insertion rates for one arm."""

    arm_id: NonEmptyStr
    n_events: int | None = Field(ge=0)
    valid_proposal_rate: float | None
    fallback_rate: float | None
    fill_empty_count: int | None = Field(ge=0)
    improve_count: int | None = Field(ge=0)
    occupied_not_better_count: int | None = Field(ge=0)


class DescriptiveContrast(AttributionModel):
    """Fixture-level descriptive readout for one estimand."""

    estimand_id: NonEmptyStr
    endpoint: NonEmptyStr
    budget_axis: BudgetAxis
    form: Literal["terminal", "anytime_auc", "interaction"] = "terminal"
    cell_means: tuple[CellMean, ...]
    paired_differences: tuple[PairedDifference, ...]
    interaction_differences: tuple[InteractionDifference, ...] = ()
    anatomy: tuple[ArmAnatomy, ...] = ()
    mean_difference: float | None
    complete_pairs: int = Field(ge=0)

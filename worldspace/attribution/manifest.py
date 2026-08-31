"""Typed manifests for the controlled-attribution harness."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

from worldspace.attribution.hashing import canonical_sha256

SCHEMA_VERSION = "attribution-1.0"

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
EvidenceTier = Literal["feasibility", "design_pilot", "confirmatory", "robustness"]
ArmRole = Literal["focal", "baseline", "control", "sensitivity"]
BudgetAxis = Literal[
    "proposal",
    "valid_proposal",
    "evaluation",
    "llm_call_attempted",
    "llm_call_completed",
    "prompt_token",
    "completion_token",
    "token",
    "evaluator_wall_time",
    "llm_latency",
    "wall_time",
    "monetary",
]
TreatmentAxis = Literal[
    "initialization",
    "selector",
    "generator",
    "prompt_channel",
    "repair_fallback",
    "gate",
    "replacement",
    "allocation",
    "budget",
    "representation",
    "model",
    "evaluator",
]


class AttributionModel(BaseModel):
    """Strict immutable base for all attribution schema records."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ComponentSpec(AttributionModel):
    """One resolved versioned scientific component."""

    kind: NonEmptyStr
    version: NonEmptyStr
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    content_hashes: dict[str, Sha256] = Field(default_factory=dict)


class BudgetCaps(AttributionModel):
    """Independent run caps; ``None`` means explicitly uncapped."""

    proposal_slots: int | None = Field(ge=0)
    valid_proposals: int | None = Field(ge=0)
    evaluator_calls: int | None = Field(ge=0)
    llm_calls_attempted: int | None = Field(ge=0)
    llm_calls_completed: int | None = Field(ge=0)
    prompt_tokens: int | None = Field(ge=0)
    completion_tokens: int | None = Field(ge=0)
    total_tokens: int | None = Field(ge=0)
    evaluator_wall_seconds: float | None = Field(ge=0.0)
    llm_latency_seconds: float | None = Field(ge=0.0)
    wall_seconds: float | None = Field(ge=0.0)
    monetary_cost: float | None = Field(ge=0.0)


class BudgetTreatment(ComponentSpec):
    """Budget component with caps, analysis axes, and stopping precedence."""

    caps: BudgetCaps
    indexing_axes: tuple[BudgetAxis, ...]
    stopping_precedence: tuple[BudgetAxis, ...]

    @model_validator(mode="after")
    def _unique_axes(self) -> Self:
        if len(set(self.indexing_axes)) != len(self.indexing_axes):
            raise ValueError("budget indexing_axes must be unique")
        if len(set(self.stopping_precedence)) != len(self.stopping_precedence):
            raise ValueError("budget stopping_precedence must be unique")
        return self


class TreatmentVector(AttributionModel):
    """Mandatory nine-component pipeline treatment."""

    initialization: ComponentSpec
    selector: ComponentSpec
    generator: ComponentSpec
    prompt_channel: ComponentSpec
    repair_fallback: ComponentSpec
    gate: ComponentSpec
    replacement: ComponentSpec
    allocation: ComponentSpec
    budget: BudgetTreatment


class ArmManifest(AttributionModel):
    """One complete resolved experiment arm."""

    arm_id: NonEmptyStr
    label: NonEmptyStr
    role: ArmRole
    treatment: TreatmentVector
    representation: ComponentSpec
    model: ComponentSpec
    evaluator: ComponentSpec
    reference_arm_id: NonEmptyStr | None = None
    expected_differences: tuple[TreatmentAxis, ...] = ()

    @model_validator(mode="after")
    def _reference_required_for_differences(self) -> Self:
        if self.expected_differences and self.reference_arm_id is None:
            raise ValueError("expected_differences require reference_arm_id")
        if len(set(self.expected_differences)) != len(self.expected_differences):
            raise ValueError("expected_differences must be unique")
        if self.reference_arm_id == self.arm_id:
            raise ValueError("an arm cannot reference itself")
        return self


class EstimandSpec(AttributionModel):
    """One predeclared run-level contrast or interaction."""

    estimand_id: NonEmptyStr
    endpoint: NonEmptyStr
    form: Literal["terminal", "anytime_auc", "interaction"]
    budget_axis: BudgetAxis
    treatment_arm_ids: tuple[NonEmptyStr, ...]
    control_arm_ids: tuple[NonEmptyStr, ...]
    paired_by: tuple[NonEmptyStr, ...]
    alternative: Literal[
        "greater", "less", "two_sided", "equivalence", "non_inferiority"
    ]
    margin: float | None
    interaction_formula: NonEmptyStr | None = None
    confirmatory_family: NonEmptyStr | None = None
    multiplicity_rule: NonEmptyStr | None = None
    missing_policy: NonEmptyStr
    minimum_complete_pairs: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_form(self) -> Self:
        if not self.treatment_arm_ids or not self.control_arm_ids:
            raise ValueError("estimand must identify treatment and control arms")
        if self.form == "interaction" and self.interaction_formula is None:
            raise ValueError("interaction estimand requires interaction_formula")
        if self.alternative in {"equivalence", "non_inferiority"} and self.margin is None:
            raise ValueError(f"{self.alternative} estimand requires a margin")
        return self


class ReplicationPlan(AttributionModel):
    """Run seeds and scientific blocking variables."""

    seeds: tuple[int, ...]
    domain_instance_ids: tuple[NonEmptyStr, ...]
    api_block_ids: tuple[NonEmptyStr, ...] = ()
    paired_by: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def _nonempty_unique_values(self) -> Self:
        if not self.seeds:
            raise ValueError("replication seeds must not be empty")
        if not self.domain_instance_ids:
            raise ValueError("domain_instance_ids must not be empty")
        for name, values in (
            ("seeds", self.seeds),
            ("domain_instance_ids", self.domain_instance_ids),
            ("api_block_ids", self.api_block_ids),
            ("paired_by", self.paired_by),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique")
        return self


class CostPolicy(AttributionModel):
    """Dated price-table identity and study-level resource approval."""

    currency: NonEmptyStr
    price_table_id: NonEmptyStr
    price_table_hash: Sha256
    approved_total_cost: float | None = Field(ge=0.0)
    missing_usage_policy: Literal["reject", "report_missing", "estimate_separately"]


class FailurePolicy(AttributionModel):
    """Prospective treatment of generation, API, and evaluator failures."""

    generation_failure: Literal["consume_and_continue", "abort_run"]
    evaluator_failure: Literal["consume_and_continue", "retry", "abort_run"]
    maximum_evaluator_retries: int = Field(ge=0)
    incomplete_run: Literal["exclude", "retain_with_failure_endpoint", "abort_study"]


class PrivacyPolicy(AttributionModel):
    """Release class for raw prompts and provider responses."""

    raw_prompts: Literal["public", "private", "discard"]
    raw_responses: Literal["public", "private", "discard"]
    publish_sanitized_events: bool


class AdapterCapabilities(AttributionModel):
    """Machine-readable declaration of one adapter's supported treatments."""

    schema_version: Literal["attribution-1.0"] = SCHEMA_VERSION
    adapter_id: NonEmptyStr
    adapter_version: NonEmptyStr
    domain_id: NonEmptyStr
    initialization_kinds: tuple[NonEmptyStr, ...]
    selectors: tuple[NonEmptyStr, ...]
    generators: tuple[NonEmptyStr, ...]
    prompt_channels: tuple[NonEmptyStr, ...]
    repair_fallback_kinds: tuple[NonEmptyStr, ...]
    gate_modes: tuple[NonEmptyStr, ...]
    replacement_kinds: tuple[NonEmptyStr, ...]
    allocation_kinds: tuple[NonEmptyStr, ...]
    budget_axes: tuple[BudgetAxis, ...]
    archive_types: tuple[NonEmptyStr, ...]
    supports_full_proposal_log: bool
    supports_warm_start: bool
    stochastic_evaluation: bool
    native_fitness_min: float
    native_fitness_max: float
    empty_cell_fitness: float

    @model_validator(mode="after")
    def _validate_capabilities(self) -> Self:
        collections = (
            self.initialization_kinds,
            self.selectors,
            self.generators,
            self.prompt_channels,
            self.repair_fallback_kinds,
            self.gate_modes,
            self.replacement_kinds,
            self.allocation_kinds,
            self.budget_axes,
            self.archive_types,
        )
        if any(not values for values in collections):
            raise ValueError("capability collections must not be empty")
        if self.native_fitness_max <= self.native_fitness_min:
            raise ValueError("native fitness maximum must exceed minimum")
        if not self.native_fitness_min <= self.empty_cell_fitness <= self.native_fitness_max:
            raise ValueError("empty-cell fitness must lie within native fitness bounds")
        return self


class StudyManifest(AttributionModel):
    """Frozen experiment-family design before job expansion."""

    schema_version: Literal["attribution-1.0"] = SCHEMA_VERSION
    study_id: NonEmptyStr
    programme_id: NonEmptyStr
    protocol_id: NonEmptyStr
    protocol_hash: Sha256
    evidence_tier: EvidenceTier
    domain_id: NonEmptyStr
    domain_version: NonEmptyStr
    adapter_id: NonEmptyStr
    adapter_version: NonEmptyStr
    task_instance_set: tuple[NonEmptyStr, ...]
    estimands: tuple[EstimandSpec, ...]
    arms: tuple[ArmManifest, ...]
    replication: ReplicationPlan
    cost_policy: CostPolicy
    failure_policy: FailurePolicy
    privacy_policy: PrivacyPolicy

    @model_validator(mode="after")
    def _validate_design_graph(self) -> Self:
        if not self.task_instance_set:
            raise ValueError("task_instance_set must not be empty")
        if not self.arms:
            raise ValueError("arms must not be empty")
        if not self.estimands:
            raise ValueError("estimands must not be empty")
        arm_by_id = _unique_by_id(self.arms, "arm_id", "arm")
        _unique_by_id(self.estimands, "estimand_id", "estimand")
        for arm in self.arms:
            if arm.reference_arm_id is None:
                continue
            reference = arm_by_id.get(arm.reference_arm_id)
            if reference is None:
                raise ValueError(
                    f"arm {arm.arm_id!r} references unknown arm "
                    f"{arm.reference_arm_id!r}"
                )
            observed = differing_treatment_axes(arm, reference)
            expected = set(arm.expected_differences)
            if observed != expected:
                raise ValueError(
                    f"arm {arm.arm_id!r} expected differences {sorted(expected)}, "
                    f"observed {sorted(observed)}"
                )
        known_arms = set(arm_by_id)
        for estimand in self.estimands:
            referenced = set(estimand.treatment_arm_ids) | set(
                estimand.control_arm_ids
            )
            unknown = referenced - known_arms
            if unknown:
                raise ValueError(
                    f"estimand {estimand.estimand_id!r} references unknown arms "
                    f"{sorted(unknown)}"
                )
        return self


class RunManifestCore(AttributionModel):
    """Resolved scientific and execution identity before self-hashing."""

    schema_version: Literal["attribution-1.0"] = SCHEMA_VERSION
    run_id: NonEmptyStr
    study_id: NonEmptyStr
    arm_id: NonEmptyStr
    pair_id: NonEmptyStr
    block_id: NonEmptyStr
    evidence_tier: EvidenceTier
    protocol_id: NonEmptyStr
    protocol_hash: Sha256
    domain_id: NonEmptyStr
    domain_version: NonEmptyStr
    adapter_id: NonEmptyStr
    adapter_version: NonEmptyStr
    seed: int
    domain_instance_id: NonEmptyStr
    initial_archive_id: NonEmptyStr
    initial_archive_hash: Sha256
    treatment: TreatmentVector
    representation: ComponentSpec
    model: ComponentSpec
    evaluator: ComponentSpec
    treatment_hash: Sha256
    study_manifest_hash: Sha256
    dependency_hashes: dict[str, Sha256]
    output_paths: dict[str, NonEmptyStr]
    expected_artifacts: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def _verify_treatment_hash(self) -> Self:
        observed = scientific_treatment_hash(
            self.treatment,
            representation=self.representation,
            model=self.model,
            evaluator=self.evaluator,
        )
        if observed != self.treatment_hash:
            raise ValueError(
                f"treatment_hash mismatch: expected {observed}, "
                f"got {self.treatment_hash}"
            )
        if not self.expected_artifacts:
            raise ValueError("expected_artifacts must not be empty")
        return self


class RunManifest(RunManifestCore):
    """Immutable resolved run manifest with a non-self-referential hash."""

    run_manifest_hash: Sha256

    @model_validator(mode="after")
    def _verify_run_hash(self) -> Self:
        observed = canonical_sha256(
            self,
            omit_keys=frozenset({"run_manifest_hash"}),
        )
        if observed != self.run_manifest_hash:
            raise ValueError(
                f"run_manifest_hash mismatch: expected {observed}, "
                f"got {self.run_manifest_hash}"
            )
        return self


def scientific_treatment_hash(
    treatment: TreatmentVector,
    *,
    representation: ComponentSpec,
    model: ComponentSpec,
    evaluator: ComponentSpec,
) -> str:
    """Hash all scientific treatment components, excluding arm labels."""
    return canonical_sha256(
        {
            "treatment": treatment.model_dump(mode="json"),
            "representation": representation.model_dump(mode="json"),
            "model": model.model_dump(mode="json"),
            "evaluator": evaluator.model_dump(mode="json"),
        }
    )


def arm_treatment_hash(arm: ArmManifest) -> str:
    """Return the scientific treatment hash for one arm."""
    return scientific_treatment_hash(
        arm.treatment,
        representation=arm.representation,
        model=arm.model,
        evaluator=arm.evaluator,
    )


def study_manifest_hash(study: StudyManifest) -> str:
    """Return a canonical hash of one validated study design."""
    return canonical_sha256(study)


def freeze_run_manifest(payload: Mapping[str, Any] | RunManifestCore) -> RunManifest:
    """Validate a resolved run core, compute its hash, and freeze it."""
    if isinstance(payload, RunManifestCore):
        unresolved = payload.model_dump(
            mode="json",
            exclude={"run_manifest_hash"},
        )
        core = RunManifestCore.model_validate(unresolved)
    else:
        core = RunManifestCore.model_validate(payload)
    core_payload = core.model_dump(mode="json")
    core_payload["run_manifest_hash"] = canonical_sha256(core)
    return RunManifest.model_validate(core_payload)


def differing_treatment_axes(
    arm: ArmManifest,
    reference: ArmManifest,
) -> set[TreatmentAxis]:
    """Return exact scientific component differences between two arms."""
    differences: set[TreatmentAxis] = set()
    for axis in (
        "initialization",
        "selector",
        "generator",
        "prompt_channel",
        "repair_fallback",
        "gate",
        "replacement",
        "allocation",
        "budget",
    ):
        if getattr(arm.treatment, axis) != getattr(reference.treatment, axis):
            differences.add(axis)  # type: ignore[arg-type]
    for axis in ("representation", "model", "evaluator"):
        if getattr(arm, axis) != getattr(reference, axis):
            differences.add(axis)  # type: ignore[arg-type]
    return differences


def _unique_by_id(
    values: tuple[AttributionModel, ...],
    field: str,
    label: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        identifier = str(getattr(value, field))
        if identifier in result:
            raise ValueError(f"duplicate {label} id {identifier!r}")
        result[identifier] = value
    return result

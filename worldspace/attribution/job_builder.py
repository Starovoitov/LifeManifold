"""Factorial job expansion without launching domain runners."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from worldspace.attribution.design import (
    DesignCell,
    DesignMatrix,
    JobPlan,
    PlannedContrast,
    ResourceProjection,
)
from worldspace.attribution.hashing import canonical_sha256
from worldspace.attribution.manifest import (
    AdapterCapabilities,
    ArmManifest,
    RunManifest,
    RunManifestCore,
    StudyManifest,
    arm_treatment_hash,
    differing_treatment_axes,
    freeze_run_manifest,
    study_manifest_hash,
)
from worldspace.attribution.validation import (
    AdmissionIssue,
    AttributionAdmissionError,
    validate_study_capabilities,
)


@dataclass(frozen=True)
class InitialArchiveRef:
    """Fingerprint for one domain-instance initial archive."""

    archive_id: str
    archive_hash: str


@dataclass(frozen=True)
class JobBuildContext:
    """Non-scientific expansion inputs required to freeze run manifests."""

    output_root: str
    initial_archives: Mapping[str, InitialArchiveRef]
    dependency_hashes: Mapping[str, str]
    expected_artifacts: tuple[str, ...]
    unit_monetary_cost: float = 0.0
    order_seed: int = 0
    reserved_pilot_seeds: frozenset[int] = frozenset()
    reserved_pilot_archive_hashes: frozenset[str] = frozenset()
    aliased_arm_groups: tuple[frozenset[str], ...] = ()
    seed_api_blocks: Mapping[int, str] | None = None


def build_factorial_job_plan(
    study: StudyManifest,
    capabilities: AdapterCapabilities,
    context: JobBuildContext,
) -> JobPlan:
    """Expand a study into frozen run manifests and a design matrix.

    This function never launches jobs. Callers must use a separate explicit
    execution command.
    """
    issues = list(validate_study_capabilities(study, capabilities))
    issues.extend(_validate_build_context(study, context))
    issues.extend(_duplicate_treatment_issues(study, context.aliased_arm_groups))
    issues.extend(_pilot_contamination_issues(study, context))
    if issues:
        raise AttributionAdmissionError(issues)

    study_hash = study_manifest_hash(study)
    arm_by_id = {arm.arm_id: arm for arm in study.arms}
    contrasts = _planned_contrasts(study, arm_by_id)

    seed_blocks = _seed_block_assignment(study, context)
    runs: list[RunManifest] = []
    cells: list[DesignCell] = []
    pair_map: dict[str, list[str]] = defaultdict(list)

    for arm in study.arms:
        treatment_hash = arm_treatment_hash(arm)
        for seed in study.replication.seeds:
            block_id = seed_blocks[seed]
            for domain_instance_id in study.replication.domain_instance_ids:
                archive = context.initial_archives[domain_instance_id]
                pair_id = _pair_id(
                    study, seed=seed, domain_instance_id=domain_instance_id
                )
                run_id = _run_id(
                    study,
                    arm_id=arm.arm_id,
                    seed=seed,
                    domain_instance_id=domain_instance_id,
                    block_id=block_id,
                )
                output_dir = "/".join(
                    (
                        context.output_root.rstrip("/"),
                        study.study_id,
                        study.evidence_tier,
                        study.domain_id,
                        arm.arm_id,
                        run_id,
                    )
                )
                core = RunManifestCore(
                    run_id=run_id,
                    study_id=study.study_id,
                    arm_id=arm.arm_id,
                    pair_id=pair_id,
                    block_id=block_id,
                    evidence_tier=study.evidence_tier,
                    protocol_id=study.protocol_id,
                    protocol_hash=study.protocol_hash,
                    domain_id=study.domain_id,
                    domain_version=study.domain_version,
                    adapter_id=study.adapter_id,
                    adapter_version=study.adapter_version,
                    seed=seed,
                    domain_instance_id=domain_instance_id,
                    initial_archive_id=archive.archive_id,
                    initial_archive_hash=archive.archive_hash,
                    treatment=arm.treatment,
                    representation=arm.representation,
                    model=arm.model,
                    evaluator=arm.evaluator,
                    treatment_hash=treatment_hash,
                    study_manifest_hash=study_hash,
                    currency=study.cost_policy.currency,
                    price_table_id=study.cost_policy.price_table_id,
                    price_table_hash=study.cost_policy.price_table_hash,
                    dependency_hashes=dict(context.dependency_hashes),
                    output_paths={"run_dir": output_dir},
                    expected_artifacts=context.expected_artifacts,
                )
                frozen = freeze_run_manifest(core)
                runs.append(frozen)
                cells.append(
                    DesignCell(
                        run_id=run_id,
                        arm_id=arm.arm_id,
                        arm_label=arm.label,
                        pair_id=pair_id,
                        block_id=block_id,
                        seed=seed,
                        domain_instance_id=domain_instance_id,
                        initial_archive_id=archive.archive_id,
                        initial_archive_hash=archive.archive_hash,
                        treatment_hash=treatment_hash,
                        run_manifest_hash=frozen.run_manifest_hash,
                        output_dir=output_dir,
                    )
                )
                pair_map[pair_id].append(run_id)

    projection = _project_resources(study, run_count=len(runs), context=context)
    if not projection.within_approved_cap:
        raise AttributionAdmissionError(
            [
                AdmissionIssue(
                    code="builder.budget_cap",
                    message=(
                        f"projected monetary cost {projection.monetary_cost} "
                        f"exceeds approved cap {projection.approved_total_cost}"
                    ),
                )
            ]
        )

    pair_map_frozen = {
        pair_id: tuple(sorted(run_ids)) for pair_id, run_ids in sorted(pair_map.items())
    }
    design_payload = {
        "schema_version": "attribution-1.0",
        "study_id": study.study_id,
        "study_manifest_hash": study_hash,
        "evidence_tier": study.evidence_tier,
        "domain_id": study.domain_id,
        "protocol_hash": study.protocol_hash,
        "cells": [cell.model_dump(mode="json") for cell in cells],
        "pair_map": pair_map_frozen,
        "contrasts": [item.model_dump(mode="json") for item in contrasts],
    }
    design_payload["design_matrix_hash"] = canonical_sha256(
        design_payload,
        omit_keys=frozenset({"design_matrix_hash"}),
    )
    design = DesignMatrix.model_validate(design_payload)
    execution_order = _interleaved_execution_order(
        cells,
        order_seed=context.order_seed,
    )
    return JobPlan(
        study_id=study.study_id,
        study_manifest_hash=study_hash,
        evidence_tier=study.evidence_tier,
        runs=tuple(runs),
        design=design,
        projection=projection,
        execution_order=execution_order,
        preflight_issues=(),
        launched=False,
    )


def _validate_build_context(
    study: StudyManifest,
    context: JobBuildContext,
) -> list[AdmissionIssue]:
    issues: list[AdmissionIssue] = []
    if not context.output_root.strip():
        issues.append(
            AdmissionIssue("builder.output_root", "output_root must not be empty")
        )
    if not context.expected_artifacts:
        issues.append(
            AdmissionIssue(
                "builder.expected_artifacts",
                "expected_artifacts must not be empty",
            )
        )
    if context.unit_monetary_cost < 0:
        issues.append(
            AdmissionIssue(
                "builder.unit_cost",
                "unit_monetary_cost must be non-negative",
            )
        )
    missing_instances = [
        instance_id
        for instance_id in study.replication.domain_instance_ids
        if instance_id not in context.initial_archives
    ]
    if missing_instances:
        issues.append(
            AdmissionIssue(
                code="builder.initial_archive",
                message=(
                    "missing initial archive refs for domain instances "
                    f"{missing_instances!r}"
                ),
            )
        )
    unknown_instances = sorted(
        set(context.initial_archives) - set(study.replication.domain_instance_ids)
    )
    if unknown_instances:
        issues.append(
            AdmissionIssue(
                code="builder.initial_archive_extra",
                message=f"unknown domain instances in archive map {unknown_instances!r}",
            )
        )
    declared_blocks = study.replication.api_block_ids
    if declared_blocks:
        if context.seed_api_blocks is None:
            issues.append(
                AdmissionIssue(
                    code="builder.seed_api_blocks",
                    message="seed_api_blocks required when api_block_ids are declared",
                )
            )
        else:
            for seed in study.replication.seeds:
                block = context.seed_api_blocks.get(seed)
                if block is None:
                    issues.append(
                        AdmissionIssue(
                            code="builder.seed_api_blocks",
                            message=f"seed {seed} lacks an API block assignment",
                        )
                    )
                elif block not in declared_blocks:
                    issues.append(
                        AdmissionIssue(
                            code="builder.seed_api_blocks",
                            message=(
                                f"seed {seed} assigned unknown API block {block!r}"
                            ),
                        )
                    )
            extra_seeds = sorted(
                set(context.seed_api_blocks) - set(study.replication.seeds)
            )
            if extra_seeds:
                issues.append(
                    AdmissionIssue(
                        code="builder.seed_api_blocks_extra",
                        message=f"seed_api_blocks has unknown seeds {extra_seeds!r}",
                    )
                )
    elif context.seed_api_blocks:
        issues.append(
            AdmissionIssue(
                code="builder.seed_api_blocks_unexpected",
                message="seed_api_blocks provided but study declares no api_block_ids",
            )
        )
    known_arms = {arm.arm_id for arm in study.arms}
    for group in context.aliased_arm_groups:
        if len(group) < 2:
            issues.append(
                AdmissionIssue(
                    code="builder.alias_group",
                    message="aliased arm groups must contain at least two arm_ids",
                )
            )
        unknown = sorted(group - known_arms)
        if unknown:
            issues.append(
                AdmissionIssue(
                    code="builder.alias_unknown_arm",
                    message=f"alias group references unknown arms {unknown!r}",
                )
            )
    return issues


def _duplicate_treatment_issues(
    study: StudyManifest,
    aliased_arm_groups: Sequence[frozenset[str]],
) -> list[AdmissionIssue]:
    issues: list[AdmissionIssue] = []
    by_hash: dict[str, list[ArmManifest]] = defaultdict(list)
    for arm in study.arms:
        by_hash[arm_treatment_hash(arm)].append(arm)
    alias_sets = tuple(aliased_arm_groups)
    for treatment_hash, arms in by_hash.items():
        if len(arms) < 2:
            continue
        arm_ids = frozenset(arm.arm_id for arm in arms)
        labels = {arm.label for arm in arms}
        if arm_ids in alias_sets:
            continue
        if len(labels) > 1 or len(arm_ids) > 1:
            issues.append(
                AdmissionIssue(
                    code="builder.duplicate_treatment_hash",
                    message=(
                        "arms "
                        f"{sorted(arm_ids)!r} share treatment hash "
                        f"{treatment_hash[:12]}… under labels "
                        f"{sorted(labels)!r} without a declared alias group"
                    ),
                )
            )
    return issues


def _pilot_contamination_issues(
    study: StudyManifest,
    context: JobBuildContext,
) -> list[AdmissionIssue]:
    issues: list[AdmissionIssue] = []
    if study.evidence_tier != "confirmatory":
        return issues
    contaminated_seeds = sorted(
        set(study.replication.seeds) & set(context.reserved_pilot_seeds)
    )
    if contaminated_seeds:
        issues.append(
            AdmissionIssue(
                code="builder.pilot_seed",
                message=(
                    "confirmatory study reuses reserved design-pilot seeds "
                    f"{contaminated_seeds!r}"
                ),
            )
        )
    archive_hashes = {ref.archive_hash for ref in context.initial_archives.values()}
    contaminated_archives = sorted(
        archive_hashes & set(context.reserved_pilot_archive_hashes)
    )
    if contaminated_archives:
        issues.append(
            AdmissionIssue(
                code="builder.pilot_archive",
                message=(
                    "confirmatory study reuses reserved design-pilot archives "
                    f"{[item[:12] + '…' for item in contaminated_archives]!r}"
                ),
            )
        )
    return issues


def _planned_contrasts(
    study: StudyManifest,
    arm_by_id: Mapping[str, ArmManifest],
) -> tuple[PlannedContrast, ...]:
    contrasts: list[PlannedContrast] = []
    for estimand in study.estimands:
        for treatment_id in estimand.treatment_arm_ids:
            for control_id in estimand.control_arm_ids:
                treatment = arm_by_id[treatment_id]
                control = arm_by_id[control_id]
                axes = tuple(sorted(differing_treatment_axes(treatment, control)))
                controlled = False
                if treatment.reference_arm_id == control_id:
                    controlled = set(axes) == set(treatment.expected_differences)
                elif control.reference_arm_id == treatment_id:
                    controlled = set(axes) == set(control.expected_differences)
                contrasts.append(
                    PlannedContrast(
                        estimand_id=estimand.estimand_id,
                        treatment_arm_id=treatment_id,
                        control_arm_id=control_id,
                        differing_axes=axes,  # type: ignore[arg-type]
                        controlled_axes_match=controlled,
                    )
                )
    return tuple(contrasts)


def _seed_block_assignment(
    study: StudyManifest,
    context: JobBuildContext,
) -> dict[int, str]:
    if not study.replication.api_block_ids:
        return {seed: "local" for seed in study.replication.seeds}
    assert context.seed_api_blocks is not None
    return {seed: context.seed_api_blocks[seed] for seed in study.replication.seeds}


def _pair_id(study: StudyManifest, *, seed: int, domain_instance_id: str) -> str:
    parts: list[str] = ["pair"]
    for key in study.replication.paired_by:
        if key == "seed":
            parts.append(f"seed-{seed}")
        elif key == "domain_instance_id":
            parts.append(domain_instance_id)
        else:
            parts.append(f"{key}-unspecified")
    return "__".join(parts)


def _run_id(
    study: StudyManifest,
    *,
    arm_id: str,
    seed: int,
    domain_instance_id: str,
    block_id: str,
) -> str:
    return "__".join(
        (
            study.study_id,
            arm_id,
            f"seed-{seed}",
            domain_instance_id,
            block_id,
        )
    )


def _project_resources(
    study: StudyManifest,
    *,
    run_count: int,
    context: JobBuildContext,
) -> ResourceProjection:
    proposal_slots = 0
    evaluator_calls = 0
    llm_attempts = 0
    total_tokens = 0
    proposals_uncapped = False
    evaluators_uncapped = False
    llm_uncapped = False
    tokens_uncapped = False
    for arm in study.arms:
        caps = arm.treatment.budget.caps
        arm_runs = run_count // len(study.arms)
        if caps.proposal_slots is None:
            proposals_uncapped = True
        else:
            proposal_slots += caps.proposal_slots * arm_runs
        if caps.evaluator_calls is None:
            evaluators_uncapped = True
        else:
            evaluator_calls += caps.evaluator_calls * arm_runs
        if caps.llm_calls_attempted is None:
            llm_uncapped = True
        else:
            llm_attempts += caps.llm_calls_attempted * arm_runs
        if caps.total_tokens is None:
            tokens_uncapped = True
        else:
            total_tokens += caps.total_tokens * arm_runs
    monetary = context.unit_monetary_cost * run_count
    approved = study.cost_policy.approved_total_cost
    return ResourceProjection(
        run_count=run_count,
        proposal_slots=None if proposals_uncapped else proposal_slots,
        evaluator_calls=None if evaluators_uncapped else evaluator_calls,
        llm_calls_attempted=None if llm_uncapped else llm_attempts,
        total_tokens=None if tokens_uncapped else total_tokens,
        monetary_cost=monetary,
        approved_total_cost=approved,
        within_approved_cap=approved is None or monetary <= approved,
    )


def _interleaved_execution_order(
    cells: Sequence[DesignCell],
    *,
    order_seed: int,
) -> dict[str, tuple[str, ...]]:
    by_block: dict[str, list[DesignCell]] = defaultdict(list)
    for cell in cells:
        by_block[cell.block_id].append(cell)
    ordered: dict[str, tuple[str, ...]] = {}
    for block_id in sorted(by_block):
        block_cells = by_block[block_id]
        arm_ids = sorted({cell.arm_id for cell in block_cells})
        rng = random.Random(
            canonical_sha256(
                {"order_seed": order_seed, "block_id": block_id, "arms": arm_ids}
            )
        )
        arm_order = list(arm_ids)
        rng.shuffle(arm_order)
        by_seed: dict[int, dict[str, DesignCell]] = defaultdict(dict)
        for cell in block_cells:
            by_seed[cell.seed][cell.arm_id] = cell
        sequence: list[str] = []
        for seed in sorted(by_seed):
            for arm_id in arm_order:
                sequence.append(by_seed[seed][arm_id].run_id)
        ordered[block_id] = tuple(sequence)
    return ordered

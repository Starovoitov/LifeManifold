"""Analysis admission and descriptive estimand readouts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from worldspace.attribution.design import (
    CellMean,
    DescriptiveContrast,
    DesignMatrix,
    PairedDifference,
)
from worldspace.attribution.manifest import (
    EstimandSpec,
    StudyManifest,
    study_manifest_hash,
)
from worldspace.attribution.records import RunSummary
from worldspace.attribution.validation import (
    AdmissionIssue,
    AttributionAdmissionError,
    admit_analysis_cohort,
)

_ENDPOINT_GETTERS: dict[str, str] = {
    "normalized_qd_score": "normalized_qd_score",
    "raw_qd_score": "raw_qd_score",
    "coverage": "coverage",
    "occupied_cells": "occupied_cells",
    "maximum_elite_quality": "maximum_elite_quality",
    "occupied_mean_quality": "occupied_mean_quality",
}


def admit_design_matrix_cohort(
    summaries: Iterable[RunSummary],
    *,
    study: StudyManifest,
    estimand: EstimandSpec,
    design: DesignMatrix,
) -> tuple[RunSummary, ...]:
    """Admit rows only against an explicit frozen design matrix.

    Glob discovery is insufficient: every summary must match a design cell, and
    confirmatory cohorts reject pilot or feasibility contamination.
    """
    rows = tuple(summaries)
    issues: list[AdmissionIssue] = []
    if estimand.estimand_id not in {item.estimand_id for item in study.estimands}:
        raise AttributionAdmissionError(
            [
                AdmissionIssue(
                    code="analysis.unknown_estimand",
                    message=f"estimand {estimand.estimand_id!r} is not in the study",
                )
            ]
        )
    study_hash = study_manifest_hash(study)
    if design.study_manifest_hash != study_hash:
        issues.append(
            AdmissionIssue(
                code="analysis.design_study_hash",
                message="design matrix study_manifest_hash does not match study",
            )
        )
    if design.study_id != study.study_id:
        issues.append(
            AdmissionIssue(
                code="analysis.design_study_id",
                message="design matrix study_id does not match study manifest",
            )
        )
    if design.evidence_tier != study.evidence_tier:
        issues.append(
            AdmissionIssue(
                code="analysis.design_tier",
                message="design matrix evidence_tier does not match study manifest",
            )
        )
    if design.domain_id != study.domain_id:
        issues.append(
            AdmissionIssue(
                code="analysis.design_domain",
                message="design matrix domain_id does not match study manifest",
            )
        )
    if design.protocol_hash != study.protocol_hash:
        issues.append(
            AdmissionIssue(
                code="analysis.design_protocol",
                message="design matrix protocol_hash does not match study manifest",
            )
        )

    expected_arms = frozenset(estimand.treatment_arm_ids) | frozenset(
        estimand.control_arm_ids
    )
    design_by_run = {cell.run_id: cell for cell in design.cells}
    if study.evidence_tier == "confirmatory" and design.evidence_tier != "confirmatory":
        issues.append(
            AdmissionIssue(
                code="analysis.confirmatory_design",
                message="confirmatory analysis requires a confirmatory design matrix",
            )
        )
    if design.evidence_tier == "confirmatory":
        for row in rows:
            if row.evidence_tier in {"feasibility", "design_pilot"}:
                issues.append(
                    AdmissionIssue(
                        code="analysis.pilot_contamination",
                        message=(
                            f"run {row.run_id!r} has pilot/feasibility tier "
                            f"{row.evidence_tier!r} in a confirmatory design"
                        ),
                    )
                )

    seen_run_ids: set[str] = set()
    for row in rows:
        cell = design_by_run.get(row.run_id)
        if cell is None:
            issues.append(
                AdmissionIssue(
                    code="analysis.undeclared_run",
                    message=f"run {row.run_id!r} is absent from the design matrix",
                )
            )
            continue
        if row.run_id in seen_run_ids:
            issues.append(
                AdmissionIssue(
                    code="analysis.duplicate_run",
                    message=f"run {row.run_id!r} appears more than once",
                )
            )
        seen_run_ids.add(row.run_id)
        checks = (
            ("arm_id", row.arm_id, cell.arm_id),
            ("pair_id", row.pair_id, cell.pair_id),
            ("seed", row.seed, cell.seed),
            ("domain_instance_id", row.domain_instance_id, cell.domain_instance_id),
            (
                "initial_archive_hash",
                row.initial_archive_hash,
                cell.initial_archive_hash,
            ),
            ("treatment_hash", row.treatment_hash, cell.treatment_hash),
            ("run_manifest_hash", row.run_manifest_hash, cell.run_manifest_hash),
            ("study_manifest_hash", row.study_manifest_hash, study_hash),
            ("evidence_tier", row.evidence_tier, design.evidence_tier),
            ("domain_id", row.domain_id, design.domain_id),
        )
        for field, actual, expected in checks:
            if actual != expected:
                issues.append(
                    AdmissionIssue(
                        code=f"analysis.cell_mismatch_{field}",
                        message=(
                            f"run {row.run_id!r} {field}={actual!r} does not match "
                            f"design cell {expected!r}"
                        ),
                    )
                )
        if cell.arm_id not in expected_arms:
            issues.append(
                AdmissionIssue(
                    code="analysis.arm_outside_estimand",
                    message=(
                        f"run {row.run_id!r} arm {cell.arm_id!r} is outside "
                        f"estimand {estimand.estimand_id!r}"
                    ),
                )
            )

    if issues:
        raise AttributionAdmissionError(issues)

    admitted = admit_analysis_cohort(
        rows,
        expected_arm_ids=expected_arms,
        minimum_complete_pairs=estimand.minimum_complete_pairs,
    )
    return admitted


def descriptive_contrast(
    summaries: Iterable[RunSummary],
    *,
    estimand: EstimandSpec,
) -> DescriptiveContrast:
    """Compute absolute cell means and paired differences for fixtures."""
    rows = tuple(summaries)
    endpoint = estimand.endpoint
    values_by_arm: dict[str, list[float]] = defaultdict(list)
    by_pair: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = _endpoint_value(row, endpoint)
        if value is None:
            continue
        values_by_arm[row.arm_id].append(value)
        by_pair[row.pair_id][row.arm_id] = value

    cell_means = tuple(
        CellMean(
            arm_id=arm_id,
            endpoint=endpoint,
            budget_axis=estimand.budget_axis,
            n=len(values),
            mean=(sum(values) / len(values)) if values else None,
        )
        for arm_id, values in sorted(values_by_arm.items())
    )

    treatment_ids = tuple(estimand.treatment_arm_ids)
    control_ids = tuple(estimand.control_arm_ids)
    paired: list[PairedDifference] = []
    for pair_id, arms in sorted(by_pair.items()):
        for treatment_id in treatment_ids:
            for control_id in control_ids:
                if treatment_id not in arms or control_id not in arms:
                    continue
                treatment_value = arms[treatment_id]
                control_value = arms[control_id]
                paired.append(
                    PairedDifference(
                        pair_id=pair_id,
                        treatment_arm_id=treatment_id,
                        control_arm_id=control_id,
                        treatment_value=treatment_value,
                        control_value=control_value,
                        difference=treatment_value - control_value,
                    )
                )
    mean_difference = (
        sum(item.difference for item in paired) / len(paired) if paired else None
    )
    return DescriptiveContrast(
        estimand_id=estimand.estimand_id,
        endpoint=endpoint,
        budget_axis=estimand.budget_axis,
        cell_means=cell_means,
        paired_differences=tuple(paired),
        mean_difference=mean_difference,
        complete_pairs=len({item.pair_id for item in paired}),
    )


def _endpoint_value(row: RunSummary, endpoint: str) -> float | None:
    archive_field = _ENDPOINT_GETTERS.get(endpoint)
    if archive_field is not None:
        value = getattr(row.final_archive, archive_field)
        if value is None:
            return None
        return float(value)
    counters: Mapping[str, float | int | None] = {
        "proposal_slots": row.final_counters.proposal_slots,
        "evaluator_completions": row.final_counters.evaluator_completions,
        "total_tokens": row.final_counters.total_tokens,
        "monetary_cost": row.final_counters.monetary_cost,
        "wall_seconds": row.final_counters.wall_seconds,
    }
    if endpoint not in counters:
        raise AttributionAdmissionError(
            [
                AdmissionIssue(
                    code="analysis.unknown_endpoint",
                    message=f"unsupported endpoint {endpoint!r}",
                )
            ]
        )
    value = counters[endpoint]
    return None if value is None else float(value)

"""Analysis admission and descriptive estimand readouts."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from worldspace.attribution.design import (
    ArmAnatomy,
    CellMean,
    DescriptiveContrast,
    DesignMatrix,
    InteractionDifference,
    PairedDifference,
)
from worldspace.attribution.manifest import (
    BudgetAxis,
    EstimandSpec,
    StudyManifest,
    study_manifest_hash,
)
from worldspace.attribution.records import BudgetCheckpoint, ProposalEvent, RunSummary
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
_BUDGET_COUNTER_FIELDS: dict[BudgetAxis, str] = {
    "proposal": "proposal_slots",
    "valid_proposal": "valid_proposals",
    "evaluation": "evaluator_completions",
    "llm_call_attempted": "llm_attempts",
    "llm_call_completed": "llm_completions",
    "prompt_token": "prompt_tokens",
    "completion_token": "completion_tokens",
    "token": "total_tokens",
    "evaluator_wall_time": "evaluator_seconds",
    "llm_latency": "llm_latency_seconds",
    "wall_time": "wall_seconds",
    "monetary": "monetary_cost",
}
_INTERACTION_FORMULA = re.compile(
    r"^\(\s*([A-Za-z0-9_.-]+)\s*-\s*([A-Za-z0-9_.-]+)\s*\)\s*-\s*"
    r"\(\s*([A-Za-z0-9_.-]+)\s*-\s*([A-Za-z0-9_.-]+)\s*\)$"
)


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
    checkpoints_by_run: Mapping[str, Sequence[BudgetCheckpoint]] | None = None,
    events_by_run: Mapping[str, Sequence[ProposalEvent]] | None = None,
) -> DescriptiveContrast:
    """Compute absolute cell means and paired differences for fixtures.

    ``anytime_auc`` and ``interaction`` fail closed when traces, completeness,
    or the interaction formula cannot support the declared form. Holm / t-CIs
    are not computed here.
    """
    rows = tuple(summaries)
    values_by_run: dict[str, float] = {}
    for row in rows:
        if estimand.form == "anytime_auc":
            values_by_run[row.run_id] = _anytime_auc_for_row(
                row,
                estimand=estimand,
                checkpoints_by_run=checkpoints_by_run,
            )
        else:
            value = _endpoint_value(row, estimand.endpoint)
            if value is not None:
                values_by_run[row.run_id] = value

    values_by_arm: dict[str, list[float]] = defaultdict(list)
    by_pair: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = values_by_run.get(row.run_id)
        if value is None:
            continue
        values_by_arm[row.arm_id].append(value)
        by_pair[row.pair_id][row.arm_id] = value

    cell_means = tuple(
        CellMean(
            arm_id=arm_id,
            endpoint=estimand.endpoint,
            budget_axis=estimand.budget_axis,
            n=len(values),
            mean=(sum(values) / len(values)) if values else None,
        )
        for arm_id, values in sorted(values_by_arm.items())
    )
    anatomy = _arm_anatomy(rows, events_by_run=events_by_run)

    if estimand.form == "interaction":
        interaction_diffs = _interaction_differences(by_pair, estimand=estimand)
        mean_difference = (
            sum(item.difference for item in interaction_diffs) / len(interaction_diffs)
            if interaction_diffs
            else None
        )
        return DescriptiveContrast(
            estimand_id=estimand.estimand_id,
            endpoint=estimand.endpoint,
            budget_axis=estimand.budget_axis,
            form=estimand.form,
            cell_means=cell_means,
            paired_differences=(),
            interaction_differences=interaction_diffs,
            anatomy=anatomy,
            mean_difference=mean_difference,
            complete_pairs=len({item.pair_id for item in interaction_diffs}),
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
        endpoint=estimand.endpoint,
        budget_axis=estimand.budget_axis,
        form=estimand.form,
        cell_means=cell_means,
        paired_differences=tuple(paired),
        interaction_differences=(),
        anatomy=anatomy,
        mean_difference=mean_difference,
        complete_pairs=len({item.pair_id for item in paired}),
    )


def parse_interaction_formula(
    formula: str,
) -> tuple[str, str, str, str]:
    """Parse ``(A - B) - (C - D)`` into four arm ids."""
    match = _INTERACTION_FORMULA.match(formula.strip())
    if match is None:
        raise AttributionAdmissionError(
            [
                AdmissionIssue(
                    code="analysis.interaction_formula",
                    message=(
                        "interaction_formula must be '(A - B) - (C - D)' "
                        f"with arm ids, got {formula!r}"
                    ),
                )
            ]
        )
    return match.group(1), match.group(2), match.group(3), match.group(4)


def _anytime_auc_for_row(
    row: RunSummary,
    *,
    estimand: EstimandSpec,
    checkpoints_by_run: Mapping[str, Sequence[BudgetCheckpoint]] | None,
) -> float:
    if row.event_completeness == "summary_only":
        raise AttributionAdmissionError(
            [
                AdmissionIssue(
                    code="analysis.anytime_summary_only",
                    message=(
                        f"run {row.run_id!r} is summary_only; "
                        "anytime_auc requires a trace or prospective ledger"
                    ),
                )
            ]
        )
    if not checkpoints_by_run or row.run_id not in checkpoints_by_run:
        raise AttributionAdmissionError(
            [
                AdmissionIssue(
                    code="analysis.anytime_missing_trace",
                    message=f"run {row.run_id!r} has no checkpoints for anytime_auc",
                )
            ]
        )
    return _trapezoid_auc(
        checkpoints_by_run[row.run_id],
        run_id=row.run_id,
        budget_axis=estimand.budget_axis,
        endpoint=estimand.endpoint,
    )


def _trapezoid_auc(
    checkpoints: Sequence[BudgetCheckpoint],
    *,
    run_id: str,
    budget_axis: BudgetAxis,
    endpoint: str,
) -> float:
    counter_field = _BUDGET_COUNTER_FIELDS[budget_axis]
    points_by_x: dict[float, float] = {}
    for checkpoint in checkpoints:
        if checkpoint.indexed_by != budget_axis:
            continue
        completeness = checkpoint.source_completeness.get(budget_axis)
        if completeness == "unavailable":
            raise AttributionAdmissionError(
                [
                    AdmissionIssue(
                        code="analysis.anytime_axis_unavailable",
                        message=(
                            f"run {run_id!r} checkpoint {checkpoint.checkpoint_index} "
                            f"marks {budget_axis} unavailable"
                        ),
                    )
                ]
            )
        x_raw = getattr(checkpoint.counters, counter_field)
        y_raw = _checkpoint_endpoint(checkpoint, endpoint)
        if x_raw is None or y_raw is None:
            raise AttributionAdmissionError(
                [
                    AdmissionIssue(
                        code="analysis.anytime_incomplete_point",
                        message=(
                            f"run {run_id!r} checkpoint {checkpoint.checkpoint_index} "
                            "is missing index or endpoint value"
                        ),
                    )
                ]
            )
        points_by_x[float(x_raw)] = float(y_raw)
    points = sorted(points_by_x.items())
    if len(points) < 2:
        raise AttributionAdmissionError(
            [
                AdmissionIssue(
                    code="analysis.anytime_insufficient_trace",
                    message=(
                        f"run {run_id!r} needs at least two {budget_axis} "
                        "checkpoints for anytime_auc"
                    ),
                )
            ]
        )
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area


def _checkpoint_endpoint(checkpoint: BudgetCheckpoint, endpoint: str) -> float | None:
    archive_field = _ENDPOINT_GETTERS.get(endpoint)
    if archive_field is not None:
        value = getattr(checkpoint.archive, archive_field)
        return None if value is None else float(value)
    if endpoint == "occupied_cells":
        return float(checkpoint.archive.occupied_cells)
    return None


def _interaction_differences(
    by_pair: Mapping[str, Mapping[str, float]],
    *,
    estimand: EstimandSpec,
) -> tuple[InteractionDifference, ...]:
    formula = estimand.interaction_formula
    if not formula:
        raise AttributionAdmissionError(
            [
                AdmissionIssue(
                    code="analysis.interaction_formula",
                    message="interaction estimand requires interaction_formula",
                )
            ]
        )
    arm_a, arm_b, arm_c, arm_d = parse_interaction_formula(formula)
    declared = set(estimand.treatment_arm_ids) | set(estimand.control_arm_ids)
    missing = {arm_a, arm_b, arm_c, arm_d} - declared
    if missing:
        raise AttributionAdmissionError(
            [
                AdmissionIssue(
                    code="analysis.interaction_undeclared_arm",
                    message=(
                        f"interaction_formula names undeclared arms {sorted(missing)}"
                    ),
                )
            ]
        )
    diffs: list[InteractionDifference] = []
    for pair_id, arms in sorted(by_pair.items()):
        needed = (arm_a, arm_b, arm_c, arm_d)
        if any(arm_id not in arms for arm_id in needed):
            continue
        minuend = arms[arm_a] - arms[arm_b]
        subtrahend = arms[arm_c] - arms[arm_d]
        diffs.append(
            InteractionDifference(
                pair_id=pair_id,
                formula=formula,
                minuend_treatment_arm_id=arm_a,
                minuend_control_arm_id=arm_b,
                subtrahend_treatment_arm_id=arm_c,
                subtrahend_control_arm_id=arm_d,
                minuend=minuend,
                subtrahend=subtrahend,
                difference=minuend - subtrahend,
            )
        )
    if not diffs:
        raise AttributionAdmissionError(
            [
                AdmissionIssue(
                    code="analysis.interaction_incomplete_cells",
                    message=(
                        "no complete 2x2 pair has all four interaction arms "
                        f"{arm_a}, {arm_b}, {arm_c}, {arm_d}"
                    ),
                )
            ]
        )
    return tuple(diffs)


def _arm_anatomy(
    rows: Sequence[RunSummary],
    *,
    events_by_run: Mapping[str, Sequence[ProposalEvent]] | None,
) -> tuple[ArmAnatomy, ...]:
    events_by_arm: dict[str, list[ProposalEvent]] = defaultdict(list)
    summaries_by_arm: dict[str, list[RunSummary]] = defaultdict(list)
    for row in rows:
        summaries_by_arm[row.arm_id].append(row)
        if events_by_run and row.run_id in events_by_run:
            events_by_arm[row.arm_id].extend(events_by_run[row.run_id])
    anatomy: list[ArmAnatomy] = []
    for arm_id in sorted(summaries_by_arm):
        events = events_by_arm.get(arm_id, [])
        if events:
            valid = sum(
                1
                for event in events
                if event.generation.structurally_valid is True
                or event.generation.parse_valid is True
            )
            fallbacks = sum(1 for event in events if event.generation.fallback)
            anatomy.append(
                ArmAnatomy(
                    arm_id=arm_id,
                    n_events=len(events),
                    valid_proposal_rate=valid / len(events),
                    fallback_rate=fallbacks / len(events),
                    fill_empty_count=sum(
                        1
                        for event in events
                        if event.evaluation.insertion == "fill_empty"
                    ),
                    improve_count=sum(
                        1 for event in events if event.evaluation.insertion == "improve"
                    ),
                    occupied_not_better_count=sum(
                        1
                        for event in events
                        if event.evaluation.insertion == "occupied_not_better"
                    ),
                )
            )
            continue
        slots = [row.final_counters.proposal_slots for row in summaries_by_arm[arm_id]]
        valids = [
            row.final_counters.valid_proposals for row in summaries_by_arm[arm_id]
        ]
        if all(slot is not None for slot in slots) and all(
            valid is not None for valid in valids
        ):
            total_slots = sum(int(slot) for slot in slots if slot is not None)
            total_valid = sum(int(valid) for valid in valids if valid is not None)
            rate = (total_valid / total_slots) if total_slots else None
        else:
            rate = None
        anatomy.append(
            ArmAnatomy(
                arm_id=arm_id,
                n_events=None,
                valid_proposal_rate=rate,
                fallback_rate=None,
                fill_empty_count=None,
                improve_count=None,
                occupied_not_better_count=None,
            )
        )
    return tuple(anatomy)


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

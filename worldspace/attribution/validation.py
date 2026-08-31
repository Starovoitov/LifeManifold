"""Cross-record admission checks for attribution studies and analyses."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from worldspace.attribution.manifest import AdapterCapabilities, StudyManifest
from worldspace.attribution.records import RunSummary


@dataclass(frozen=True)
class AdmissionIssue:
    """One stable machine-readable validation failure."""

    code: str
    message: str


class AttributionAdmissionError(ValueError):
    """Raised when a study or analysis cohort fails closed."""

    def __init__(self, issues: Iterable[AdmissionIssue]) -> None:
        self.issues = tuple(issues)
        super().__init__(
            "; ".join(f"{item.code}: {item.message}" for item in self.issues)
        )


def validate_study_capabilities(
    study: StudyManifest,
    capabilities: AdapterCapabilities,
) -> tuple[AdmissionIssue, ...]:
    """Return unsupported domain-adapter requirements without substitution."""
    issues: list[AdmissionIssue] = []
    for field in ("domain_id", "adapter_id", "adapter_version"):
        actual = getattr(study, field)
        supported = getattr(capabilities, field)
        if actual != supported:
            issues.append(
                AdmissionIssue(
                    code=f"capability.{field}",
                    message=f"study requires {actual!r}, adapter declares {supported!r}",
                )
            )

    supported_axes = set(capabilities.budget_axes)
    for arm in study.arms:
        requirements = (
            (
                "initialization",
                arm.treatment.initialization.kind,
                capabilities.initialization_kinds,
            ),
            ("selector", arm.treatment.selector.kind, capabilities.selectors),
            ("generator", arm.treatment.generator.kind, capabilities.generators),
            (
                "prompt_channel",
                arm.treatment.prompt_channel.kind,
                capabilities.prompt_channels,
            ),
            (
                "repair_fallback",
                arm.treatment.repair_fallback.kind,
                capabilities.repair_fallback_kinds,
            ),
            ("gate", arm.treatment.gate.kind, capabilities.gate_modes),
            (
                "allocation",
                arm.treatment.allocation.kind,
                capabilities.allocation_kinds,
            ),
            (
                "replacement",
                arm.treatment.replacement.kind,
                capabilities.replacement_kinds,
            ),
        )
        for axis, required, supported in requirements:
            if required not in supported:
                issues.append(
                    AdmissionIssue(
                        code=f"capability.{axis}",
                        message=(
                            f"arm {arm.arm_id!r} requires {required!r}; "
                            f"adapter supports {sorted(supported)!r}"
                        ),
                    )
                )
        archive_type = arm.treatment.replacement.parameters.get("archive_type")
        if not isinstance(archive_type, str):
            issues.append(
                AdmissionIssue(
                    code="capability.archive_type",
                    message=(
                        f"arm {arm.arm_id!r} replacement must declare string "
                        "parameters.archive_type"
                    ),
                )
            )
        elif archive_type not in capabilities.archive_types:
            issues.append(
                AdmissionIssue(
                    code="capability.archive_type",
                    message=(
                        f"arm {arm.arm_id!r} requires archive type "
                        f"{archive_type!r}; adapter supports "
                        f"{sorted(capabilities.archive_types)!r}"
                    ),
                )
            )
        missing_axes = set(arm.treatment.budget.indexing_axes) - supported_axes
        if missing_axes:
            issues.append(
                AdmissionIssue(
                    code="capability.budget_axes",
                    message=(
                        f"arm {arm.arm_id!r} requires unsupported budget axes "
                        f"{sorted(missing_axes)!r}"
                    ),
                )
            )
        if (
            arm.treatment.initialization.kind != "empty"
            and not capabilities.supports_warm_start
        ):
            issues.append(
                AdmissionIssue(
                    code="capability.warm_start",
                    message=(
                        f"arm {arm.arm_id!r} requires non-empty initialization "
                        "but adapter declares no warm-start support"
                    ),
                )
            )
    return tuple(issues)


def require_study_capabilities(
    study: StudyManifest,
    capabilities: AdapterCapabilities,
) -> None:
    """Raise when any study treatment is unsupported by its adapter."""
    issues = validate_study_capabilities(study, capabilities)
    if issues:
        raise AttributionAdmissionError(issues)


def admit_analysis_cohort(
    summaries: Iterable[RunSummary],
    *,
    expected_arm_ids: Iterable[str],
    minimum_complete_pairs: int,
    require_full_events: bool = False,
) -> tuple[RunSummary, ...]:
    """Fail closed on mixed identities, incomplete pairs, or duplicate cells."""
    rows = tuple(summaries)
    expected_arms = frozenset(expected_arm_ids)
    issues: list[AdmissionIssue] = []
    if not rows:
        raise AttributionAdmissionError(
            [AdmissionIssue("cohort.empty", "analysis cohort is empty")]
        )
    if not expected_arms:
        raise AttributionAdmissionError(
            [AdmissionIssue("cohort.arms", "expected_arm_ids must not be empty")]
        )
    if minimum_complete_pairs < 1:
        raise ValueError("minimum_complete_pairs must be at least one")

    for field in (
        "schema_version",
        "study_id",
        "evidence_tier",
        "domain_id",
        "domain_version",
        "adapter_id",
        "adapter_version",
        "evaluator_hash",
        "protocol_hash",
        "study_manifest_hash",
    ):
        values = {getattr(row, field) for row in rows}
        if len(values) != 1:
            issues.append(
                AdmissionIssue(
                    code=f"cohort.mixed_{field}",
                    message=f"analysis rows contain multiple {field} values",
                )
            )

    treatment_by_arm: dict[str, set[str]] = defaultdict(set)
    rows_by_pair: dict[str, dict[str, RunSummary]] = defaultdict(dict)
    for row in rows:
        if row.arm_id not in expected_arms:
            issues.append(
                AdmissionIssue(
                    code="cohort.unexpected_arm",
                    message=f"run {row.run_id!r} has unexpected arm {row.arm_id!r}",
                )
            )
        treatment_by_arm[row.arm_id].add(row.treatment_hash)
        pair = rows_by_pair[row.pair_id]
        if row.arm_id in pair:
            issues.append(
                AdmissionIssue(
                    code="cohort.duplicate_pair_arm",
                    message=f"pair {row.pair_id!r} repeats arm {row.arm_id!r}",
                )
            )
        else:
            pair[row.arm_id] = row
        if require_full_events and row.event_completeness != "full":
            issues.append(
                AdmissionIssue(
                    code="cohort.incomplete_events",
                    message=f"run {row.run_id!r} lacks full proposal events",
                )
            )
        if not row.completed:
            issues.append(
                AdmissionIssue(
                    code="cohort.incomplete_run",
                    message=f"run {row.run_id!r} is incomplete",
                )
            )

    for arm_id, hashes in treatment_by_arm.items():
        if len(hashes) != 1:
            issues.append(
                AdmissionIssue(
                    code="cohort.treatment_drift",
                    message=f"arm {arm_id!r} has multiple treatment hashes",
                )
            )

    complete_pairs = 0
    for pair_id, pair in rows_by_pair.items():
        observed = frozenset(pair)
        if observed == expected_arms:
            complete_pairs += 1
            for field in ("seed", "domain_instance_id", "initial_archive_hash"):
                values = {getattr(row, field) for row in pair.values()}
                if len(values) != 1:
                    issues.append(
                        AdmissionIssue(
                            code=f"cohort.unpaired_{field}",
                            message=(
                                f"pair {pair_id!r} contains multiple " f"{field} values"
                            ),
                        )
                    )
            continue
        issues.append(
            AdmissionIssue(
                code="cohort.incomplete_pair",
                message=(
                    f"pair {pair_id!r} has arms {sorted(observed)!r}, "
                    f"expected {sorted(expected_arms)!r}"
                ),
            )
        )
    if complete_pairs < minimum_complete_pairs:
        issues.append(
            AdmissionIssue(
                code="cohort.minimum_pairs",
                message=(
                    f"found {complete_pairs} complete pairs, "
                    f"require {minimum_complete_pairs}"
                ),
            )
        )
    if issues:
        raise AttributionAdmissionError(issues)
    return rows

"""Admission guards for confirmatory analysis cohorts."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from worldspace.attribution.validation import AdmissionIssue, AttributionAdmissionError


def refuse_contaminated_artifact_paths(
    paths: Iterable[Path | str],
    *,
    refuse_q1_nightly: bool = True,
) -> None:
    """Fail closed when feasibility design data or Q1 summaries are offered."""
    issues: list[AdmissionIssue] = []
    for raw in paths:
        text = str(raw).replace("\\", "/")
        lower = text.lower()
        if refuse_q1_nightly and (
            "nightly_run_summary.json" in lower
            and (
                "/artifacts/nightly/" in lower
                or "/q1/" in lower
                or "scheduler" in lower
            )
        ):
            issues.append(
                AdmissionIssue(
                    code="confirmatory.refuse_q1_nightly",
                    message=f"refusing Q1 nightly summary path: {text}",
                )
            )
            continue
        if "feasibility" in lower and (
            text.endswith(".jsonl") or text.endswith(".json")
        ):
            issues.append(
                AdmissionIssue(
                    code="confirmatory.refuse_feasibility_jsonl",
                    message=f"refusing feasibility artifact: {text}",
                )
            )
            continue
        if "/artifacts/controlled_attribution/" in lower and (
            "/nas201/" in lower or "/pcg/" in lower
        ):
            # Feasibility smoke/isolated reports are not confirmatory inputs.
            if any(
                token in lower
                for token in (
                    "smoke",
                    "isolated",
                    "repair_pair",
                    "feasibility",
                )
            ):
                issues.append(
                    AdmissionIssue(
                        code="confirmatory.refuse_feasibility_design_data",
                        message=f"refusing feasibility design-data path: {text}",
                    )
                )
    if issues:
        raise AttributionAdmissionError(issues)


def refuse_mixed_evidence_tier(tiers: Iterable[str]) -> None:
    values = {str(tier) for tier in tiers}
    if len(values) != 1:
        raise AttributionAdmissionError(
            [
                AdmissionIssue(
                    code="confirmatory.mixed_evidence_tier",
                    message=(
                        "analysis cohort mixes evidence_tier values: "
                        f"{sorted(values)}"
                    ),
                )
            ]
        )
    if "confirmatory" not in values and values:
        # Allow a single non-confirmatory tier for non-confirmatory analyses.
        # Confirmatory callers must pass confirmatory explicitly.
        pass

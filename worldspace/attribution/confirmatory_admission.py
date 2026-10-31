"""Admission guards for confirmatory analysis cohorts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

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


def refuse_missing_llm_usage(
    summaries: Iterable[Mapping[str, Any]],
    *,
    missing_usage_policy: str,
) -> None:
    """Reject LLM arms that spent a call and did not report tokens.

    ``reject`` does not apply to random or genetic arms, and it does not apply
    to an LLM arm that never left the slot (no completed call and no HTTP).
    """
    if missing_usage_policy != "reject":
        return
    issues: list[AdmissionIssue] = []
    for index, summary in enumerate(summaries):
        generator = summary.get("generator")
        llm_arm = generator == "llm" or summary.get("llm_enabled") is True
        if not llm_arm:
            continue
        completed = int(summary.get("llm_calls_completed") or 0)
        http_calls = int(summary.get("llm_http_calls") or 0)
        if completed == 0 and http_calls == 0:
            continue
        if summary.get("total_tokens") is None:
            issues.append(
                AdmissionIssue(
                    code="confirmatory.missing_llm_usage",
                    message=(
                        f"summary {index} is an LLM arm with empty token usage"
                    ),
                )
            )
    if issues:
        raise AttributionAdmissionError(issues)


def refuse_response_model_mismatch(
    summaries: Iterable[Mapping[str, Any]],
) -> None:
    """Reject an LLM arm whose observed response model is not the pinned snapshot."""
    issues: list[AdmissionIssue] = []
    for index, summary in enumerate(summaries):
        expected = summary.get("expected_response_model")
        actual = summary.get("response_model")
        flagged = summary.get("response_model_mismatch") is True
        differs = (
            isinstance(expected, str)
            and bool(expected)
            and isinstance(actual, str)
            and bool(actual)
            and actual != expected
        )
        if not flagged and not differs:
            continue
        issues.append(
            AdmissionIssue(
                code="confirmatory.response_model_mismatch",
                message=(
                    f"summary {index} response_model {actual!r} "
                    f"does not match expected {expected!r}"
                ),
            )
        )
    if issues:
        raise AttributionAdmissionError(issues)

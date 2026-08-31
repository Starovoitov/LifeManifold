#!/usr/bin/env python3
"""Validate and render the controlled-attribution literature registries.

Registries under ``artifacts/controlled_attribution/`` are local design notes,
not confirmatory evidence, and are not part of the committed tree.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

DEFAULT_ROOT = Path("artifacts/controlled_attribution")
REGISTRIES = {
    "search_runs": "search_runs.jsonl",
    "screening": "screening.jsonl",
    "papers": "papers.jsonl",
    "comparisons": "comparisons.jsonl",
    "internal_claims": "internal_claims.jsonl",
    "adjudication": "adjudication.jsonl",
}
COMPONENTS = (
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
)
COMPONENT_STATUSES = {"matched", "changed", "unclear", "not_applicable"}
BUDGET_AXES = {
    "proposal",
    "valid_proposal",
    "evaluation",
    "llm_call",
    "token",
    "wall_time",
    "monetary",
}
SEED_TITLES = {
    "openelm": ("evolution through large models",),
    "llmatic": ("llmatic",),
    "in_context_qd": (
        "large language models as in context ai generators for quality diversity",
    ),
    "codeevolve": ("codeevolve",),
    "loongflow": ("loongflow",),
    "adaptevolve": ("adaptevolve",),
    "dei": ("diversity in evolutionary inference",),
}


class RegistryError(ValueError):
    """Raised for an invalid literature registry."""


def _normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RegistryError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise RegistryError(f"{path}:{line_number}: row must be an object")
        row["_source_line"] = line_number
        rows.append(row)
    return rows


def load_registries(root: Path) -> dict[str, list[dict[str, Any]]]:
    return {name: _load_jsonl(root / filename) for name, filename in REGISTRIES.items()}


def _require(
    row: dict[str, Any],
    fields: Iterable[str],
    registry: str,
    errors: list[str],
) -> None:
    line = row.get("_source_line", "?")
    for field in fields:
        if field not in row:
            errors.append(f"{registry}:{line}: missing {field}")


def _check_unique(
    rows: list[dict[str, Any]],
    key: str,
    registry: str,
    errors: list[str],
) -> set[str]:
    seen: set[str] = set()
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            errors.append(f"{registry}:{row.get('_source_line', '?')}: invalid {key}")
        elif value in seen:
            errors.append(f"{registry}: duplicate {key}={value}")
        else:
            seen.add(value)
    return seen


def _validate_treatment_vector(comparison: dict[str, Any], errors: list[str]) -> None:
    comparison_id = comparison.get("comparison_id", "?")
    vector = comparison.get("treatment_vector")
    if not isinstance(vector, dict):
        errors.append(f"comparison {comparison_id}: treatment_vector must be an object")
        return
    missing = set(COMPONENTS) - set(vector)
    extra = set(vector) - set(COMPONENTS)
    if missing:
        errors.append(
            f"comparison {comparison_id}: missing components {sorted(missing)}"
        )
    if extra:
        errors.append(f"comparison {comparison_id}: unknown components {sorted(extra)}")
    for component in COMPONENTS:
        assessment = vector.get(component)
        if not isinstance(assessment, dict):
            continue
        if set(assessment) != {"status", "focal", "baseline", "evidence"}:
            errors.append(
                f"comparison {comparison_id}: {component} requires "
                "status/focal/baseline/evidence"
            )
            continue
        if assessment["status"] not in COMPONENT_STATUSES:
            errors.append(
                f"comparison {comparison_id}: invalid {component} status "
                f"{assessment['status']!r}"
            )
        if not isinstance(assessment["evidence"], str):
            errors.append(
                f"comparison {comparison_id}: {component}.evidence must be a string"
            )


def validate_registries(
    registries: dict[str, list[dict[str, Any]]],
) -> list[str]:
    errors: list[str] = []
    search_runs = registries["search_runs"]
    screening = registries["screening"]
    papers = registries["papers"]
    comparisons = registries["comparisons"]
    internal_claims = registries["internal_claims"]

    required = {
        "search_runs": (
            "record_type",
            "search_id",
            "executed_at",
            "source",
            "query_family",
            "exact_query",
            "filters",
            "result_count",
            "formal",
            "export_path",
        ),
        "screening": (
            "record_type",
            "screening_id",
            "title",
            "discovery_source",
            "stage",
            "decision",
            "reason_code",
            "audited",
        ),
        "papers": (
            "record_type",
            "paper_id",
            "title",
            "authors",
            "year",
            "version_status",
            "stratum",
            "identifiers",
            "urls",
            "domain",
            "task",
            "code_available",
            "screening_id",
        ),
        "comparisons": (
            "record_type",
            "comparison_id",
            "paper_id",
            "claim",
            "focal_arm",
            "baseline_arm",
            "endpoint",
            "budget_axes",
            "sample",
            "reported_result",
            "treatment_vector",
            "identified_effects",
            "unidentified_effects",
            "source",
        ),
        "internal_claims": (
            "record_type",
            "claim_id",
            "claim",
            "domain",
            "bundled_contrast",
            "controlled_contrast",
            "matched_components",
            "confounded_components",
            "endpoint",
            "evidence_role",
            "prospective_status",
            "artifact_paths",
        ),
    }
    expected_types = {
        "search_runs": "search_run",
        "screening": "screening",
        "papers": "paper",
        "comparisons": "comparison",
        "internal_claims": "internal_claim",
    }
    for registry, fields in required.items():
        for row in registries[registry]:
            _require(row, fields, registry, errors)
            if row.get("record_type") != expected_types[registry]:
                errors.append(
                    f"{registry}:{row.get('_source_line', '?')}: "
                    f"record_type must be {expected_types[registry]!r}"
                )

    _check_unique(search_runs, "search_id", "search_runs", errors)
    screening_ids = _check_unique(screening, "screening_id", "screening", errors)
    paper_ids = _check_unique(papers, "paper_id", "papers", errors)
    _check_unique(comparisons, "comparison_id", "comparisons", errors)
    _check_unique(internal_claims, "claim_id", "internal_claims", errors)

    for row in search_runs:
        count = row.get("result_count")
        if not isinstance(count, int) or count < 0:
            errors.append(
                f"search_runs:{row.get('_source_line', '?')}: invalid result_count"
            )
        if not isinstance(row.get("formal"), bool):
            errors.append(
                f"search_runs:{row.get('_source_line', '?')}: formal must be boolean"
            )

    valid_stages = {"title_abstract", "full_text"}
    valid_decisions = {"include", "exclude", "unclear"}
    for row in screening:
        if row.get("stage") not in valid_stages:
            errors.append(f"screening:{row.get('_source_line', '?')}: invalid stage")
        if row.get("decision") not in valid_decisions:
            errors.append(f"screening:{row.get('_source_line', '?')}: invalid decision")
        if row.get("decision") == "exclude" and not row.get("reason_code"):
            errors.append(
                f"screening:{row.get('_source_line', '?')}: exclusion needs reason_code"
            )

    for paper in papers:
        if paper.get("screening_id") not in screening_ids:
            errors.append(
                f"paper {paper.get('paper_id', '?')}: unknown screening_id "
                f"{paper.get('screening_id')!r}"
            )
        if paper.get("stratum") not in {"core", "adjacent"}:
            errors.append(f"paper {paper.get('paper_id', '?')}: invalid stratum")
        if not isinstance(paper.get("urls"), list) or not paper.get("urls"):
            errors.append(f"paper {paper.get('paper_id', '?')}: urls must be non-empty")

    for comparison in comparisons:
        comparison_id = comparison.get("comparison_id", "?")
        if comparison.get("paper_id") not in paper_ids:
            errors.append(
                f"comparison {comparison_id}: unknown paper_id "
                f"{comparison.get('paper_id')!r}"
            )
        _validate_treatment_vector(comparison, errors)
        budget_axes = comparison.get("budget_axes")
        if not isinstance(budget_axes, dict):
            errors.append(f"comparison {comparison_id}: invalid budget_axes")
        else:
            reported = set(budget_axes.get("reported", []))
            omitted = set(budget_axes.get("omitted", []))
            if not reported <= BUDGET_AXES or not omitted <= BUDGET_AXES:
                errors.append(f"comparison {comparison_id}: unknown budget axis")
            if reported & omitted:
                errors.append(
                    f"comparison {comparison_id}: budget axis both reported and omitted"
                )
        source = comparison.get("source")
        if (
            not isinstance(source, dict)
            or not source.get("url")
            or not source.get("section")
        ):
            errors.append(f"comparison {comparison_id}: source needs url and section")

    for claim in internal_claims:
        if claim.get("prospective_status") != "historical_or_design_only":
            errors.append(
                f"internal claim {claim.get('claim_id', '?')}: invalid prospective_status"
            )
        paths = claim.get("artifact_paths")
        if not isinstance(paths, list) or not paths:
            errors.append(
                f"internal claim {claim.get('claim_id', '?')}: artifact_paths required"
            )

    if papers:
        missing_seeds = [
            seed for seed, recovered in seed_recall(papers).items() if not recovered
        ]
        if missing_seeds:
            errors.append(f"seed-recall gate failed: {missing_seeds}")
    if papers and not any(row.get("formal") for row in search_runs):
        errors.append("no formal search run logged")
    full_text_inclusions = [
        row
        for row in screening
        if row.get("stage") == "full_text" and row.get("decision") == "include"
    ]
    if any(not row.get("audited") for row in full_text_inclusions):
        errors.append("all included full texts must be audited")
    auditable_exclusions = [
        row
        for row in screening
        if row.get("stage") == "title_abstract"
        and row.get("decision") == "exclude"
        and row.get("reason_code") != "duplicate"
    ]
    if auditable_exclusions:
        audited = sum(bool(row.get("audited")) for row in auditable_exclusions)
        if audited / len(auditable_exclusions) < 0.10:
            errors.append(
                "title/abstract exclusion audit is below 10%: "
                f"{audited}/{len(auditable_exclusions)}"
            )

    return errors


def seed_recall(papers: list[dict[str, Any]]) -> dict[str, bool]:
    corpus = " | ".join(_normalize_title(str(row.get("title", ""))) for row in papers)
    return {
        seed: any(_normalize_title(alias) in corpus for alias in aliases)
        for seed, aliases in SEED_TITLES.items()
    }


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend(
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in rows
    )
    return lines


def render_readout(
    registries: dict[str, list[dict[str, Any]]], errors: list[str]
) -> str:
    searches = registries["search_runs"]
    screenings = registries["screening"]
    papers = registries["papers"]
    comparisons = registries["comparisons"]
    internal = registries["internal_claims"]

    formal_results = sum(
        int(row["result_count"]) for row in searches if row.get("formal")
    )
    title_rows = [row for row in screenings if row.get("stage") == "title_abstract"]
    duplicate_rows = [
        row for row in title_rows if row.get("reason_code") == "duplicate"
    ]
    unique_title_rows = [
        row for row in title_rows if row.get("reason_code") != "duplicate"
    ]
    full_text_rows = [row for row in screenings if row.get("stage") == "full_text"]
    exclusion_reasons = Counter(
        str(row.get("reason_code"))
        for row in screenings
        if row.get("decision") == "exclude"
    )
    strata = Counter(str(row.get("stratum")) for row in papers)
    recall = seed_recall(papers)

    component_counts = {
        component: Counter(
            comparison["treatment_vector"][component]["status"]
            for comparison in comparisons
            if isinstance(comparison.get("treatment_vector"), dict)
            and isinstance(comparison["treatment_vector"].get(component), dict)
        )
        for component in COMPONENTS
    }
    budget_counts = Counter(
        axis
        for comparison in comparisons
        for axis in comparison.get("budget_axes", {}).get("reported", [])
    )

    lines = [
        "# Literature and claim inventory",
        "",
        "**Cutoff:** 2026-08-31  ",
        "**Protocol:** `SCOPING_PROTOCOL.md`  ",
        "**Schema:** `schema.json`  ",
        "**Evidence role:** design evidence; not confirmatory evidence",
        "",
        "## Readout status",
        "",
        f"- Registry validation: {'PASS' if not errors else 'FAIL'}",
        f"- Formal search runs: {sum(bool(row.get('formal')) for row in searches)}",
        f"- Raw records returned across formal queries: {formal_results}",
        f"- Unique title/abstract records charted: {len(unique_title_rows)}",
        f"- Duplicate source/version records removed: {len(duplicate_rows)}",
        f"- Full texts assessed: {len(full_text_rows)}",
        f"- Included papers: {len(papers)} ({strata['core']} core; "
        f"{strata['adjacent']} adjacent)",
        f"- Extracted focal-vs-baseline contrasts: {len(comparisons)}",
        f"- Separate internal historical/design claims: {len(internal)}",
        "",
        "Raw query counts overlap and must not be interpreted as unique records.",
        "The inventory charts one principal attribution-relevant contrast per "
        "included report; it is not an exhaustive transcription of every result table.",
        "",
        "## Main inventory finding",
        "",
        f"- {sum(component_counts[c]['changed'] for c in ('selector', 'allocation'))} "
        "selector/allocation changes are coded across the 58 principal contrasts "
        "(components may co-occur). Bundled orchestration changes are common enough "
        "that a generic `LLM vs baseline` label is not an adequate treatment.",
        f"- Initialization is unclear in {component_counts['initialization']['unclear']}/"
        f"{len(comparisons)} contrasts and repair/fallback in "
        f"{component_counts['repair_fallback']['unclear']}/{len(comparisons)}.",
        f"- Budget matching is explicit in only {component_counts['budget']['matched']}/"
        f"{len(comparisons)} contrasts. Token and monetary axes are reported in "
        f"{budget_counts['token']} and {budget_counts['monetary']} principal contrasts.",
        f"- Exact numeric principal effects were charted for "
        f"{sum(row.get('reported_result', {}).get('direction') != 'unclear' for row in comparisons)}/"
        f"{len(comparisons)} contrasts; absent values remain `unclear`, not inferred.",
        "",
        "## Seed-recall gate",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["seed", "recovered"],
            [[seed, "yes" if passed else "no"] for seed, passed in recall.items()],
        )
    )
    lines.extend(["", "## Screening flow", ""])
    lines.extend(
        _markdown_table(
            ["stage", "count"],
            [
                ["raw formal-query returns", formal_results],
                ["duplicates removed", len(duplicate_rows)],
                ["unique title/abstract records screened", len(unique_title_rows)],
                ["full texts assessed", len(full_text_rows)],
                ["included papers", len(papers)],
                [
                    "excluded screening decisions",
                    sum(row.get("decision") == "exclude" for row in screenings),
                ],
            ],
        )
    )
    if exclusion_reasons:
        lines.extend(["", "Exclusion reasons:", ""])
        lines.extend(
            f"- `{reason}`: {count}"
            for reason, count in sorted(exclusion_reasons.items())
        )

    lines.extend(["", "## Component attribution matrix", ""])
    lines.extend(
        _markdown_table(
            ["component", "matched", "changed", "unclear", "N/A"],
            [
                [
                    component,
                    component_counts[component]["matched"],
                    component_counts[component]["changed"],
                    component_counts[component]["unclear"],
                    component_counts[component]["not_applicable"],
                ]
                for component in COMPONENTS
            ],
        )
    )
    lines.extend(["", "## Reported budget axes", ""])
    lines.extend(
        _markdown_table(
            ["axis", "contrasts reporting axis"],
            [[axis, budget_counts[axis]] for axis in sorted(BUDGET_AXES)],
        )
    )

    lines.extend(["", "## External papers and extracted contrasts", ""])
    comparisons_by_paper: dict[str, list[dict[str, Any]]] = {}
    for comparison in comparisons:
        comparisons_by_paper.setdefault(str(comparison.get("paper_id")), []).append(
            comparison
        )
    for paper in sorted(papers, key=lambda row: (row.get("year", 0), row["title"])):
        lines.extend(
            [
                f"### {paper['title']} ({paper['year']})",
                "",
                f"- Stratum: `{paper['stratum']}`",
                f"- Domain/task: {paper['domain']} / {paper['task']}",
                f"- Source: {paper['urls'][0]}",
            ]
        )
        for comparison in comparisons_by_paper.get(paper["paper_id"], []):
            vector = comparison["treatment_vector"]
            changed = [
                component
                for component in COMPONENTS
                if vector[component]["status"] == "changed"
            ]
            unclear = [
                component
                for component in COMPONENTS
                if vector[component]["status"] == "unclear"
            ]
            lines.extend(
                [
                    f"- `{comparison['comparison_id']}`: "
                    f"{comparison['focal_arm']} vs {comparison['baseline_arm']} — "
                    f"{comparison['claim']}",
                    f"  - Endpoint: {comparison['endpoint']}",
                    "  - Changed: " + (", ".join(changed) if changed else "none coded"),
                    "  - Unclear: " + (", ".join(unclear) if unclear else "none"),
                    f"  - Evidence: {comparison['source']['section']}",
                ]
            )
        lines.append("")

    lines.extend(["## Internal historical/design evidence", ""])
    for claim in internal:
        lines.extend(
            [
                f"- `{claim['claim_id']}` ({claim['domain']}, "
                f"`{claim['evidence_role']}`): {claim['claim']}",
                f"  - Controlled contrast: {claim['controlled_contrast']}",
                f"  - Status: `{claim['prospective_status']}`",
            ]
        )

    all_included_audited = all(
        row.get("audited") for row in full_text_rows if row.get("decision") == "include"
    )
    seed_gate = bool(recall) and all(recall.values())
    complete_vectors = all(
        isinstance(row.get("treatment_vector"), dict)
        and set(row["treatment_vector"]) == set(COMPONENTS)
        for row in comparisons
    )
    gate_rows = [
        ["registries validate", not errors],
        ["all protocol seeds recovered", seed_gate],
        ["included full texts audited", all_included_audited],
        ["all contrasts have complete vectors", complete_vectors],
        ["internal claims are separate", bool(internal)],
        [
            "at least one formal search logged",
            any(row.get("formal") for row in searches),
        ],
    ]
    lines.extend(["", "## Next-stage gate", ""])
    lines.extend(f"- [{'x' if passed else ' '}] {label}" for label, passed in gate_rows)
    lines.extend(
        [
            "",
            "Minimum next-stage manifest fields derived from the inventory:",
            "",
            "`domain`, `task`, `tier`, `initialization`, `selector`, `generator`, "
            "`prompt_channel`, `repair_fallback`, `gate`, `replacement`, "
            "`allocation`, `budget` (all axes), `representation`, `model`, "
            "`evaluator`, `seed`, `initial_archive_hash`, `prompt_hash`, "
            "`treatment_vector_hash`, and artifact provenance.",
            "",
            "## Interpretation boundary",
            "",
            "The frequencies above describe only this declared retrieved corpus. "
            "They are not estimates of all LLM+QD research. Heterogeneous endpoints "
            "and treatment bundles are not pooled into a meta-analytic effect.",
            "",
        ]
    )
    return "\n".join(lines)


def _strip_source_lines(registries: dict[str, list[dict[str, Any]]]) -> None:
    for rows in registries.values():
        for row in rows:
            row.pop("_source_line", None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    registries = load_registries(args.root)
    errors = validate_registries(registries)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if not args.validate_only:
        output = args.output or args.root / "CLAIM_INVENTORY.md"
        output.write_text(render_readout(registries, errors), encoding="utf-8")
        print(f"Wrote {output}")
    _strip_source_lines(registries)
    print(
        "Validated "
        + ", ".join(f"{name}={len(rows)}" for name, rows in registries.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

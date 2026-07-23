#!/usr/bin/env python3
"""Write the gated descriptive analysis for a five-arm maze pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

CONDITIONS = (
    "genetic",
    "genetic_filter",
    "llm_stub",
    "llm_hints",
    "llm_hints_filter",
)


def _trace_at(path: Path, metric: str, budget: int) -> float:
    by_eval: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get(metric) is not None:
            by_eval[int(row["evaluations"])] = float(row[metric])
    xs = np.asarray(sorted(by_eval), dtype=float)
    ys = np.asarray([by_eval[int(item)] for item in xs], dtype=float)
    return float(np.interp(float(budget), xs, ys))


def _trace_auc(path: Path, metric: str, budget: int) -> float:
    by_eval: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get(metric) is not None:
            by_eval[int(row["evaluations"])] = float(row[metric])
    xs = np.asarray(sorted(by_eval), dtype=float)
    ys = np.asarray([by_eval[int(item)] for item in xs], dtype=float)
    grid = np.arange(0, budget + 1, 50, dtype=float)
    if grid[-1] != budget:
        grid = np.append(grid, float(budget))
    return float(np.trapezoid(np.interp(grid, xs, ys), grid) / budget)


def _evaluations_to_coverage(path: Path, threshold: float) -> int | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if float(row.get("coverage", 0.0)) >= threshold:
            return int(row["evaluations"])
    return None


def _llm_audit(row: dict[str, Any]) -> dict[str, Any]:
    audit = row.get("llm_audit")
    return audit if isinstance(audit, dict) else {}


def _gate_lines(summaries: dict[str, dict[str, Any]]) -> list[str]:
    filter_skips = [
        100.0 * float(summaries[condition].get("skip_rate", 0.0))
        for condition in ("genetic_filter", "llm_hints_filter")
    ]
    filter_pass = all(25.0 <= rate <= 45.0 for rate in filter_skips)
    llm_conditions = ("llm_stub", "llm_hints", "llm_hints_filter")
    fallbacks = [
        100.0
        * float(
            _llm_audit(summaries[condition]).get(
                "fallback_rate",
                summaries[condition].get("llm_fallback_rate", 0.0),
            )
            or 0.0
        )
        for condition in llm_conditions
    ]
    fallback_pass = all(rate <= 5.0 for rate in fallbacks)
    parse_rates = [
        100.0
        * float(
            _llm_audit(summaries[condition]).get(
                "parse_success_rate",
                summaries[condition].get("llm_parse_success_rate", 0.0),
            )
            or 0.0
        )
        for condition in llm_conditions
    ]
    parse_pass = all(rate >= 95.0 for rate in parse_rates)
    genetic_cov = 100.0 * float(summaries["genetic"]["coverage"])
    llm_best_cov = max(
        100.0 * float(summaries[condition]["coverage"]) for condition in llm_conditions
    )
    genetic_dominates = genetic_cov >= llm_best_cov
    lines = [
        "## Quality gates (seed 0, descriptive)",
        "",
        "| Gate | Target | Result |",
        "|------|--------|--------|",
        f"| Filter skip | 25–45% | **{'PASS' if filter_pass else 'FAIL'}** — "
        f"genetic_filter **{filter_skips[0]:.1f}%**, "
        f"llm_hints_filter **{filter_skips[1]:.1f}%** |",
        f"| LLM fallback | ≤5% | **{'PASS' if fallback_pass else 'FAIL'}** — "
        + ", ".join(
            f"`{condition}` {rate:.1f}%"
            for condition, rate in zip(llm_conditions, fallbacks, strict=True)
        )
        + " |",
        f"| LLM parse | ≥95% | **{'PASS' if parse_pass else 'FAIL'}** — "
        + ", ".join(
            f"`{condition}` {rate:.1f}%"
            for condition, rate in zip(llm_conditions, parse_rates, strict=True)
        )
        + " |",
        "",
        f"**Genetic ME vs best LLM arm (coverage):** genetic **{genetic_cov:.2f}%** vs "
        f"best LLM **{llm_best_cov:.2f}%** "
        f"({'genetic dominates' if genetic_dominates else 'LLM competitive'}).",
        "",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/experiments/q1-v5-maze-pilot"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    summaries: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        path = root / condition / "seed_0/nightly_run_summary.json"
        if not path.is_file():
            raise SystemExit(f"missing completed pilot: {path}")
        summaries[condition] = json.loads(path.read_text(encoding="utf-8"))
    common_budget = min(int(row["evaluations"]) for row in summaries.values())
    genetic = summaries["genetic"]
    genetic_filter = summaries["genetic_filter"]
    llm_hints = summaries["llm_hints"]
    filter_delta = 100.0 * (
        float(genetic_filter["coverage"]) - float(genetic["coverage"])
    )
    matched_filter_delta = 100.0 * (
        _trace_at(
            root / "genetic_filter/seed_0/archive_trace.jsonl",
            "coverage",
            common_budget,
        )
        - _trace_at(
            root / "genetic/seed_0/archive_trace.jsonl",
            "coverage",
            common_budget,
        )
    )
    lines = [
        "# B5 Maze pipeline — seed-0 live pilot @ 5k",
        "",
        "**Tier:** `q1-v5-maze-pilot`",
        "**Protocol:** [`EXPERIMENT_PROTOCOL_Q1_v5.md`](../../EXPERIMENT_PROTOCOL_Q1_v5.md)",
        "**Status:** five arms @ 5,000 proposals, seed 0; **exploratory** (not F-B5 confirmatory).",
        "",
        *_gate_lines(summaries),
        "## Final fixed-proposal levels",
        "",
        "| Arm | Evaluations | Skip % | Coverage % | Mean fitness | QD-score | Wall min | LLM calls | Fallback % |",
        "|-----|------------:|-------:|-----------:|-------------:|---------:|---------:|----------:|-----------:|",
    ]
    for condition in CONDITIONS:
        row = summaries[condition]
        audit = _llm_audit(row)
        calls = int(audit.get("attempts", row.get("llm_calls", 0)) or 0)
        fallback = float(
            audit.get("fallback_rate", row.get("llm_fallback_rate", 0.0)) or 0.0
        )
        lines.append(
            f"| `{condition}` | {int(row['evaluations']):,} | "
            f"{100.0 * float(row.get('skip_rate', 0.0)):.1f} | "
            f"{100.0 * float(row['coverage']):.2f} | "
            f"{float(row['mean_best_fitness']):.4f} | "
            f"{float(row['qd_score']):.1f} | "
            f"{float(row['elapsed_seconds']) / 60.0:.1f} | "
            f"{calls:,} | {100.0 * fallback:.1f} |"
        )
    lines.extend(
        [
            "",
            f"## Matched real-evaluation checkpoint ({common_budget:,})",
            "",
            "| Arm | Coverage % | Mean fitness | QD-score | Coverage AUC | QD-score AUC |",
            "|-----|-----------:|-------------:|---------:|-------------:|-------------:|",
        ]
    )
    for condition in CONDITIONS:
        trace = root / condition / "seed_0/archive_trace.jsonl"
        lines.append(
            f"| `{condition}` | "
            f"{100.0 * _trace_at(trace, 'coverage', common_budget):.2f} | "
            f"{_trace_at(trace, 'mean_best_fitness', common_budget):.4f} | "
            f"{_trace_at(trace, 'qd_score', common_budget):.1f} | "
            f"{_trace_auc(trace, 'coverage', common_budget):.4f} | "
            f"{_trace_auc(trace, 'qd_score', common_budget):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Paired contrasts (seed 0, descriptive)",
            "",
            "| Contrast | Δ coverage (pp) | Notes |",
            "|----------|----------------:|-------|",
            f"| genetic_filter − genetic (terminal) | {filter_delta:+.2f} | fixed 5k proposals |",
            f"| genetic_filter − genetic (@ {common_budget:,} evals) | {matched_filter_delta:+.2f} | matched-eval sample efficiency |",
            f"| llm_hints − llm_stub | "
            f"{100.0 * (float(llm_hints['coverage']) - float(summaries['llm_stub']['coverage'])):+.2f} | hint content |",
            f"| genetic − llm_hints | "
            f"{100.0 * (float(genetic['coverage']) - float(llm_hints['coverage'])):+.2f} | ME vs bundled LLM |",
            "",
            "## Evaluations to coverage thresholds (descriptive)",
            "",
            "| Arm | 25% | 40% | 50% |",
            "|-----|----:|----:|----:|",
        ]
    )
    for condition in CONDITIONS:
        trace = root / condition / "seed_0/archive_trace.jsonl"
        values = [
            _evaluations_to_coverage(trace, threshold)
            for threshold in (0.25, 0.40, 0.50)
        ]
        rendered = [
            f"{value:,}" if value is not None else "not reached" for value in values
        ]
        lines.append(f"| `{condition}` | {' | '.join(rendered)} |")
    lines.extend(
        [
            "",
            "## Gate verdict",
            "",
            "This is a **single-seed exploratory pilot** for stack parity (protocol v5 gate 7). "
            "It does **not** enter Holm family F-B5-maze (requires ≥5 matched seeds). "
            "Magnitudes are maze-specific; compare directionally to B4 dungeon PARTIAL pattern "
            "(genetic ME dominance + filter AUC at matched evals).",
            "",
            "Artifacts: `summary.csv`, per-run traces/archives under each arm directory.",
            "",
        ]
    )
    output = args.output or (root / "ANALYSIS.md")
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

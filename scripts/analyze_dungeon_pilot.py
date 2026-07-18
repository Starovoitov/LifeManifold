#!/usr/bin/env python3
"""Write the gated descriptive analysis for a five-arm dungeon pilot."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/experiments/q1-v4-dungeon-pilot"),
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
    lines = [
        "# B4 Dungeon factorial — seed-0 feasibility pilot",
        "",
        "**Status:** five arms complete at 5,000 proposals; descriptive gate only.",
        "",
        "## Final fixed-proposal levels",
        "",
        "| Arm | Evaluations | Skip % | Coverage % | Mean fitness | QD-score | Wall min | LLM calls | Fallback % |",
        "|-----|------------:|-------:|-----------:|-------------:|---------:|---------:|----------:|-----------:|",
    ]
    for condition in CONDITIONS:
        row = summaries[condition]
        audit = row.get("llm_audit")
        audit = audit if isinstance(audit, dict) else {}
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
            "## Evaluations to coverage thresholds (descriptive)",
            "",
            "These thresholds were frozen only for a resumed matrix, not before this pilot.",
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
    projected_calls = 3 * 5 * 19_500
    lines.extend(
        [
            "",
            "## Gate verdict",
            "",
            "Two live quality gates fail: filter arms skip only 19.8–20.3% "
            "(target 25–45%), and LLM fallback rises to 28.1–34.0% "
            "(valid parse 66.0–71.9%, below the 95% gate).",
            "",
            f"The planned seeds 0–4 full stage projects **{projected_calls:,} paid "
            "LLM calls**, above the frozen 100,000-call gate. The matrix therefore "
            "**STOPS after the feasibility pilot**. No v4 Holm verdict is computed "
            "from one seed. Engineering and artifact-contract validation pass; "
            "confirmatory cross-domain claims remain open.",
            "",
            "Artifacts: `anytime_coverage.png`, `anytime_qd_score.png`, "
            "`summary.csv`, and all per-run traces/archives.",
            "",
        ]
    )
    output = args.output or (root / "ANALYSIS.md")
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

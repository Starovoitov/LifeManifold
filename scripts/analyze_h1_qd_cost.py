#!/usr/bin/env python3
"""Matched H1 companion metrics: QD-score, best fitness, wall, API calls.

Primary estimand remains mean terminal coverage TOST (analyze_h1_matched_providers /
Table tab:decomposition). This script fills the QD/cost board reviewers expect and
flags that occupied-bin mean fitness can mislead when coverage differs.

AUC / coverage–quality curves need archive_trace on both arms; frozen q1-full/hints
lacks traces (mixed-stack 2×2 re-run will close that gap).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SU_ROOT = ROOT / "artifacts/experiments/q1-stub-uniform-sensitivity/stub_uniform"
HI_ROOT = ROOT / "artifacts/experiments/q1-full/hints"
SU_CSV = ROOT / "artifacts/experiments/q1-stub-uniform-sensitivity/summary.csv"
HI_CSV = ROOT / "artifacts/experiments/q1-full/summary.csv"
OUT_DIR = ROOT / "artifacts/experiments/h1-qd-cost"


def _rows(csv_path: Path, condition: str) -> list[dict[str, str]]:
    out = []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("condition") == condition and int(row["seed"]) <= 9:
                out.append(row)
    return sorted(out, key=lambda r: int(r["seed"]))


def _cov_pct(row: dict[str, str]) -> float:
    v = float(row["coverage_pct"])
    return v * 100.0 if v <= 1.5 else v


def _max_archive_fitness(archive: Path) -> float:
    best = float("-inf")
    with archive.open() as fh:
        for line in fh:
            fit = float(json.loads(line).get("fitness", 0.0))
            if fit > best:
                best = fit
    return best


def _mean_sd(xs: list[float]) -> dict[str, float]:
    arr = np.asarray(xs, dtype=float)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def analyze(seeds: list[int]) -> dict[str, Any]:
    su_rows = {int(r["seed"]): r for r in _rows(SU_CSV, "stub_uniform")}
    hi_rows = {int(r["seed"]): r for r in _rows(HI_CSV, "hints")}
    per: list[dict[str, Any]] = []
    for seed in seeds:
        su, hi = su_rows[seed], hi_rows[seed]
        su_sum = json.loads(
            (SU_ROOT / f"seed_{seed}" / "nightly_run_summary.json").read_text()
        )
        hi_sum = json.loads(
            (HI_ROOT / f"seed_{seed}" / "nightly_run_summary.json").read_text()
        )
        su_best = _max_archive_fitness(
            SU_ROOT / f"seed_{seed}" / "map_elites_archive.jsonl"
        )
        hi_best = _max_archive_fitness(
            HI_ROOT / f"seed_{seed}" / "map_elites_archive.jsonl"
        )
        per.append(
            {
                "seed": seed,
                "stub_uniform": {
                    "coverage_pct": _cov_pct(su),
                    "mean_best_fitness": float(su["mean_best_fitness"]),
                    "best_fitness": su_best,
                    "qd_score": float(su["qd_score"]),
                    "wall_min": float(su["elapsed_seconds"]) / 60.0,
                    "llm_calls": int(su_sum.get("llm_emit_attempts") or 0),
                    "has_archive_trace": (
                        SU_ROOT / f"seed_{seed}" / "archive_trace.jsonl"
                    ).is_file(),
                },
                "hints": {
                    "coverage_pct": _cov_pct(hi),
                    "mean_best_fitness": float(hi["mean_best_fitness"]),
                    "best_fitness": hi_best,
                    "qd_score": float(hi["qd_score"]),
                    "wall_min": float(hi["elapsed_seconds"]) / 60.0,
                    "llm_calls": int(hi_sum.get("llm_emit_attempts") or 0),
                    "has_archive_trace": (
                        HI_ROOT / f"seed_{seed}" / "archive_trace.jsonl"
                    ).is_file(),
                },
            }
        )

    def collect(arm: str, key: str) -> list[float]:
        return [float(r[arm][key]) for r in per]

    def paired(key: str) -> dict[str, float]:
        d = [float(r["hints"][key]) - float(r["stub_uniform"][key]) for r in per]
        return _mean_sd(d)

    summary = {
        "n": len(per),
        "seeds": seeds,
        "stub_uniform": {
            k: _mean_sd(collect("stub_uniform", k))
            for k in (
                "coverage_pct",
                "mean_best_fitness",
                "best_fitness",
                "qd_score",
                "wall_min",
                "llm_calls",
            )
        },
        "hints": {
            k: _mean_sd(collect("hints", k))
            for k in (
                "coverage_pct",
                "mean_best_fitness",
                "best_fitness",
                "qd_score",
                "wall_min",
                "llm_calls",
            )
        },
        "delta_hints_minus_stub_uniform": {
            k: paired(k)
            for k in (
                "coverage_pct",
                "mean_best_fitness",
                "best_fitness",
                "qd_score",
                "wall_min",
                "llm_calls",
            )
        },
        "archive_trace_seeds": {
            "stub_uniform": sum(
                1 for r in per if r["stub_uniform"]["has_archive_trace"]
            ),
            "hints": sum(1 for r in per if r["hints"]["has_archive_trace"]),
        },
        "limits": {
            "mean_best_fitness": (
                "Occupied-bin average; can rise when coverage falls. Prefer QD-score."
            ),
            "auc": (
                "Frozen q1-full/hints lacks archive_trace.jsonl; matched AUC "
                "coverage/QD and coverage–quality curves deferred to mixed-2x2 re-run."
            ),
            "tokens": (
                "Per-token usage not logged; API $ estimated from llm_emit_attempts "
                "× published qwen-turbo rates (same 6500 calls both arms)."
            ),
        },
        "per_seed": per,
    }
    return summary


def write_md(summary: dict[str, Any], path: Path) -> None:
    su, hi, d = (
        summary["stub_uniform"],
        summary["hints"],
        summary["delta_hints_minus_stub_uniform"],
    )
    lines = [
        "# Matched H1 QD / cost companions (descriptive)",
        "",
        "Primary claim remains exploratory mean terminal-coverage TOST "
        "(Table `tab:decomposition`).",
        "",
        "| Metric | stub_uniform | hints | Δ (hints−su) |",
        "|--------|-------------:|------:|-------------:|",
        (
            f"| Coverage (%) | {su['coverage_pct']['mean']:.2f}±{su['coverage_pct']['sd']:.2f} | "
            f"{hi['coverage_pct']['mean']:.2f}±{hi['coverage_pct']['sd']:.2f} | "
            f"{d['coverage_pct']['mean']:+.2f}±{d['coverage_pct']['sd']:.2f} |"
        ),
        (
            f"| QD-score | {su['qd_score']['mean']:.1f}±{su['qd_score']['sd']:.1f} | "
            f"{hi['qd_score']['mean']:.1f}±{hi['qd_score']['sd']:.1f} | "
            f"{d['qd_score']['mean']:+.1f}±{d['qd_score']['sd']:.1f} |"
        ),
        (
            f"| Mean fitness (occupied bins) | "
            f"{su['mean_best_fitness']['mean']:.4f}±{su['mean_best_fitness']['sd']:.4f} | "
            f"{hi['mean_best_fitness']['mean']:.4f}±{hi['mean_best_fitness']['sd']:.4f} | "
            f"{d['mean_best_fitness']['mean']:+.4f}±{d['mean_best_fitness']['sd']:.4f} |"
        ),
        (
            f"| Best fitness (archive max) | "
            f"{su['best_fitness']['mean']:.4f}±{su['best_fitness']['sd']:.4f} | "
            f"{hi['best_fitness']['mean']:.4f}±{hi['best_fitness']['sd']:.4f} | "
            f"{d['best_fitness']['mean']:+.4f}±{d['best_fitness']['sd']:.4f} |"
        ),
        (
            f"| Wall (min) | {su['wall_min']['mean']:.1f}±{su['wall_min']['sd']:.1f} | "
            f"{hi['wall_min']['mean']:.1f}±{hi['wall_min']['sd']:.1f} | "
            f"{d['wall_min']['mean']:+.1f}±{d['wall_min']['sd']:.1f} |"
        ),
        (
            f"| LLM API calls | {su['llm_calls']['mean']:.0f} | "
            f"{hi['llm_calls']['mean']:.0f} | "
            f"{d['llm_calls']['mean']:+.0f} |"
        ),
        "",
        f"archive_trace seeds: stub_uniform={summary['archive_trace_seeds']['stub_uniform']}, "
        f"hints={summary['archive_trace_seeds']['hints']}.",
        "",
        f"- {summary['limits']['mean_best_fitness']}",
        f"- {summary['limits']['auc']}",
        f"- {summary['limits']['tokens']}",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    summary = analyze(list(range(10)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "h1_qd_cost.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_md(summary, OUT_DIR / "ANALYSIS.md")
    print((OUT_DIR / "ANALYSIS.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

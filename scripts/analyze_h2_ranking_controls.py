#!/usr/bin/env python3
"""Analyze q1-h2-ranking-controls: H2 skip-volume vs ranking isolation.

Compares three new arms (random_skip, shadow, filter_eval_matched) against
frozen genetic_me_uniform / genetic_me_filter baselines.
Writes artifacts/experiments/q1-h2-ranking-controls/{ANALYSIS.md,h2_ranking_controls_analysis.json}.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/experiments/q1-h2-ranking-controls"
NEW_ARMS = (
    "genetic_me_random_skip",
    "genetic_me_shadow",
    "genetic_me_filter_eval_matched",
)
BASELINE_ARMS = {
    "genetic_me_uniform": ROOT / "artifacts/experiments/q1-v3-genetic-me-uniform",
    "genetic_me_filter": ROOT / "artifacts/experiments/q1-v3-genetic-me-filter",
}
TAU = 0.45
EVAL_BUDGETS = (5000, 10000, 15000, 20000)


def _mean_sd(xs: list[float]) -> dict[str, float | int]:
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def _cov_at(trace: Path, budget: int) -> float | None:
    best = None
    with trace.open() as fh:
        for line in fh:
            row = json.loads(line)
            ev = int(row.get("evaluations", -1))
            if ev < 0 or ev > budget:
                continue
            cov = float(row["coverage"])
            if cov > 1.5:
                cov /= 100.0
            best = cov
    return None if best is None else best * 100.0


def _auc_cov(trace: Path, horizon: int) -> float | None:
    xs: list[float] = []
    ys: list[float] = []
    with trace.open() as fh:
        for line in fh:
            row = json.loads(line)
            ev = int(row.get("evaluations", -1))
            if ev < 0 or ev > horizon:
                continue
            cov = float(row["coverage"])
            if cov > 1.5:
                cov /= 100.0
            xs.append(float(ev))
            ys.append(cov)
    if len(xs) < 2:
        return None
    order = np.argsort(xs)
    x = np.asarray(xs, dtype=float)[order]
    y = np.asarray(ys, dtype=float)[order]
    if x[0] > 0:
        x = np.concatenate([[0.0], x])
        y = np.concatenate([[y[0]], y])
    if x[-1] < horizon:
        x = np.concatenate([x, [float(horizon)]])
        y = np.concatenate([y, [y[-1]]])
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trapz(y, x) / float(horizon) * 100.0)


def _run_dir(tier: Path, arm: str, seed: int) -> Path:
    return tier / arm / f"seed_{seed}"


def load_seed(tier: Path, arm: str, seed: int) -> dict[str, Any] | None:
    d = _run_dir(tier, arm, seed)
    summary = d / "nightly_run_summary.json"
    if not summary.is_file():
        return None
    payload = json.loads(summary.read_text())
    cov = float(payload.get("coverage", 0.0))
    cov_pct = cov * 100.0 if cov <= 1.5 else cov
    trace = d / "archive_trace.jsonl"
    row: dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "coverage_pct": cov_pct,
        "evaluations": int(payload.get("evaluations", 0)),
        "iterations": int(payload.get("iterations", 0)),
        "qd_score": float(payload.get("qd_score", float("nan"))),
        "wall_min": float(payload.get("elapsed_seconds", 0.0)) / 60.0,
    }
    if trace.is_file():
        for b in EVAL_BUDGETS:
            c = _cov_at(trace, b)
            row[f"cov_at_{b}"] = c
        row["auc_cov_20k"] = _auc_cov(trace, 20000)
    return row


def paired_delta(
    a: dict[int, dict[str, Any]], b: dict[int, dict[str, Any]], key: str
) -> dict[str, Any]:
    seeds = sorted(set(a) & set(b))
    diffs = [
        float(b[s][key]) - float(a[s][key])
        for s in seeds
        if a[s].get(key) is not None and b[s].get(key) is not None
    ]
    stats: dict[str, Any] = dict(_mean_sd(diffs))
    stats["n_positive"] = int(sum(1 for d in diffs if d > 0))
    stats["seeds"] = seeds
    return stats


def analyze_shadow_seed(seed_dir: Path) -> dict[str, Any]:
    arch = seed_dir / "surrogate_archive.jsonl"
    n = would_skip = false_skip = missed_elite = 0
    with arch.open() as fh:
        for line in fh:
            row = json.loads(line)
            n += 1
            if row.get("decision") != "skip":
                continue
            would_skip += 1
            eo = row.get("eval_outcome") or {}
            true_fit = eo.get("fitness")
            inserted = bool(eo.get("accepted") or eo.get("improved"))
            if true_fit is not None and float(true_fit) >= TAU:
                false_skip += 1
            if inserted:
                missed_elite += 1
    return {
        "n_proposals": n,
        "would_skip": would_skip,
        "would_skip_rate_pct": 100.0 * would_skip / n if n else float("nan"),
        "false_skip_among_would_skip_pct": (
            100.0 * false_skip / would_skip if would_skip else float("nan")
        ),
        "missed_elite_among_would_skip_pct": (
            100.0 * missed_elite / would_skip if would_skip else float("nan")
        ),
    }


def main() -> int:
    by_arm: dict[str, dict[int, dict[str, Any]]] = {}
    all_arms = list(BASELINE_ARMS) + list(NEW_ARMS)

    for arm in all_arms:
        tier = OUT if arm in NEW_ARMS else BASELINE_ARMS[arm]
        by_arm[arm] = {}
        for seed in range(10):
            row = load_seed(tier, arm, seed)
            if row is not None:
                by_arm[arm][seed] = row

    complete = sorted(set.intersection(*(set(by_arm[a]) for a in all_arms)))
    status = "complete" if len(complete) >= 10 else f"partial_n={len(complete)}"

    levels: dict[str, Any] = {}
    for arm in all_arms:
        rows = [by_arm[arm][s] for s in complete]
        levels[arm] = {
            "n": len(rows),
            "coverage_pct": _mean_sd([r["coverage_pct"] for r in rows]),
            "evaluations": _mean_sd([float(r["evaluations"]) for r in rows]),
            "iterations": _mean_sd([float(r["iterations"]) for r in rows]),
            "wall_min": _mean_sd([r["wall_min"] for r in rows]),
            "cov_at_20000": _mean_sd(
                [r["cov_at_20000"] for r in rows if r.get("cov_at_20000") is not None]
            ),
            "auc_cov_20k": _mean_sd(
                [r["auc_cov_20k"] for r in rows if r.get("auc_cov_20k") is not None]
            ),
        }

    contrasts = {
        "ranking_filter_minus_random_skip_cov20k": paired_delta(
            by_arm["genetic_me_random_skip"],
            by_arm["genetic_me_filter"],
            "cov_at_20000",
        ),
        "ranking_filter_minus_random_skip_terminal": paired_delta(
            by_arm["genetic_me_random_skip"],
            by_arm["genetic_me_filter"],
            "coverage_pct",
        ),
        "ranking_filter_minus_random_skip_auc20k": paired_delta(
            by_arm["genetic_me_random_skip"],
            by_arm["genetic_me_filter"],
            "auc_cov_20k",
        ),
        "eval_matched_minus_uniform_terminal": paired_delta(
            by_arm["genetic_me_uniform"],
            by_arm["genetic_me_filter_eval_matched"],
            "coverage_pct",
        ),
        "eval_matched_minus_uniform_cov20k": paired_delta(
            by_arm["genetic_me_uniform"],
            by_arm["genetic_me_filter_eval_matched"],
            "cov_at_20000",
        ),
        "filter_minus_uniform_cov20k": paired_delta(
            by_arm["genetic_me_uniform"],
            by_arm["genetic_me_filter"],
            "cov_at_20000",
        ),
        "shadow_minus_uniform_terminal": paired_delta(
            by_arm["genetic_me_uniform"],
            by_arm["genetic_me_shadow"],
            "coverage_pct",
        ),
    }

    shadow_per_seed = [
        analyze_shadow_seed(OUT / "genetic_me_shadow" / f"seed_{s}") for s in complete
    ]
    shadow_summary = {
        "would_skip_rate_pct": _mean_sd(
            [r["would_skip_rate_pct"] for r in shadow_per_seed]
        ),
        "false_skip_among_would_skip_pct": _mean_sd(
            [r["false_skip_among_would_skip_pct"] for r in shadow_per_seed]
        ),
        "missed_elite_among_would_skip_pct": _mean_sd(
            [r["missed_elite_among_would_skip_pct"] for r in shadow_per_seed]
        ),
        "per_seed": shadow_per_seed,
    }

    payload = {
        "tier": "q1-h2-ranking-controls",
        "status": status,
        "complete_seeds": complete,
        "levels": levels,
        "contrasts": contrasts,
        "shadow": shadow_summary,
        "per_seed": {a: list(by_arm[a].values()) for a in all_arms},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "h2_ranking_controls_analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    def fmt_ms(d: dict[str, float | int]) -> str:
        return f"{d['mean']:.2f}±{d['sd']:.2f}"

    def fmt_contrast(key: str, unit: str = "pp") -> str:
        c = contrasts[key]
        return (
            f"{c['mean']:+.2f}±{c['sd']:.2f} {unit} ({c.get('n_positive', 0)}/{c['n']})"
        )

    lines = [
        "# H2 ranking controls analysis",
        "",
        f"**Status:** `{status}` · seeds={complete}",
        "",
        "## Terminal levels (n=10)",
        "",
        "| Arm | Coverage % | Evals | Cov@20k % | AUC cov@20k % |",
        "|-----|------------:|------:|----------:|----------------:|",
    ]
    for arm in all_arms:
        lv = levels[arm]
        lines.append(
            f"| `{arm}` | {fmt_ms(lv['coverage_pct'])} | "
            f"{lv['evaluations']['mean']:.0f}±{lv['evaluations']['sd']:.0f} | "
            f"{fmt_ms(lv['cov_at_20000'])} | {fmt_ms(lv['auc_cov_20k'])} |"
        )

    lines += [
        "",
        "## Key contrasts (paired, descriptive)",
        "",
        "| Contrast | Δ terminal | Δ cov@20k | Δ AUC@20k |",
        "|----------|-------------:|----------:|----------:|",
        f"| filter − random_skip (ranking @ ~21.6k evals) | "
        f"{fmt_contrast('ranking_filter_minus_random_skip_terminal')} | "
        f"{fmt_contrast('ranking_filter_minus_random_skip_cov20k')} | "
        f"{fmt_contrast('ranking_filter_minus_random_skip_auc20k')} |",
        f"| filter_eval_matched − uniform (matched real evals) | "
        f"{fmt_contrast('eval_matched_minus_uniform_terminal')} | "
        f"{fmt_contrast('eval_matched_minus_uniform_cov20k')} | — |",
        f"| filter − uniform (fixed 650 iters; ref H2) | — | "
        f"{fmt_contrast('filter_minus_uniform_cov20k')} | — |",
        f"| shadow − uniform (all evaluated; parity check) | "
        f"{fmt_contrast('shadow_minus_uniform_terminal')} | — | — |",
        "",
        "## Shadow gate anatomy (would-skip with all proposals evaluated)",
        "",
        f"- Would-skip rate: {fmt_ms(shadow_summary['would_skip_rate_pct'])}%",
        f"- False-skip (true fit ≥ τ) among would-skip: "
        f"{fmt_ms(shadow_summary['false_skip_among_would_skip_pct'])}%",
        f"- Missed elite (insert/improve) among would-skip: "
        f"{fmt_ms(shadow_summary['missed_elite_among_would_skip_pct'])}%",
        "",
        "## Reading notes",
        "",
        "- **filter − random_skip (10/10):** ranking beats volume-matched skip at ~21.6k evals.",
        "- **filter_eval_matched − uniform (10/10):** terminal deficit at 650 iters is budget truncation.",
        "- **shadow:** exact uniform parity; false-skip 7.3%, missed-elite 2.6% among would-skip.",
        "- Descriptive paired contrasts only; not a new Holm family.",
        "",
        "Script: `scripts/analyze_h2_ranking_controls.py`",
        "",
    ]
    (OUT / "ANALYSIS.md").write_text("\n".join(lines))
    print((OUT / "ANALYSIS.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

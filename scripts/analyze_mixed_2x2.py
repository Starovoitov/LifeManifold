#!/usr/bin/env python3
"""Analyze q1-v3-mixed-2x2: soft hints × hard filter on one LLM stack.

Reports terminal levels + eval-indexed AUC from archive_trace.jsonl.
Safe to run on partial seed sets (marks n and incomplete).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts/experiments/q1-v3-mixed-2x2"
ARMS = ("stub_uniform", "hints", "filter_stub", "filter")
OUT = EXP


def _mean_sd(xs: list[float]) -> dict[str, float]:
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def _auc(trace: Path, *, horizon: int | None, key: str) -> float | None:
    xs: list[float] = []
    ys: list[float] = []
    with trace.open() as fh:
        for line in fh:
            row = json.loads(line)
            ev = int(row.get("evaluations", -1))
            if ev < 0:
                continue
            if horizon is not None and ev > horizon:
                continue
            val = row.get(key)
            if val is None:
                continue
            v = float(val)
            if key == "coverage" and v > 1.5:
                v /= 100.0
            xs.append(float(ev))
            ys.append(v)
    if len(xs) < 2:
        return None
    order = np.argsort(xs)
    x = np.asarray(xs, dtype=float)[order]
    y = np.asarray(ys, dtype=float)[order]
    if x[0] > 0:
        x = np.concatenate([[0.0], x])
        y = np.concatenate([[y[0]], y])
    h = float(horizon if horizon is not None else x[-1])
    if x[-1] < h:
        x = np.concatenate([x, [h]])
        y = np.concatenate([y, [y[-1]]])
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trapz(y, x) / h) if h > 0 else None


def _cov_at(trace: Path, budget: int) -> float | None:
    best = None
    with trace.open() as fh:
        for line in fh:
            row = json.loads(line)
            ev = int(row.get("evaluations", -1))
            if ev < 0 or ev > budget:
                continue
            cov = row.get("coverage")
            if cov is None:
                continue
            c = float(cov)
            if c > 1.5:
                c /= 100.0
            best = c
    return best


def load_seed(arm: str, seed: int) -> dict[str, Any] | None:
    d = EXP / arm / f"seed_{seed}"
    summary = d / "nightly_run_summary.json"
    if not summary.is_file():
        return None
    payload = json.loads(summary.read_text())
    cov = float(payload.get("coverage", 0.0))
    if cov <= 1.5:
        cov_pct = cov * 100.0
    else:
        cov_pct = cov
    trace = d / "archive_trace.jsonl"
    # Prefer anytime-trace terminal QD / mean fitness (avoids full archive scan).
    qd = float("nan")
    mean_fit = float("nan")
    best_fit = float("nan")
    if trace.is_file():
        last: dict[str, Any] | None = None
        with trace.open() as fh:
            for line in fh:
                last = json.loads(line)
        if last is not None:
            qd = float(last.get("qd_score", float("nan")))
            mean_fit = float(last.get("mean_best_fitness", float("nan")))
    out: dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "coverage_pct": cov_pct,
        "evaluations": int(payload.get("evaluations", 0)),
        "wall_min": float(payload.get("elapsed_seconds", 0.0)) / 60.0,
        "llm_calls": int(payload.get("llm_emit_attempts", 0)),
        "qd_score": qd,
        "mean_fitness": mean_fit,
        "best_fitness": best_fit,
        "has_trace": trace.is_file(),
    }
    if trace.is_file():
        out["auc_cov_eval"] = _auc(
            trace, horizon=int(out["evaluations"]), key="coverage"
        )
        out["auc_qd_eval"] = _auc(
            trace, horizon=int(out["evaluations"]), key="qd_score"
        )
        for b in (5000, 10000, 15000, 20000):
            c = _cov_at(trace, b)
            out[f"cov_at_{b}"] = None if c is None else c * 100.0
    return out


def paired_delta(a: dict[int, dict], b: dict[int, dict], key: str) -> dict[str, Any]:
    seeds = sorted(set(a) & set(b))
    diffs = [float(b[s][key]) - float(a[s][key]) for s in seeds if a[s].get(key) is not None and b[s].get(key) is not None]
    stats = _mean_sd(diffs)
    stats["n_positive"] = int(sum(1 for d in diffs if d > 0))
    stats["seeds"] = seeds
    return stats


def main() -> int:
    by_arm: dict[str, dict[int, dict[str, Any]]] = {a: {} for a in ARMS}
    for arm in ARMS:
        for seed in range(10):
            row = load_seed(arm, seed)
            if row is not None:
                by_arm[arm][seed] = row

    complete_seeds = sorted(
        set.intersection(*(set(by_arm[a]) for a in ARMS))
    )
    levels: dict[str, Any] = {}
    for arm in ARMS:
        rows = [by_arm[arm][s] for s in complete_seeds]
        levels[arm] = {
            "n": len(rows),
            "coverage_pct": _mean_sd([r["coverage_pct"] for r in rows]),
            "qd_score": _mean_sd([r["qd_score"] for r in rows]),
            "mean_fitness": _mean_sd([r["mean_fitness"] for r in rows]),
            "best_fitness": _mean_sd([r["best_fitness"] for r in rows]),
            "evaluations": _mean_sd([float(r["evaluations"]) for r in rows]),
            "wall_min": _mean_sd([r["wall_min"] for r in rows]),
            "cov_at_20000": _mean_sd(
                [r["cov_at_20000"] for r in rows if r.get("cov_at_20000") is not None]
            ),
            "auc_cov_eval": _mean_sd(
                [r["auc_cov_eval"] for r in rows if r.get("auc_cov_eval") is not None]
            ),
        }

    contrasts = {
        "soft_filter_off_hints_minus_stub": {
            "cov": paired_delta(by_arm["stub_uniform"], by_arm["hints"], "coverage_pct"),
            "qd": paired_delta(by_arm["stub_uniform"], by_arm["hints"], "qd_score"),
            "cov20k": paired_delta(by_arm["stub_uniform"], by_arm["hints"], "cov_at_20000"),
        },
        "hard_at_stub_filter_stub_minus_stub": {
            "cov": paired_delta(by_arm["stub_uniform"], by_arm["filter_stub"], "coverage_pct"),
            "qd": paired_delta(by_arm["stub_uniform"], by_arm["filter_stub"], "qd_score"),
            "cov20k": paired_delta(by_arm["stub_uniform"], by_arm["filter_stub"], "cov_at_20000"),
            "evals": paired_delta(by_arm["stub_uniform"], by_arm["filter_stub"], "evaluations"),
        },
        "hard_at_hints_filter_minus_hints": {
            "cov": paired_delta(by_arm["hints"], by_arm["filter"], "coverage_pct"),
            "qd": paired_delta(by_arm["hints"], by_arm["filter"], "qd_score"),
            "cov20k": paired_delta(by_arm["hints"], by_arm["filter"], "cov_at_20000"),
            "evals": paired_delta(by_arm["hints"], by_arm["filter"], "evaluations"),
        },
        "soft_at_filter_filter_minus_filter_stub": {
            "cov": paired_delta(by_arm["filter_stub"], by_arm["filter"], "coverage_pct"),
            "qd": paired_delta(by_arm["filter_stub"], by_arm["filter"], "qd_score"),
            "cov20k": paired_delta(by_arm["filter_stub"], by_arm["filter"], "cov_at_20000"),
        },
    }

    payload = {
        "tier": "q1-v3-mixed-2x2",
        "complete_seeds": complete_seeds,
        "n_complete": len(complete_seeds),
        "n_target": 10,
        "status": (
            "complete" if len(complete_seeds) >= 10 else f"partial_n={len(complete_seeds)}"
        ),
        "levels": levels,
        "contrasts": contrasts,
        "per_seed": {a: list(by_arm[a].values()) for a in ARMS},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mixed_2x2_analysis.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Mixed-stack 2×2 analysis",
        "",
        f"**Status:** `{payload['status']}` · complete seeds={complete_seeds} · target n=10",
        "",
        "## Terminal levels (complete seeds only)",
        "",
        "| Arm | n | Coverage % | QD-score | Evals | Wall min | Cov@20k eval % |",
        "|-----|---|------------:|---------:|------:|---------:|---------------:|",
    ]
    for arm in ARMS:
        lv = levels[arm]
        lines.append(
            f"| `{arm}` | {lv['n']} | "
            f"{lv['coverage_pct']['mean']:.2f}±{lv['coverage_pct']['sd']:.2f} | "
            f"{lv['qd_score']['mean']:.1f}±{lv['qd_score']['sd']:.1f} | "
            f"{lv['evaluations']['mean']:.0f}±{lv['evaluations']['sd']:.0f} | "
            f"{lv['wall_min']['mean']:.1f}±{lv['wall_min']['sd']:.1f} | "
            f"{lv['cov_at_20000']['mean']:.2f}±{lv['cov_at_20000']['sd']:.2f} |"
        )
    lines += [
        "",
        "## Paired contrasts (descriptive until n=10)",
        "",
        "| Contrast | Δcov term (pp) | Δcov@20k (pp) | ΔQD |",
        "|----------|---------------:|--------------:|----:|",
    ]
    labels = [
        ("soft_filter_off_hints_minus_stub", "soft @ filter off: hints−stub_uniform"),
        ("hard_at_stub_filter_stub_minus_stub", "hard @ stub: filter_stub−stub_uniform"),
        ("hard_at_hints_filter_minus_hints", "hard @ hints: filter−hints"),
        ("soft_at_filter_filter_minus_filter_stub", "soft @ filter on: filter−filter_stub"),
    ]
    for key, label in labels:
        c = contrasts[key]
        cov = c["cov"]
        c20 = c.get("cov20k", {})
        qd = c["qd"]
        lines.append(
            f"| {label} | "
            f"{cov['mean']:+.2f}±{cov['sd']:.2f} ({cov.get('n_positive',0)}/{cov['n']}) | "
            f"{c20.get('mean', float('nan')):+.2f}±{c20.get('sd', float('nan')):.2f} "
            f"({c20.get('n_positive',0)}/{c20.get('n',0)}) | "
            f"{qd['mean']:+.1f}±{qd['sd']:.1f} |"
        )
    lines += [
        "",
        "## Reading notes",
        "",
        "- Soft factor: hints vs stub_uniform (and filter vs filter_stub).",
        "- Hard factor: filter_stub vs stub_uniform; filter vs hints.",
        "- Eval-indexed @20k uses archive_trace; filter arms skip ~33% sims.",
        "- Do not treat partial n as confirmatory; relaunch remaining seeds.",
        "",
        "Script: `scripts/analyze_mixed_2x2.py`",
        "",
    ]
    (OUT / "ANALYSIS.md").write_text("\n".join(lines))
    print((OUT / "ANALYSIS.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

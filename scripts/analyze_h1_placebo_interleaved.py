#!/usr/bin/env python3
"""Analyze q1-h1-placebo-interleaved: paired hints vs hints_placebo.

Same calendar/workers by construction. Safe on partial seed sets.
Writes ANALYSIS.md + JSON under the tier root.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts/experiments/q1-h1-placebo-interleaved"
ARMS = ("hints", "hints_placebo")
OUT = EXP
FROZEN_HINTS = ROOT / "artifacts/experiments/q1-v3-mixed-2x2/hints"
TOST_BAND_PP = 2.0


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
            if 0 <= ev <= budget:
                cov = float(row["coverage"])
                best = cov * 100.0 if cov <= 1.5 else cov
    return best


def _paired_tost(diffs: list[float], *, band: float = TOST_BAND_PP) -> dict[str, Any]:
    arr = np.asarray(diffs, dtype=float)
    n = int(arr.size)
    if n < 2:
        return {"n": n, "decision": "n/a", "note": "need n>=2"}
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    se = sd / np.sqrt(n)
    # two one-sided t tests vs ±band
    t_low = (mean - (-band)) / se
    t_high = (band - mean) / se
    df = n - 1
    p_low = float(stats.t.sf(t_low, df))
    p_high = float(stats.t.sf(t_high, df))
    p_tost = max(p_low, p_high)
    ci = stats.t.interval(0.90, df, loc=mean, scale=se)
    accept = p_tost < 0.05 and abs(mean) <= band
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "p_tost": p_tost,
        "ci90": [float(ci[0]), float(ci[1])],
        "band_pp": band,
        "decision": "ACCEPT" if accept else "REJECT",
        "note": "exploratory descriptive only — not confirmatory",
    }


def load_seed(arm_root: Path, arm: str, seed: int) -> dict[str, Any] | None:
    d = arm_root / f"seed_{seed}"
    summary = d / "nightly_run_summary.json"
    if not summary.is_file():
        return None
    payload = json.loads(summary.read_text())
    cov = float(payload.get("coverage", 0.0))
    cov_pct = cov * 100.0 if cov <= 1.5 else cov
    row: dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "coverage_pct": cov_pct,
        "evaluations": int(payload.get("evaluations", 0)),
        "wall_min": float(payload.get("elapsed_seconds", 0.0)) / 60.0,
        "llm_parallel_workers": payload.get("llm_parallel_workers"),
        "llm_emit_attempts": payload.get("llm_emit_attempts"),
        "llm_emit_fallbacks": payload.get("llm_emit_fallbacks"),
    }
    qd = payload.get("qd_score")
    if qd is not None:
        row["qd_score"] = float(qd)
    best = payload.get("best_fitness")
    if best is not None:
        row["best_fitness"] = float(best)
    trace = d / "archive_trace.jsonl"
    if trace.is_file():
        row["cov_at_20000"] = _cov_at(trace, 20000)
    return row


def main() -> int:
    by_arm: dict[str, dict[int, dict[str, Any]]] = {a: {} for a in ARMS}
    for arm in ARMS:
        for seed in range(10):
            row = load_seed(EXP / arm, arm, seed)
            if row is not None:
                by_arm[arm][seed] = row

    complete = sorted(set.intersection(*(set(by_arm[a]) for a in ARMS)))
    status = "complete" if len(complete) >= 10 else f"partial_n={len(complete)}"

    levels = {}
    for arm in ARMS:
        rows = [by_arm[arm][s] for s in complete]
        levels[arm] = {
            "n": len(rows),
            "coverage_pct": _mean_sd([r["coverage_pct"] for r in rows]),
            "cov_at_20000": _mean_sd(
                [r["cov_at_20000"] for r in rows if r.get("cov_at_20000") is not None]
            ),
            "wall_min": _mean_sd([r["wall_min"] for r in rows]),
        }

    diffs_term = [
        by_arm["hints_placebo"][s]["coverage_pct"] - by_arm["hints"][s]["coverage_pct"]
        for s in complete
    ]
    diffs_20k = [
        by_arm["hints_placebo"][s]["cov_at_20000"] - by_arm["hints"][s]["cov_at_20000"]
        for s in complete
        if by_arm["hints_placebo"][s].get("cov_at_20000") is not None
        and by_arm["hints"][s].get("cov_at_20000") is not None
    ]

    wilcoxon_p = None
    if len(diffs_term) >= 5 and any(d != 0 for d in diffs_term):
        wilcoxon_p = float(
            cast(Any, stats.wilcoxon(diffs_term, alternative="two-sided")).pvalue
        )

    contrast = {
        "delta_terminal": dict(_mean_sd(diffs_term)),
        "delta_cov20k": dict(_mean_sd(diffs_20k)),
        "n_positive_terminal": int(sum(1 for d in diffs_term if d > 0)),
        "n_positive_cov20k": int(sum(1 for d in diffs_20k if d > 0)),
        "wilcoxon_p_terminal": wilcoxon_p,
        "tost_placebo_equiv_hints": _paired_tost(diffs_term) if diffs_term else None,
    }

    gate = None
    if complete:
        dt = float(contrast["delta_terminal"]["mean"])
        d20 = (
            float(contrast["delta_cov20k"]["mean"])
            if contrast["delta_cov20k"]["n"]
            else 0.0
        )
        extend = abs(dt) >= 2.0 or (
            contrast["delta_cov20k"]["n"] > 0 and abs(d20) >= 2.0
        )
        gate = {
            "extend_to_n10": extend,
            "reason": (
                f"|Δcov_term|={abs(dt):.2f}, |Δcov@20k|={abs(d20):.2f}; "
                f"{'EXTEND' if extend else 'STOP (matched negative / flat pilot)'}"
            ),
        }

    # Optional companion: interleaved hints vs frozen mixed-2x2 hints (same seeds)
    frozen_deltas = []
    for s in complete:
        fr = load_seed(FROZEN_HINTS, "hints_frozen", s)
        if fr is None:
            continue
        frozen_deltas.append(by_arm["hints"][s]["coverage_pct"] - fr["coverage_pct"])

    payload = {
        "tier": "q1-h1-placebo-interleaved",
        "protocol": "artifacts/Q1_H1_PLACEBO_INTERLEAVED.md",
        "status": status,
        "complete_seeds": complete,
        "levels": levels,
        "contrast_placebo_minus_hints": contrast,
        "pilot_gate": gate,
        "interleaved_hints_minus_frozen_mixed2x2_hints": dict(_mean_sd(frozen_deltas)),
        "per_seed": {
            s: {
                "hints": by_arm["hints"][s]["coverage_pct"],
                "hints_placebo": by_arm["hints_placebo"][s]["coverage_pct"],
                "delta_pp": diffs_term[complete.index(s)],
                "cov20k_hints": by_arm["hints"][s].get("cov_at_20000"),
                "cov20k_placebo": by_arm["hints_placebo"][s].get("cov_at_20000"),
                "workers_hints": by_arm["hints"][s].get("llm_parallel_workers"),
                "workers_placebo": by_arm["hints_placebo"][s].get(
                    "llm_parallel_workers"
                ),
            }
            for s in complete
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "h1_placebo_interleaved_analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# H1 interleaved placebo analysis",
        "",
        f"**Status:** `{status}` · seeds={complete}",
        "",
        "Paired `hints_placebo` − `hints` under one calendar / one worker setting "
        "(protocol `artifacts/Q1_H1_PLACEBO_INTERLEAVED.md`). Descriptive only.",
        "",
        "## Levels",
        "",
        "| Arm | n | Coverage % | Cov@20k % | Wall min |",
        "|-----|---|------------:|----------:|---------:|",
    ]
    for arm in ARMS:
        lv = levels[arm]
        c20 = lv["cov_at_20000"]
        c20_s = f"{c20['mean']:.2f}±{c20['sd']:.2f}" if c20["n"] else "—"
        lines.append(
            f"| `{arm}` | {lv['n']} | "
            f"{lv['coverage_pct']['mean']:.2f}±{lv['coverage_pct']['sd']:.2f} | "
            f"{c20_s} | "
            f"{lv['wall_min']['mean']:.1f}±{lv['wall_min']['sd']:.1f} |"
        )
    c = contrast
    tost = c.get("tost_placebo_equiv_hints") or {}
    lines += [
        "",
        "## Contrast (hints_placebo − hints)",
        "",
        f"- Δ terminal: {c['delta_terminal']['mean']:+.2f}±{c['delta_terminal']['sd']:.2f} "
        f"pp ({c['n_positive_terminal']}/{c['delta_terminal']['n']})",
        f"- Δ cov@20k: {c['delta_cov20k']['mean']:+.2f}±{c['delta_cov20k']['sd']:.2f} "
        f"pp ({c['n_positive_cov20k']}/{c['delta_cov20k']['n']})",
        f"- Wilcoxon Δterm p: {wilcoxon_p}",
        f"- TOST ±2 pp placebo≡hints: {tost.get('decision')} "
        f"(p={tost.get('p_tost')}; 90% CI {tost.get('ci90')}) — exploratory only",
        "",
    ]
    if gate is not None:
        lines += ["## Pilot gate", "", f"- {gate['reason']}", ""]
    if complete:
        lines += [
            "## Per-seed (placebo − hints)",
            "",
            "| Seed | hints | placebo | Δterm | workers |",
            "|-----:|------:|--------:|------:|--------:|",
        ]
        for s in complete:
            h = by_arm["hints"][s]
            p = by_arm["hints_placebo"][s]
            d = p["coverage_pct"] - h["coverage_pct"]
            w = h.get("llm_parallel_workers")
            lines.append(
                f"| {s} | {h['coverage_pct']:.2f} | {p['coverage_pct']:.2f} | "
                f"{d:+.2f} | {w} |"
            )
        lines.append("")
    fd = payload["interleaved_hints_minus_frozen_mixed2x2_hints"]
    lines += [
        "## Companion: interleaved hints − frozen mixed-2x2 hints",
        "",
        f"- mean Δ: {fd['mean']:+.2f}±{fd['sd']:.2f} pp (n={fd['n']}) — "
        "calendar drift check, not primary",
        "",
        "JSON: `artifacts/experiments/q1-h1-placebo-interleaved/"
        "h1_placebo_interleaved_analysis.json`",
        "",
    ]
    (OUT / "ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

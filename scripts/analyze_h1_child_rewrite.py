#!/usr/bin/env python3
"""Analyze q1-h1-child-rewrite-pilot: hints vs hints_rewrite.

Safe on partial seed sets. Writes ANALYSIS.md + JSON under the tier root.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts/experiments/q1-h1-child-rewrite-pilot"
ARMS = ("hints", "hints_rewrite")
OUT = EXP


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


def load_seed(arm: str, seed: int) -> dict[str, Any] | None:
    d = EXP / arm / f"seed_{seed}"
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
    }
    trace = d / "archive_trace.jsonl"
    if trace.is_file():
        row["cov_at_20000"] = _cov_at(trace, 20000)
    arch = d / "map_elites_archive.jsonl"
    if arch.is_file() and arm == "hints_rewrite":
        types: Counter[str] = Counter()
        with arch.open() as fh:
            for line in fh:
                rec = json.loads(line)
                meta = rec.get("metadata") or {}
                et = str(meta.get("emitter_type") or "")
                if et.startswith("llm"):
                    types[et] += 1
        row["llm_emitter_type_counts"] = dict(types)
    return row


def main() -> int:
    by_arm: dict[str, dict[int, dict[str, Any]]] = {a: {} for a in ARMS}
    for arm in ARMS:
        for seed in range(10):
            row = load_seed(arm, seed)
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
        by_arm["hints_rewrite"][s]["coverage_pct"] - by_arm["hints"][s]["coverage_pct"]
        for s in complete
    ]
    diffs_20k = [
        by_arm["hints_rewrite"][s]["cov_at_20000"] - by_arm["hints"][s]["cov_at_20000"]
        for s in complete
        if by_arm["hints_rewrite"][s].get("cov_at_20000") is not None
        and by_arm["hints"][s].get("cov_at_20000") is not None
    ]
    contrast = {
        "delta_terminal": dict(_mean_sd(diffs_term)),
        "delta_cov20k": dict(_mean_sd(diffs_20k)),
        "n_positive_terminal": int(sum(1 for d in diffs_term if d > 0)),
        "n_positive_cov20k": int(sum(1 for d in diffs_20k if d > 0)),
    }
    if complete:
        contrast["delta_terminal"]["n_positive"] = contrast["n_positive_terminal"]
        contrast["delta_cov20k"]["n_positive"] = contrast["n_positive_cov20k"]

    # Pilot gate from protocol
    gate = None
    if complete:
        dt = float(contrast["delta_terminal"]["mean"])
        d20 = float(contrast["delta_cov20k"]["mean"])
        extend = abs(dt) >= 2.0 or (
            contrast["delta_cov20k"]["n"] > 0 and abs(d20) >= 2.0
        )
        gate = {
            "extend_to_n10": extend,
            "reason": (
                f"|Δcov_term|={abs(dt):.2f}, |Δcov@20k|={abs(d20):.2f}; "
                f"{'EXTEND' if extend else 'STOP (negative pilot)'}"
            ),
        }

    payload = {
        "tier": "q1-h1-child-rewrite-pilot",
        "status": status,
        "complete_seeds": complete,
        "levels": levels,
        "contrast_rewrite_minus_hints": contrast,
        "pilot_gate": gate,
        "per_seed": {a: list(by_arm[a].values()) for a in ARMS},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "h1_child_rewrite_analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    lines = [
        "# H1 child-rewrite analysis",
        "",
        f"**Status:** `{status}` · seeds={complete}",
        "",
        "## Levels",
        "",
        "| Arm | n | Coverage % | Cov@20k % | Wall min |",
        "|-----|---|------------:|----------:|---------:|",
    ]
    for arm in ARMS:
        lv = levels[arm]
        lines.append(
            f"| `{arm}` | {lv['n']} | "
            f"{lv['coverage_pct']['mean']:.2f}±{lv['coverage_pct']['sd']:.2f} | "
            f"{lv['cov_at_20000']['mean']:.2f}±{lv['cov_at_20000']['sd']:.2f} | "
            f"{lv['wall_min']['mean']:.1f}±{lv['wall_min']['sd']:.1f} |"
        )
    c = contrast
    lines += [
        "",
        "## Contrast (hints_rewrite − hints)",
        "",
        f"- Δ terminal: {c['delta_terminal']['mean']:+.2f}±{c['delta_terminal']['sd']:.2f} "
        f"pp ({c['n_positive_terminal']}/{c['delta_terminal']['n']})",
        f"- Δ cov@20k: {c['delta_cov20k']['mean']:+.2f}±{c['delta_cov20k']['sd']:.2f} "
        f"pp ({c['n_positive_cov20k']}/{c['delta_cov20k']['n']})",
        "",
    ]
    if gate is not None:
        lines += ["## Pilot gate", "", f"- {gate['reason']}", ""]
    lines += [
        "Protocol: `artifacts/Q1_H1_CHILD_REWRITE.md`",
        "Script: `scripts/analyze_h1_child_rewrite.py`",
        "",
    ]
    (OUT / "ANALYSIS.md").write_text("\n".join(lines))
    print((OUT / "ANALYSIS.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

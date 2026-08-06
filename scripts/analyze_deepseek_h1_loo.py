#!/usr/bin/env python3
"""Offline leave-one-out for DeepSeek matched H1 (no new runs).

Reads frozen per-seed Δcov from h1_matched_analysis.json and recomputes
paired-t mean-TOST ±2 pp after dropping each seed in turn.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_q1_statistics import bootstrap_ci, param_tost

SRC = ROOT / "artifacts/experiments/q1-v3-llm/deepseek-v4-pro/h1_matched_analysis.json"
OUT_JSON = (
    ROOT / "artifacts/experiments/q1-v3-llm/deepseek-v4-pro/deepseek_outlier_loo.json"
)
OUT_MD = (
    ROOT / "artifacts/experiments/q1-v3-llm/deepseek-v4-pro/DEEPSEEK_OUTLIER_DIAG.md"
)
EQ_MARGIN_PP = 2.0


def _summary(
    deltas: np.ndarray, *, seeds: list[int], dropped: int | None
) -> dict[str, Any]:
    tost = param_tost(deltas, EQ_MARGIN_PP)
    ci95 = bootstrap_ci(deltas, stat="mean", level=0.95)
    return {
        "dropped_seed": dropped,
        "n": int(len(deltas)),
        "seeds_kept": seeds,
        "mean_delta_cov_pp": round(float(np.mean(deltas)), 4),
        "sd_delta_cov_pp": round(float(np.std(deltas, ddof=1)), 4),
        "wins_cov": int(np.sum(deltas > 0)),
        "bootstrap_ci95_mean_delta_pp": [round(ci95[0], 4), round(ci95[1], 4)],
        "tost_2pp_mean": {
            "accepted": bool(tost["accepted"]),
            "p_tost": float(tost["p_tost"]),
            "ci90": [float(tost["ci90"][0]), float(tost["ci90"][1])],
            "level": tost["level"],
        },
    }


def main() -> int:
    payload = json.loads(SRC.read_text(encoding="utf-8"))
    rows = payload["per_seed"]
    seeds = [int(r["seed"]) for r in rows]
    deltas = np.asarray([float(r["delta_cov_pp"]) for r in rows], dtype=float)

    full = _summary(deltas, seeds=seeds, dropped=None)
    loo: list[dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        keep = np.ones(len(deltas), dtype=bool)
        keep[i] = False
        kept_seeds = [s for j, s in enumerate(seeds) if j != i]
        loo.append(_summary(deltas[keep], seeds=kept_seeds, dropped=seed))

    # Cook-style influence: change in mean and whether TOST flips.
    full_accept = full["tost_2pp_mean"]["accepted"]
    full_mean = full["mean_delta_cov_pp"]
    for row in loo:
        row["delta_mean_vs_full_pp"] = round(row["mean_delta_cov_pp"] - full_mean, 4)
        row["tost_flips_vs_full"] = bool(
            row["tost_2pp_mean"]["accepted"] != full_accept
        )

    out = {
        "source": str(SRC.relative_to(ROOT)),
        "contrast": "hints_minus_stub_uniform",
        "provider": payload.get("provider"),
        "eq_margin_pp": EQ_MARGIN_PP,
        "note": (
            "Offline leave-one-out on frozen matched H1 deltas. "
            "No new API / simulator runs. Manuscript claim stays: full-n mean-TOST rejects; "
            "LOO is sensitivity / outlier influence only."
        ),
        "full_n": full,
        "leave_one_out": loo,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# DeepSeek matched H1 — offline outlier / leave-one-out",
        "",
        "**Scope:** frozen `hints − stub_uniform` @ `uniform_frontier` "
        "(DeepSeek V4 Pro). **No new runs.**",
        f"**Source:** `{SRC.relative_to(ROOT)}`",
        "",
        "## Full n=10 (reference)",
        "",
        f"| Mean Δcov | SD | Wins | Mean-TOST ±{EQ_MARGIN_PP:g} pp | 90% CI |",
        "|----------:|---:|-----:|:-----------------------------|-------:|",
        (
            f"| {full['mean_delta_cov_pp']:+.2f} | {full['sd_delta_cov_pp']:.2f} | "
            f"{full['wins_cov']}/{full['n']} | "
            f"{'accept' if full['tost_2pp_mean']['accepted'] else 'reject'} "
            f"(p={full['tost_2pp_mean']['p_tost']:.4g}) | "
            f"[{full['tost_2pp_mean']['ci90'][0]:+.2f}, "
            f"{full['tost_2pp_mean']['ci90'][1]:+.2f}] |"
        ),
        "",
        "## Leave-one-out (drop one seed, recompute mean-TOST)",
        "",
        "| Dropped | n | Mean Δcov | SD | Wins | Mean-TOST | 90% CI | Δmean vs full | Flips? |",
        "|--------:|--:|----------:|---:|-----:|:----------|-------:|--------------:|:------:|",
    ]
    for row in loo:
        t = row["tost_2pp_mean"]
        lines.append(
            f"| {row['dropped_seed']} | {row['n']} | "
            f"{row['mean_delta_cov_pp']:+.2f} | {row['sd_delta_cov_pp']:.2f} | "
            f"{row['wins_cov']}/{row['n']} | "
            f"{'accept' if t['accepted'] else 'reject'} "
            f"(p={t['p_tost']:.4g}) | "
            f"[{t['ci90'][0]:+.2f}, {t['ci90'][1]:+.2f}] | "
            f"{row['delta_mean_vs_full_pp']:+.2f} | "
            f"{'yes' if row['tost_flips_vs_full'] else 'no'} |"
        )

    seed8 = next(r for r in loo if r["dropped_seed"] == 8)
    seed8_delta = next(float(r["delta_cov_pp"]) for r in rows if int(r["seed"]) == 8)
    lines += [
        "",
        "## Reading",
        "",
        f"- Full-n mean-TOST **rejects** (90% CI exits +{EQ_MARGIN_PP:g} pp) — unchanged manuscript claim.",
        (
            f"- Dropping **seed 8** alone ({seed8_delta:+.2f} pp): "
            f"mean Δcov → {seed8['mean_delta_cov_pp']:+.2f} pp, "
            f"SD → {seed8['sd_delta_cov_pp']:.2f}, "
            f"mean-TOST → **{'accept' if seed8['tost_2pp_mean']['accepted'] else 'reject'}** "
            f"(90% CI [{seed8['tost_2pp_mean']['ci90'][0]:+.2f}, "
            f"{seed8['tost_2pp_mean']['ci90'][1]:+.2f}])."
        ),
        "- No other single-seed drop flips reject→accept (see Flips? column).",
        "- Interpretation: DeepSeek matched-H1 “outlier” is **seed-8 influence on the mean/SD**, "
        "not a provider-wide hints lift. Do **not** amend confirmatory claims from LOO; "
        "do **not** co-claim bootstrap-median TOST.",
        "",
        f"JSON: `{OUT_JSON.relative_to(ROOT)}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD)
    print(OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

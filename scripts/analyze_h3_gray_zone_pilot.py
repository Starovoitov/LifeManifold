#!/usr/bin/env python3
"""F-RQ3-gray Holm/NI: filter_gray_zone vs frozen q1-full/hints (pilot or confirmatory tier)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import stats as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_q1_statistics import (
    holm_step_down,
    noninferiority,
    vargha_delaney_a12_paired,
)

PILOT_ROOT = ROOT / "artifacts/experiments/q1-v3-h3-gray-zone-pilot"
CONFIRM_ROOT = ROOT / "artifacts/experiments/q1-v3-h3-gray-zone"
HINTS_ROOT = ROOT / "artifacts/experiments/q1-full"
PROPOSALS = 32500


def _fitness_from_csv(summary_csv: Path, condition: str, seed: int) -> float | None:
    if not summary_csv.exists():
        return None
    for row in csv.DictReader(summary_csv.open()):
        if row.get("condition") == condition and int(row["seed"]) == seed:
            val = row.get("mean_best_fitness")
            return float(val) if val not in (None, "") else None
    return None


def _load_summary(
    path: Path,
    *,
    summary_csv: Path | None = None,
    condition: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    data = json.loads(path.read_text())
    cov = data.get("coverage") or data.get("coverage_pct", 0.0)
    if cov <= 1.5:
        cov *= 100.0
    fit = float(data.get("mean_best_fitness") or data.get("fitness") or 0.0)
    if summary_csv and condition is not None and seed is not None:
        csv_fit = _fitness_from_csv(summary_csv, condition, seed)
        if csv_fit is not None:
            fit = csv_fit
    evals = int(data.get("evaluations", 0))
    skip = 100.0 * (1.0 - evals / PROPOSALS) if PROPOSALS else 0.0
    return {
        "coverage_pct": float(cov),
        "mean_best_fitness": fit,
        "evaluations": evals,
        "skip_rate_pct": skip,
        "elapsed_seconds": float(data.get("elapsed_seconds", 0.0)),
    }


def _paired_rows(seeds: list[int], *, tier_root: Path) -> list[dict[str, Any]]:
    hints_csv = HINTS_ROOT / "summary.csv"
    gz_csv = tier_root / "summary.csv"
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        gz = _load_summary(
            tier_root
            / "filter_gray_zone"
            / f"seed_{seed}"
            / "nightly_run_summary.json",
            summary_csv=gz_csv,
            condition="filter_gray_zone",
            seed=seed,
        )
        hints = _load_summary(
            HINTS_ROOT / "hints" / f"seed_{seed}" / "nightly_run_summary.json",
            summary_csv=hints_csv,
            condition="hints",
            seed=seed,
        )
        rows.append(
            {
                "seed": seed,
                "hints": hints,
                "gray_zone": gz,
                "delta_cov_pp": gz["coverage_pct"] - hints["coverage_pct"],
                "delta_fit": gz["mean_best_fitness"] - hints["mean_best_fitness"],
                "delta_fit_rel": (
                    (gz["mean_best_fitness"] - hints["mean_best_fitness"])
                    / hints["mean_best_fitness"]
                    if hints["mean_best_fitness"]
                    else 0.0
                ),
                "delta_eval": gz["evaluations"] - hints["evaluations"],
            }
        )
    return rows


def analyze(
    seeds: list[int],
    *,
    tier_root: Path,
    confirmatory: bool,
) -> dict[str, Any]:
    rows = _paired_rows(seeds, tier_root=tier_root)
    d_cov = np.array([r["delta_cov_pp"] for r in rows], dtype=float)
    d_fit_rel = np.array([r["delta_fit_rel"] for r in rows], dtype=float)
    d_eval = np.array([r["delta_eval"] for r in rows], dtype=float)

    p_eval = float(cast(Any, sp.wilcoxon(d_eval, alternative="less")).pvalue)
    ni_cov = noninferiority(d_cov, neg_margin=-3.0)
    ni_fit = noninferiority(d_fit_rel, neg_margin=-0.05)

    holm = holm_step_down(
        {
            "eval_less_hints": p_eval,
            "cov_ni_minus3pp": ni_cov["p"],
            "fit_ni_minus5pct": ni_fit["p"],
        }
    )
    family_pass = all(holm.values())

    skip_rates = [r["gray_zone"]["skip_rate_pct"] for r in rows]
    tier_name = tier_root.name
    family = "F-RQ3-gray" if confirmatory else "F-RQ3-gray-exploratory"
    payload: dict[str, Any] = {
        "family": family,
        "n": len(seeds),
        "seeds": seeds,
        "control": "q1-full/hints (frozen)",
        "treatment": f"{tier_name}/filter_gray_zone",
        "confirmatory": confirmatory,
        "note": (
            "Confirmatory F-RQ3-gray Holm/NI on pre-registered duplicate tier; "
            "does not rehabilitate historical production filter @ 33.5% skip."
            if confirmatory
            else (
                "Exploratory Holm/NI read on completed pilot runs; not a pre-registered "
                "confirmatory unlock of historical production filter."
            )
        ),
        "mean_delta_cov_pp": round(float(np.mean(d_cov)), 4),
        "sd_delta_cov_pp": round(float(np.std(d_cov, ddof=1)), 4),
        "wins_cov": int(np.sum(d_cov > 0)),
        "mean_skip_rate_pct": round(float(np.mean(skip_rates)), 3),
        "mean_eval_reduction_pct": round(
            100.0 * float(np.mean(-d_eval / PROPOSALS)), 4
        ),
        "tests": {
            "eval_less_hints": {
                "mean_delta_eval": round(float(np.mean(d_eval)), 1),
                "raw_p": p_eval,
                "a12": vargha_delaney_a12_paired(d_eval, direction="less"),
                "holm_reject": holm["eval_less_hints"],
                "wins": int(np.sum(d_eval < 0)),
            },
            "cov_ni_minus3pp": {
                "mean_delta_pp": round(float(np.mean(d_cov)), 4),
                "raw_p": ni_cov["p"],
                "ni_accepted": ni_cov["accepted"],
                "ci": [round(ni_cov["ci"][0], 4), round(ni_cov["ci"][1], 4)],
                "a12": vargha_delaney_a12_paired(d_cov, direction="greater"),
                "holm_reject": holm["cov_ni_minus3pp"],
                "per_seed_pass": int(np.sum(d_cov > -3.0)),
            },
            "fit_ni_minus5pct": {
                "mean_delta_rel": round(float(np.mean(d_fit_rel)), 4),
                "raw_p": ni_fit["p"],
                "ni_accepted": ni_fit["accepted"],
                "ci": [round(ni_fit["ci"][0], 4), round(ni_fit["ci"][1], 4)],
                "a12": vargha_delaney_a12_paired(d_fit_rel, direction="greater"),
                "holm_reject": holm["fit_ni_minus5pct"],
            },
        },
        "holm_m": 3,
        "family_pass": family_pass,
        "per_seed": rows,
    }
    return payload


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    t = payload["tests"]
    tier_label = "Confirmatory" if payload["confirmatory"] else "Exploratory"
    lines = [
        f"## {tier_label} Holm / NI (F-RQ3-gray margins)",
        "",
        f"**Tier:** `{payload['treatment'].split('/')[0]}` · "
        f"**n={payload['n']}** paired seeds vs frozen `q1-full/hints`. "
        f"**Family pass ({tier_label.lower()}):** `{payload['family_pass']}`. "
        "Does not rehabilitate historical production `filter`.",
        "",
        "| Test | Mean Δ | Raw p (method) | Holm @0.05 | PPS | Pass detail |",
        "|------|--------|----------------|------------|-----|-------------|",
        (
            f"| eval ↓ (gray − hints) | {t['eval_less_hints']['mean_delta_eval']:.0f} sims | "
            f"{t['eval_less_hints']['raw_p']:.4g} (Wilcoxon) | "
            f"{'**Yes**' if t['eval_less_hints']['holm_reject'] else 'No'} | "
            f"{t['eval_less_hints']['a12']:.2f} | {t['eval_less_hints']['wins']}/{payload['n']} fewer evals |"
        ),
        (
            f"| cov NI (Δ > −3 pp) | {t['cov_ni_minus3pp']['mean_delta_pp']:+.2f} pp | "
            f"{t['cov_ni_minus3pp']['raw_p']:.4g} (bootstrap/t NI) | "
            f"{'**Yes**' if t['cov_ni_minus3pp']['holm_reject'] else 'No'} | "
            f"{t['cov_ni_minus3pp']['a12']:.2f} | "
            f"NI accept={t['cov_ni_minus3pp']['ni_accepted']}; "
            f"{t['cov_ni_minus3pp']['per_seed_pass']}/{payload['n']} per-seed |"
        ),
        (
            f"| fit NI (Δ_rel > −5%) | {100*t['fit_ni_minus5pct']['mean_delta_rel']:+.2f}% | "
            f"{t['fit_ni_minus5pct']['raw_p']:.4g} (bootstrap/t NI) | "
            f"{'**Yes**' if t['fit_ni_minus5pct']['holm_reject'] else 'No'} | "
            f"{t['fit_ni_minus5pct']['a12']:.2f} | "
            f"NI accept={t['fit_ni_minus5pct']['ni_accepted']} |"
        ),
        "",
        "Note: NI raw p-values are one-sided t or bootstrap-median NI "
        "(protocol `noninferiority`), not exact Wilcoxon; they may be "
        f"< 1/2^n ≈ {1/2**payload['n']:.4g}. PPS = paired favorable-pair rate "
        "(JSON key `a12` kept for frozen-artifact compatibility).",
        "",
    ]
    path.write_text("\n".join(lines))


def write_analysis_md(payload: dict[str, Any], path: Path) -> None:
    t = payload["tests"]
    tier = payload["treatment"].split("/")[0]
    tier_label = "confirmatory" if payload["confirmatory"] else "exploratory pilot"
    rows = payload["per_seed"]
    g_cov = [r["gray_zone"]["coverage_pct"] for r in rows]
    h_cov = [r["hints"]["coverage_pct"] for r in rows]
    g_fit = [r["gray_zone"]["mean_best_fitness"] for r in rows]
    skips = [r["gray_zone"]["skip_rate_pct"] for r in rows]
    lines = [
        f"# H3 gray-zone {tier_label} — `{tier}`",
        "",
        f"**Tier:** `{tier}` · **Status:** seeds 0–9 DONE ({payload['n']}/{payload['n']})",
        f"**Family:** `{payload['family']}` · **Confirmatory:** `{payload['confirmatory']}`",
        "",
        "## Mechanism check",
        "",
        f"- Skip rate: **{payload['mean_skip_rate_pct']:.2f}%** (target 8–18%)",
        f"- Mean eval reduction vs hints: **{payload['mean_eval_reduction_pct']:.2f}%**",
        "",
        "## Mean levels",
        "",
        "| Arm | Cov % | Mean fit | Skip % |",
        "|-----|------:|---------:|-------:|",
        (
            f"| `hints` (frozen) | {np.mean(h_cov):.2f} ± {np.std(h_cov, ddof=1):.2f} "
            f"| {np.mean([r['hints']['mean_best_fitness'] for r in rows]):.3f} | 0.0 |"
        ),
        (
            f"| `filter_gray_zone` | {np.mean(g_cov):.2f} ± {np.std(g_cov, ddof=1):.2f} "
            f"| {np.mean(g_fit):.3f} | {np.mean(skips):.2f} |"
        ),
        "",
        "## Paired vs frozen hints",
        "",
        (
            f"- Mean Δcov: **{payload['mean_delta_cov_pp']:+.2f} pp** "
            f"(SD {payload['sd_delta_cov_pp']:.2f}; {payload['wins_cov']}/{payload['n']} wins)"
        ),
        f"- Family pass (Holm $m=3$): **{payload['family_pass']}**",
        "",
        "## Holm / NI",
        "",
        "| Test | Mean Δ | Raw p | Holm | Detail |",
        "|------|--------|-------|------|--------|",
        (
            f"| eval ↓ | {t['eval_less_hints']['mean_delta_eval']:.0f} sims | "
            f"{t['eval_less_hints']['raw_p']:.4g} | "
            f"{'Yes' if t['eval_less_hints']['holm_reject'] else 'No'} | "
            f"{t['eval_less_hints']['wins']}/{payload['n']} fewer |"
        ),
        (
            f"| cov NI (>−3 pp) | {t['cov_ni_minus3pp']['mean_delta_pp']:+.2f} pp | "
            f"{t['cov_ni_minus3pp']['raw_p']:.4g} | "
            f"{'Yes' if t['cov_ni_minus3pp']['holm_reject'] else 'No'} | "
            f"{t['cov_ni_minus3pp']['per_seed_pass']}/{payload['n']} per-seed |"
        ),
        (
            f"| fit NI (>−5% rel.) | {100*t['fit_ni_minus5pct']['mean_delta_rel']:+.2f}% | "
            f"{t['fit_ni_minus5pct']['raw_p']:.4g} | "
            f"{'Yes' if t['fit_ni_minus5pct']['holm_reject'] else 'No'} | "
            f"NI accept={t['fit_ni_minus5pct']['ni_accepted']} |"
        ),
        "",
        "See [`H3_GRAY_HOLM.md`](H3_GRAY_HOLM.md) and JSON artifact in this directory.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument(
        "--root",
        type=Path,
        default=PILOT_ROOT,
        help="Experiment tier root (pilot or confirmatory)",
    )
    parser.add_argument(
        "--confirmatory",
        action="store_true",
        help="Mark as confirmatory F-RQ3-gray (default: exploratory pilot)",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-analysis-md", type=Path, default=None)
    args = parser.parse_args()

    tier_root = args.root.resolve()
    confirmatory = args.confirmatory or tier_root.name == CONFIRM_ROOT.name
    json_name = (
        "h3_gray_zone_confirmatory_holm.json"
        if confirmatory
        else "h3_gray_zone_pilot_holm.json"
    )
    output_json = args.output_json or tier_root / json_name
    output_md = args.output_md or tier_root / "H3_GRAY_HOLM.md"
    output_analysis = args.output_analysis_md or tier_root / "ANALYSIS.md"

    payload = analyze(args.seeds, tier_root=tier_root, confirmatory=confirmatory)
    output_json.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, output_md)
    write_analysis_md(payload, output_analysis)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

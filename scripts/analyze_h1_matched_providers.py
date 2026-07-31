#!/usr/bin/env python3
"""Matched-policy H1: stub_uniform vs hints per LLM provider (exploratory TOST)."""

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

from scripts.analyze_q1_statistics import bootstrap_ci, tost

PROVIDERS: dict[str, dict[str, str]] = {
    "gpt-4o-mini": {
        "slug": "gpt-4o-mini",
        "label": "openai/gpt-4o-mini",
        "tier": "q1-h1-matched-gpt-4o-mini",
    },
    "deepseek-v4-pro": {
        "slug": "deepseek-v4-pro",
        "label": "deepseek/deepseek-v4-pro",
        "tier": "q1-h1-matched-deepseek-v4-pro",
    },
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cov_pct(payload: dict[str, Any]) -> float:
    if "coverage_pct" in payload:
        cov = float(payload["coverage_pct"])
    else:
        cov = float(payload["coverage"])
    return cov * 100.0 if cov <= 1.0 else cov


def _fitness_from_csv(summary_csv: Path, condition: str, seed: int) -> float | None:
    if not summary_csv.exists():
        return None
    for row in csv.DictReader(summary_csv.open()):
        if row.get("condition") == condition and int(row["seed"]) == seed:
            val = row.get("mean_best_fitness")
            return float(val) if val not in (None, "") else None
    return None


def _fitness(
    payload: dict[str, Any], *, summary_csv: Path, condition: str, seed: int
) -> float:
    csv_fit = _fitness_from_csv(summary_csv, condition, seed)
    if csv_fit is not None:
        return csv_fit
    return float(payload.get("mean_best_fitness") or payload.get("fitness") or 0.0)


def _wilcoxon_two_sided(delta: np.ndarray) -> float:
    if len(delta) < 1 or np.allclose(delta, 0):
        return 1.0
    result = cast(Any, sp.wilcoxon(delta, alternative="two-sided"))
    return float(result.pvalue)


def analyze_provider(slug: str, seeds: list[int]) -> dict[str, Any]:
    meta = PROVIDERS[slug]
    root = ROOT / "artifacts/experiments/q1-v3-llm" / slug
    summary_csv = root / "summary.csv"
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        su_path = root / "stub_uniform" / f"seed_{seed}" / "nightly_run_summary.json"
        hi_path = root / "hints" / f"seed_{seed}" / "nightly_run_summary.json"
        if not su_path.is_file() or not hi_path.is_file():
            raise SystemExit(f"missing paired run for {slug} seed {seed}")
        su = _load(su_path)
        hi = _load(hi_path)
        su_fit = _fitness(
            su, summary_csv=summary_csv, condition="stub_uniform", seed=seed
        )
        hi_fit = _fitness(hi, summary_csv=summary_csv, condition="hints", seed=seed)
        rows.append(
            {
                "seed": seed,
                "stub_uniform_pct": round(_cov_pct(su), 4),
                "hints_pct": round(_cov_pct(hi), 4),
                "stub_uniform_fit": round(su_fit, 6),
                "hints_fit": round(hi_fit, 6),
                "delta_cov_pp": round(_cov_pct(hi) - _cov_pct(su), 4),
                "delta_fit": round(hi_fit - su_fit, 6),
            }
        )

    d_cov = np.asarray([r["delta_cov_pp"] for r in rows], dtype=float)
    d_fit = np.asarray([r["delta_fit"] for r in rows], dtype=float)
    su_cov = np.asarray([r["stub_uniform_pct"] for r in rows], dtype=float)
    hi_cov = np.asarray([r["hints_pct"] for r in rows], dtype=float)
    eq_margin_pp = 2.0
    tost_cov = tost(d_cov, eq_margin_pp)
    ci95 = bootstrap_ci(d_cov, stat="mean", level=0.95)

    return {
        "tier": meta["tier"],
        "provider": meta["label"],
        "slug": slug,
        "contrast": "hints_minus_stub_uniform",
        "policy": "uniform_frontier",
        "n": len(rows),
        "seeds": [r["seed"] for r in rows],
        "mean_stub_uniform_pct": round(float(np.mean(su_cov)), 4),
        "sd_stub_uniform_pct": round(float(np.std(su_cov, ddof=1)), 4),
        "mean_hints_pct": round(float(np.mean(hi_cov)), 4),
        "sd_hints_pct": round(float(np.std(hi_cov, ddof=1)), 4),
        "mean_delta_cov_pp": round(float(np.mean(d_cov)), 4),
        "sd_delta_cov_pp": round(float(np.std(d_cov, ddof=1)), 4),
        "mean_delta_fit": round(float(np.mean(d_fit)), 6),
        "sd_delta_fit": round(float(np.std(d_fit, ddof=1)), 6),
        "wins_cov": int(np.sum(d_cov > 0)),
        "wilcoxon_two_sided_p": _wilcoxon_two_sided(d_cov),
        "bootstrap_ci95_mean_delta_pp": [round(ci95[0], 4), round(ci95[1], 4)],
        "tost_2pp": tost_cov,
        "per_seed": rows,
        "note": (
            "Exploratory matched-policy H1 on additional provider; reuses frozen "
            "bundled hints runs. Post-hoc TOST |Δcov|≤2 pp — not confirmatory Holm."
        ),
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    slug = payload["slug"]
    lines = [
        f"# Matched H1 — {slug} (stub_uniform vs hints)",
        "",
        f"**Tier:** `{payload['tier']}` · **Policy:** `{payload['policy']}`",
        f"**n={payload['n']}** paired seeds · contrast: hints − stub_uniform",
        "",
        "## Mean levels",
        "",
        "| Arm | Coverage (%) | Mean fit |",
        "|-----|-------------:|---------:|",
        (
            f"| stub_uniform | {payload['mean_stub_uniform_pct']:.2f} ± "
            f"{payload['sd_stub_uniform_pct']:.2f} | — |"
        ),
        (
            f"| hints (frozen G1) | {payload['mean_hints_pct']:.2f} ± "
            f"{payload['sd_hints_pct']:.2f} | — |"
        ),
        "",
        "## Paired contrast (hints − stub_uniform)",
        "",
        "| Metric | Mean Δ | SD | Wins | Wilcoxon (2-sided) | TOST |Δ|≤2 pp |",
        "|--------|-------:|---:|-----:|-------------------:|----------------|",
        (
            f"| Coverage (pp) | {payload['mean_delta_cov_pp']:+.2f} | "
            f"{payload['sd_delta_cov_pp']:.2f} | {payload['wins_cov']}/{payload['n']} | "
            f"{payload['wilcoxon_two_sided_p']:.4g} | "
            f"{'accept' if payload['tost_2pp']['accepted'] else 'reject'} "
            f"(p={payload['tost_2pp']['p_tost']:.4g}) |"
        ),
        (
            f"| Fitness | {payload['mean_delta_fit']:+.4f} | "
            f"{payload['sd_delta_fit']:.4f} | — | — | — |"
        ),
        "",
        (
            f"Bootstrap 95% CI on mean Δcov: "
            f"[{payload['bootstrap_ci95_mean_delta_pp'][0]:+.2f}, "
            f"{payload['bootstrap_ci95_mean_delta_pp'][1]:+.2f}] pp."
        ),
        "",
        "## Per-seed Δcov (pp)",
        "",
        "| Seed | stub_uniform | hints | Δ |",
        "|------|-------------:|------:|--:|",
    ]
    for row in payload["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['stub_uniform_pct']:.2f} | "
            f"{row['hints_pct']:.2f} | {row['delta_cov_pp']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "Flat matched H1: live prompt scalars vs stub constants stay within "
                "±2 pp TOST band (same epistemic slot as qwen-turbo Table decomposition)."
            ),
            "",
            f"Artifact: `{path.with_suffix('.json').name}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        action="append",
        choices=sorted(PROVIDERS),
        help="Provider slug (default: all configured providers)",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=9)
    args = parser.parse_args()
    slugs = args.provider or sorted(PROVIDERS)
    seeds = list(range(args.seed_start, args.seed_end + 1))

    combined: dict[str, Any] = {"providers": {}, "seeds": seeds}
    for slug in slugs:
        payload = analyze_provider(slug, seeds)
        out_dir = ROOT / "artifacts/experiments/q1-v3-llm" / slug
        json_path = out_dir / "h1_matched_analysis.json"
        md_path = out_dir / "H1_MATCHED_ANALYSIS.md"
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        write_markdown(payload, md_path)
        combined["providers"][slug] = payload
        print(json.dumps(payload, indent=2))

    summary_path = (
        ROOT / "artifacts/experiments/q1-v3-llm/h1_matched_providers_summary.json"
    )
    summary_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

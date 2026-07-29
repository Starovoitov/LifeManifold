#!/usr/bin/env python3
"""Analyze supplementary Sphere H2 matched me_uniform vs me_filter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_trace(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _coverage_at_evals(trace: list[dict], evals: int) -> float | None:
    for row in trace:
        if int(row["evaluations"]) >= evals:
            return float(row["coverage"])
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=ROOT / "artifacts/experiments/q1-v3-sphere-h2",
    )
    args = p.parse_args()

    seeds = list(range(10))
    u_term, f_term, skips, evals_f = [], [], [], []
    matched_delta = []
    matched_budget = None

    for seed in seeds:
        u = _load_summary(
            args.root / "me_uniform" / f"seed_{seed:02d}" / "nightly_run_summary.json"
        )
        f = _load_summary(
            args.root / "me_filter" / f"seed_{seed:02d}" / "nightly_run_summary.json"
        )
        ut = _load_trace(
            args.root / "me_uniform" / f"seed_{seed:02d}" / "archive_trace.jsonl"
        )
        ft = _load_trace(
            args.root / "me_filter" / f"seed_{seed:02d}" / "archive_trace.jsonl"
        )
        u_term.append(100.0 * float(u["coverage"]))
        f_term.append(100.0 * float(f["coverage"]))
        skips.append(100.0 * float(f["skip_rate"]))
        evals_f.append(int(f["true_evaluations"]))

    # Matched budget = min final true evals across filter seeds
    matched_budget = int(min(evals_f))
    for seed in seeds:
        ut = _load_trace(
            args.root / "me_uniform" / f"seed_{seed:02d}" / "archive_trace.jsonl"
        )
        ft = _load_trace(
            args.root / "me_filter" / f"seed_{seed:02d}" / "archive_trace.jsonl"
        )
        uc = _coverage_at_evals(ut, matched_budget)
        fc = _coverage_at_evals(ft, matched_budget)
        if uc is None or fc is None:
            raise RuntimeError(f"missing matched coverage for seed {seed}")
        matched_delta.append(100.0 * (fc - uc))

    d = np.asarray(matched_delta, dtype=np.float64)
    term_d = np.asarray(f_term, dtype=np.float64) - np.asarray(u_term, dtype=np.float64)
    # one-sided: filter better on matched-eval coverage
    w = wilcoxon(d, alternative="greater", zero_method="wilcox")
    signs = int(np.sum(d > 0))

    md = args.root / "ANALYSIS.md"
    lines = [
        "# Sphere H2 (supplementary)",
        "",
        "Matched `me_uniform` vs `me_filter` on Fontaine linear-projection Sphere "
        "(D=20, 100×100 archive, 32,500 proposals/seed, n=10).",
        "",
        "Empty-bin proposals are never skipped (same rule as primary `threshold_gate`).",
        "",
        f"- Mean skip rate: **{np.mean(skips):.1f}%** ± {np.std(skips, ddof=1):.1f}",
        f"- Mean true evals (filter): **{np.mean(evals_f):.0f}**",
        f"- Terminal coverage Δ (filter−uniform @ fixed proposals): "
        f"**{np.mean(term_d):+.2f} pp** (sign {int(np.sum(term_d>0))}/10)",
        f"- Matched-eval coverage Δ @ {matched_budget} evals: "
        f"**{np.mean(d):+.2f} pp** (sign {signs}/10; Wilcoxon one-sided greater "
        f"p={w.pvalue:.4g})",
        "",
        "Descriptive / supplementary only — not a confirmatory Holm family.",
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

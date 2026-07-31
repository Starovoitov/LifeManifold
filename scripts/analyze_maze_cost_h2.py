#!/usr/bin/env python3
"""Analyze cost-scaled maze genetic vs genetic_filter wall times (descriptive)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]


def _coverage_pct(summary: dict) -> float:
    if "coverage_pct" in summary:
        return float(summary["coverage_pct"])
    cov = float(summary.get("coverage", 0.0))
    return 100.0 * cov if cov <= 1.0 else cov


def _elapsed(summary: dict) -> float:
    for key in ("elapsed_seconds", "wall_seconds", "elapsed_s"):
        if key in summary:
            return float(summary[key])
    raise KeyError("no elapsed field in summary")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=ROOT / "artifacts/experiments/q1-v5-maze-cost-h2",
    )
    p.add_argument("--sim-cost-ms", type=float, default=10.0)
    args = p.parse_args()

    g_wall, f_wall, g_cov, f_cov, skips = [], [], [], [], []
    for seed in range(10):
        g = json.loads(
            (
                args.root / "genetic" / f"seed_{seed}" / "nightly_run_summary.json"
            ).read_text()
        )
        f = json.loads(
            (
                args.root
                / "genetic_filter"
                / f"seed_{seed}"
                / "nightly_run_summary.json"
            ).read_text()
        )
        g_wall.append(_elapsed(g))
        f_wall.append(_elapsed(f))
        g_cov.append(_coverage_pct(g))
        f_cov.append(_coverage_pct(f))
        skipped = float(f.get("skipped", 0))
        props = float(f.get("proposals", 1))
        skips.append(100.0 * skipped / props if props else float("nan"))

    d_wall = np.asarray(g_wall, dtype=np.float64) - np.asarray(f_wall, dtype=np.float64)
    w = cast(Any, wilcoxon(d_wall, alternative="greater", zero_method="wilcox"))
    p_wall = float(w.pvalue)
    signs = int(np.sum(d_wall > 0))

    md = args.root / "ANALYSIS.md"
    lines = [
        "# Maze cost-scaled H2 (supplementary wall-clock)",
        "",
        f"Matched `genetic` vs `genetic_filter` with injected `sim_cost_ms={args.sim_cost_ms}` "
        f"(above Phase-6 break-even ~3 ms; n=10).",
        "",
        f"- Mean wall genetic: **{np.mean(g_wall):.2f}s** ± {np.std(g_wall, ddof=1):.2f}",
        f"- Mean wall filter: **{np.mean(f_wall):.2f}s** ± {np.std(f_wall, ddof=1):.2f}",
        f"- Mean Δwall (genetic − filter): **{np.mean(d_wall):+.2f}s** "
        f"(sign filter-faster {signs}/10; Wilcoxon one-sided greater p={p_wall:.4g})",
        f"- Mean skip rate: **{np.nanmean(skips):.1f}%**",
        f"- Terminal coverage genetic / filter: "
        f"**{np.mean(g_cov):.1f}%** / **{np.mean(f_cov):.1f}%**",
        "",
        "Descriptive only — does not amend F-B5 Holm; fitness labels unchanged.",
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

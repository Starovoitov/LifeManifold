#!/usr/bin/env python3
"""Export matplotlib figures for the manuscript (Fig. 4–6)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "artifacts/manuscript/figures"


def _load_summary(path: Path, condition: str) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] != condition:
                continue
            seed = int(row["seed"])
            rows[seed] = {
                "coverage_pct": float(row["coverage_pct"]),
                "mean_best_fitness": float(row["mean_best_fitness"]),
            }
    return rows


def fig04_rq1_rq0(out: Path) -> None:
    stub = _load_summary(ROOT / "artifacts/experiments/q1-full/summary.csv", "stub")
    hints = _load_summary(ROOT / "artifacts/experiments/q1-full/summary.csv", "hints")
    seeds = sorted(set(stub) & set(hints))
    delta_cov = np.array([hints[s]["coverage_pct"] - stub[s]["coverage_pct"] for s in seeds])
    delta_fit = np.array(
        [hints[s]["mean_best_fitness"] - stub[s]["mean_best_fitness"] for s in seeds]
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].boxplot(
        [
            [stub[s]["coverage_pct"] for s in seeds],
            [hints[s]["coverage_pct"] for s in seeds],
        ],
        tick_labels=["stub", "hints"],
    )
    axes[0].set_ylabel("Coverage (%)")
    axes[0].set_title("F-RQ1 levels (n=10)")
    axes[0].grid(True, alpha=0.3)

    x = np.arange(len(seeds))
    axes[1].bar(x - 0.2, delta_cov, width=0.4, label="Δcov (pp)")
    axes[1].bar(x + 0.2, 100 * delta_fit, width=0.4, label="Δfit (×100)")
    axes[1].set_xticks(x, [str(s) for s in seeds])
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Paired hints − stub")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("RQ1: bundled stub vs hints (grid, paired seeds)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def fig05_ladder(out: Path) -> None:
    specs = [
        ("vanilla", ROOT / "artifacts/experiments/q1-v3-vanilla/summary.csv"),
        ("genetic_me", ROOT / "artifacts/experiments/q1-v3-genetic-me/summary.csv"),
        ("stub", ROOT / "artifacts/experiments/q1-full/summary.csv"),
        ("hints", ROOT / "artifacts/experiments/q1-full/summary.csv"),
        ("cma_me", ROOT / "artifacts/experiments/q1-v3-pyribs/summary.csv"),
        ("cma_mae", ROOT / "artifacts/experiments/q1-v3-pyribs/summary.csv"),
    ]
    labels: list[str] = []
    means: list[float] = []
    sds: list[float] = []
    for label, path in specs:
        cov = [row["coverage_pct"] for row in _load_summary(path, label).values()]
        labels.append(label)
        means.append(float(np.mean(cov)))
        sds.append(float(np.std(cov, ddof=1)) if len(cov) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    bar_colors = plt.get_cmap("tab10")(np.linspace(0, 0.7, len(labels)))
    ax.bar(x, means, yerr=sds, capsize=4, color=bar_colors)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Mean coverage (%)")
    ax.set_title("CA performance ladder (n=10 seeds)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _trace_curve(path: Path, metric: str, grid: np.ndarray) -> np.ndarray:
    by_eval: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get(metric) is not None:
            by_eval[int(row["evaluations"])] = float(row[metric])
    xs = np.array(sorted(by_eval))
    ys = np.array([by_eval[int(item)] for item in xs])
    y = np.interp(grid, xs, ys)
    if metric == "coverage":
        y = 100.0 * y
    return y


def fig06_acquisition(out: Path) -> None:
    grid = np.arange(0, 20_001, 500, dtype=float)
    arm_paths = [
        ("uniform", ROOT / "artifacts/experiments/q1-v3-genetic-me-uniform/genetic_me_uniform"),
        ("filter", ROOT / "artifacts/experiments/q1-v3-genetic-me-filter/genetic_me_filter"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"uniform": "#1f77b4", "filter": "#ff7f0e"}
    for label, arm_root in arm_paths:
        curves = []
        for seed in range(10):
            path = arm_root / f"seed_{seed}" / "archive_trace.jsonl"
            curves.append(_trace_curve(path, "coverage", grid))
        arr = np.vstack(curves)
        med = np.median(arr, axis=0)
        q25 = np.quantile(arr, 0.25, axis=0)
        q75 = np.quantile(arr, 0.75, axis=0)
        ax.plot(grid, med, label=label, color=colors[label], linewidth=2)
        ax.fill_between(grid, q25, q75, color=colors[label], alpha=0.2)
    ax.set_xlabel("Simulator evaluations")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Matched acquisition: genetic_me_uniform vs filter (n=10, IQR)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fig",
        type=int,
        choices=(4, 5, 6),
        action="append",
        dest="figs",
        help="Which figure(s) to export (repeatable)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export Fig. 4, 5, and 6",
    )
    args = parser.parse_args()
    figs = args.figs or ([4, 5, 6] if args.all else [])
    if not figs:
        parser.error("pass --fig N and/or --all")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    exporters = {
        4: (fig04_rq1_rq0, FIG_DIR / "fig04_rq1_rq0.pdf"),
        5: (fig05_ladder, FIG_DIR / "fig05_ladder.pdf"),
        6: (fig06_acquisition, FIG_DIR / "fig06_acquisition_anytime.pdf"),
    }
    for number in figs:
        fn, path = exporters[number]
        fn(path)


if __name__ == "__main__":
    main()

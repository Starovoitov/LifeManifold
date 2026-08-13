#!/usr/bin/env python3
"""Export manuscript figures (Fig. 1–8 + B4 + budget-axes schematic)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage
from matplotlib.patches import Patch

if TYPE_CHECKING:
    from worldspace.illuminators.archive import ArchiveElite

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MS_DIR = ROOT / "artifacts/manuscript"
FIG_DIR = MS_DIR / "figures"
SURROGATE_FLOW_MMD = FIG_DIR / "surrogate_flow.mmd"
PIPELINE_MMD = FIG_DIR / "pipeline.mmd"
PUPPETEER_CONFIG = FIG_DIR / "puppeteer-config.json"

# Locked after G1: same seed as Fig. 7 main panel and LLM mutation examples in text.
FIG02_DEFAULT_SEED = 4
FIG02_DEFAULT_CONDITION = "hints"


def _export_mermaid(mmd: Path, out: Path, *, width: int, height: int) -> None:
    """Export a mermaid diagram via mermaid-cli at print resolution."""
    if not mmd.is_file():
        raise FileNotFoundError(mmd)
    out.parent.mkdir(parents=True, exist_ok=True)
    base_cmd = [
        "npx",
        "--yes",
        "@mermaid-js/mermaid-cli@10.9.0",
        "-p",
        str(PUPPETEER_CONFIG),
        "-i",
        str(mmd),
        "-b",
        "white",
        "-w",
        str(width),
        "-H",
        str(height),
        "-s",
        "3",
    ]
    subprocess.run(
        [*base_cmd, "-o", str(out), "-e", "pdf", "-f"],
        check=True,
        cwd=FIG_DIR,
    )
    print(f"Wrote {out}")
    png = out.with_suffix(".png")
    subprocess.run([*base_cmd, "-o", str(png)], check=True, cwd=FIG_DIR)
    print(f"Wrote {png}")


def fig01_pipeline(out: Path) -> None:
    """Fig. 1: two surrogate integration roles (conceptual architecture)."""
    _export_mermaid(PIPELINE_MMD, out, width=1400, height=1800)


def fig03_surrogate_flow(out: Path) -> None:
    """Fig. 3: detailed surrogate feature → before-/after-generation flow."""
    _export_mermaid(SURROGATE_FLOW_MMD, out, width=1400, height=2400)


def _fig02_archive_path(*, seed: int, condition: str) -> Path:
    return (
        ROOT
        / f"artifacts/experiments/q1-full/{condition}/seed_{seed}/map_elites_archive.jsonl"
    )


def _fig02_pick_elites(
    archive_path: Path,
) -> tuple[
    tuple[str, ArchiveElite],
    tuple[str, ArchiveElite],
    tuple[str, ArchiveElite],
]:
    from worldspace.illuminators.archive import load_and_collapse_jsonl

    archive = load_and_collapse_jsonl(archive_path)
    elites: list[ArchiveElite] = [
        elite
        for cell_id in range(archive.n_cells)
        if (elite := archive.get_cell(cell_id)) is not None
    ]
    if len(elites) < 3:
        msg = f"need >=3 filled elites for Fig. 2, got {len(elites)}: {archive_path}"
        raise ValueError(msg)
    elites.sort(key=lambda elite: elite.fitness)
    # 10th percentile of filled cells by fitness (not the archive-min, not 10th-from-max).
    low_idx = max(1, len(elites) // 10)
    picks = (
        ("high", elites[-1]),
        ("median", elites[len(elites) // 2]),
        ("p10", elites[low_idx]),
    )
    for _label, elite in picks:
        if elite.world_spec is None:
            msg = f"elite missing world_spec in {archive_path}"
            raise ValueError(msg)
    return picks


def _fig02_life_food_rgb(life: np.ndarray, food: np.ndarray) -> np.ndarray:
    """Match ``worldspace.visualizer.diagnostics._life_food_rgb`` swatches."""
    life_f = life.astype(np.float32)
    food_f = food.astype(np.float32)
    rgb = np.zeros((*life.shape, 3), dtype=np.float32)
    rgb[(life_f < 0.5) & (food_f < 0.5)] = (0.10, 0.11, 0.15)
    rgb[(life_f >= 0.5) & (food_f < 0.5)] = (0.29, 0.61, 0.56)
    rgb[(life_f < 0.5) & (food_f >= 0.5)] = (0.79, 0.64, 0.15)
    rgb[(life_f >= 0.5) & (food_f >= 0.5)] = (0.43, 0.48, 0.31)
    return np.clip(rgb, 0.0, 1.0)


def _fig02_genome_hash(elite: ArchiveElite) -> str:
    from worldspace.illuminators.evaluation import _canonical_payload

    if elite.world_spec is None:
        return "—"
    digest = hashlib.sha256(
        _canonical_payload(elite.world_spec).encode("utf-8")
    ).hexdigest()
    return digest[:12]


def fig02_elite_worlds(
    out: Path,
    *,
    seed: int = FIG02_DEFAULT_SEED,
    condition: str = FIG02_DEFAULT_CONDITION,
) -> None:
    from worldspace import math as ws_math
    from worldspace.illuminators.evaluation import bin_center
    from worldspace.simulator import run_world

    archive_path = _fig02_archive_path(seed=seed, condition=condition)
    picks = _fig02_pick_elites(archive_path)
    tier_title = {
        "high": "High (max filled)",
        "median": "Median (50th of filled)",
        "p10": "Low (~10th percentile of filled)",
    }
    col_titles = (
        "Life + food · boundary\n(morphology; not a fitness term)",
        "2×2 heterogeneity\n(TH in T_topo, w=0.10)",
        "Live cells: food neighbours\n(ecology proxy; not a direct term)",
    )

    fig, axes = plt.subplots(3, 3, figsize=(11.2, 10.0), layout="constrained")
    meta_panels: list[dict[str, object]] = []
    imh = None
    for row, (label, elite) in enumerate(picks):
        if elite.world_spec is None:
            raise ValueError("elite missing world_spec")
        result = run_world(elite.world_spec)
        life, food = result.final_life, result.final_food
        if life is None or food is None:
            raise ValueError("run_world missing final_life/final_food")
        boundary = ws_math.topology_interface_strength_map(life)
        hetero = ws_math.topology_2x2_heterogeneity_map(life)
        fnb = ws_math.food_neighbor_fraction_map(food)
        i, j = elite.bin
        stab, div = bin_center(i, j, 50)
        ghash = _fig02_genome_hash(elite)
        meas = elite.measures or {}
        fit = float(elite.fitness)
        row_label = (
            f"{tier_title[label]}\n"
            f"f={fit:.3f}  bin ({i},{j})  "
            f"S={stab:.3f} D={div:.3f}\n"
            f"{condition} seed {seed}  hash={ghash}"
        )

        base = _fig02_life_food_rgb(life, food)
        warm = np.stack([boundary, boundary * 0.42, boundary * 0.12], axis=-1)
        blended = np.clip(base + warm * 0.42, 0.0, 1.0)
        axes[row, 0].imshow(blended, origin="lower", interpolation="nearest")
        axes[row, 0].set_ylabel(row_label, fontsize=8)
        imh = axes[row, 1].imshow(
            hetero,
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        live = life > 0.5
        t = fnb.astype(np.float32)
        warm_adj = np.clip(
            np.stack(
                [0.18 + 0.75 * t, 0.32 + 0.38 * (1.0 - t), 0.48 + 0.35 * (1.0 - t)],
                axis=-1,
            ),
            0.0,
            1.0,
        )
        blend_adj = np.where(
            live[..., None], (1.0 - 0.48) * base + 0.48 * warm_adj, base
        )
        axes[row, 2].imshow(blend_adj, origin="lower", interpolation="nearest")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        if row == 0:
            for ax, title in zip(axes[0], col_titles, strict=True):
                ax.set_title(title, fontsize=9)
        meta_panels.append(
            {
                "tier": label,
                "fitness": fit,
                "bin": [int(i), int(j)],
                "stability": float(meas.get("stability", stab)),
                "diversity": float(meas.get("diversity", div)),
                "genome_sha256_12": ghash,
            }
        )
    if imh is not None:
        cbar = fig.colorbar(
            imh, ax=axes[:, 1], fraction=0.046, pad=0.04, ticks=[0.0, 1.0]
        )
        cbar.set_label("TH")

    legend = [
        Patch(facecolor=(0.10, 0.11, 0.15), edgecolor="0.3", label="Empty"),
        Patch(facecolor=(0.29, 0.61, 0.56), edgecolor="0.3", label="Life only"),
        Patch(facecolor=(0.79, 0.64, 0.15), edgecolor="0.3", label="Food only"),
        Patch(facecolor=(0.43, 0.48, 0.31), edgecolor="0.3", label="Life + food"),
        Patch(facecolor=(1.0, 0.42, 0.12), edgecolor="0.3", label="Boundary tint (TI)"),
        Patch(
            facecolor=plt.get_cmap("magma")(0.0),
            edgecolor="0.3",
            label="2×2 uniform (0)",
        ),
        Patch(
            facecolor=plt.get_cmap("magma")(1.0), edgecolor="0.3", label="2×2 mixed (1)"
        ),
    ]
    fig.legend(
        handles=legend,
        loc="outside lower center",
        ncol=4,
        fontsize=8,
        frameon=True,
    )
    fig.suptitle(
        f"Outcome morphology ({condition}, MAP-Elites seed {seed}): "
        "max / 50th / ~10th-percentile filled-cell fitness",
        fontsize=11,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    meta = {
        "seed": seed,
        "condition": condition,
        "archive": str(archive_path.relative_to(ROOT)),
        "selection": {
            "high": "max fitness among filled cells",
            "median": "median fitness among filled cells",
            "p10": "≈10th percentile of filled cells (index max(1, n_filled//10))",
        },
        "panels": meta_panels,
        "note": "2×2 heterogeneity is a binary mixed-corners indicator (TH), not a continuous field.",
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Wrote {meta_path}")


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
    stub_cov = np.array([stub[s]["coverage_pct"] for s in seeds], dtype=float)
    hints_cov = np.array([hints[s]["coverage_pct"] for s in seeds], dtype=float)
    delta_cov = hints_cov - stub_cov
    delta_fit = np.array(
        [hints[s]["mean_best_fitness"] - stub[s]["mean_best_fitness"] for s in seeds],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    # Paired slope / dot plot of matched seeds (preferable to summary bars at n=10).
    x_pair = np.array([0.0, 1.0])
    for s, y0, y1 in zip(seeds, stub_cov, hints_cov):
        axes[0].plot(
            x_pair,
            [y0, y1],
            color="0.65",
            linewidth=1.0,
            alpha=0.85,
            zorder=1,
        )
        axes[0].scatter(
            [0.0, 1.0],
            [y0, y1],
            color=["#0072B2", "#D55E00"],
            s=36,
            zorder=2,
            edgecolors="0.2",
            linewidths=0.4,
        )
        axes[0].annotate(
            str(s),
            (1.0, y1),
            textcoords="offset points",
            xytext=(5, 0),
            fontsize=7,
            color="0.25",
        )
    axes[0].set_xticks([0.0, 1.0], ["stub", "hints"])
    axes[0].set_xlim(-0.25, 1.35)
    axes[0].set_ylabel("Coverage (%)")
    axes[0].set_title("F-RQ1 paired levels (n=10)")
    axes[0].grid(True, axis="y", alpha=0.3)

    x = np.arange(len(seeds))
    axes[1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1].scatter(
        x - 0.08,
        delta_cov,
        s=42,
        label="Δcov (pp)",
        color="#0072B2",
        zorder=2,
        edgecolors="0.2",
        linewidths=0.4,
    )
    axes[1].scatter(
        x + 0.08,
        100 * delta_fit,
        s=42,
        marker="s",
        label="Δfit (×100)",
        color="#D55E00",
        zorder=2,
        edgecolors="0.2",
        linewidths=0.4,
    )
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
    # Terminal coverage ladder: policy jump, after-generation filter, LLM stack, pyribs refs.
    specs = [
        ("vanilla", ROOT / "artifacts/experiments/q1-v3-vanilla/summary.csv"),
        ("genetic_me", ROOT / "artifacts/experiments/q1-v3-genetic-me/summary.csv"),
        ("stub", ROOT / "artifacts/experiments/q1-full/summary.csv"),
        (
            "genetic_me_uniform",
            ROOT / "artifacts/experiments/q1-v3-genetic-me-uniform/summary.csv",
        ),
        (
            "genetic_me_filter",
            ROOT / "artifacts/experiments/q1-v3-genetic-me-filter/summary.csv",
        ),
        ("filter", ROOT / "artifacts/experiments/q1-full/summary.csv"),
        ("hints", ROOT / "artifacts/experiments/q1-full/summary.csv"),
        ("cma_me", ROOT / "artifacts/experiments/q1-v3-pyribs/summary.csv"),
        ("cma_mae", ROOT / "artifacts/experiments/q1-v3-pyribs/summary.csv"),
    ]
    from scipy import stats as _stats

    labels: list[str] = []
    means: list[float] = []
    cis: list[float] = []
    ns: list[int] = []
    for label, path in specs:
        cov = [row["coverage_pct"] for row in _load_summary(path, label).values()]
        n = len(cov)
        mean = float(np.mean(cov))
        sd = float(np.std(cov, ddof=1)) if n > 1 else 0.0
        # Student-t 95% CI half-width on the mean (same recipe for every bar).
        ci = float(_stats.t.ppf(0.975, n - 1)) * sd / np.sqrt(n) if n > 1 else 0.0
        labels.append(label)
        means.append(mean)
        cis.append(ci)
        ns.append(n)

    if len(set(ns)) != 1:
        raise RuntimeError(
            f"fig05_ladder: inconsistent n across arms: {dict(zip(labels, ns))}"
        )

    # Tick labels: mark H2/H3 after-generation arms; bars remain terminal coverage.
    tick_labels = {
        "genetic_me_filter": "me_filter (H2)",
        "filter": "LLM+filter (H3)",
    }
    display = [tick_labels.get(lab, lab) for lab in labels]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(labels))
    bar_colors = plt.get_cmap("tab10")(np.linspace(0, 0.7, len(labels)))
    bars = ax.bar(
        x,
        means,
        yerr=cis,
        capsize=3,
        color=bar_colors,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    # Hatch after-generation arms so they are not read as a simple ranking pair.
    for lab, bar in zip(labels, bars):
        if lab == "genetic_me_filter":
            bar.set_hatch("//")
            bar.set_edgecolor("0.2")
        elif lab == "filter":
            bar.set_hatch("\\\\")
            bar.set_edgecolor("0.2")
    ax.set_xticks(x, display, rotation=25, ha="right")
    ax.set_ylabel("Mean terminal coverage (%)")
    ax.set_title(
        f"Primary-grid coverage ladder (terminal @ fixed iterations; "
        f"mean ± Student-t 95% CI; n={ns[0]})"
    )
    ax.text(
        0.02,
        0.98,
        "H2 claim = eval-indexed (per sim), not bar height\n"
        "H3 = LLM+filter stack (descriptive / not confirmatory)",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "0.6",
            "alpha": 0.92,
        },
    )
    # Callout on the H2 terminal bar: claim is per-sim efficiency, not this height.
    h2_idx = labels.index("genetic_me_filter")
    ax.annotate(
        "H2: read per-sim\ncurves, not this bar",
        xy=(h2_idx, means[h2_idx]),
        xytext=(h2_idx + 1.35, means[h2_idx] + 6.5),
        textcoords="data",
        fontsize=7.5,
        ha="left",
        va="bottom",
        color="0.15",
        arrowprops={
            "arrowstyle": "->",
            "color": "0.25",
            "lw": 1.0,
            "connectionstyle": "arc3,rad=0.12",
        },
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.45",
            "alpha": 0.95,
        },
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    for label, mean, ci, n in zip(labels, means, cis, ns):
        print(f"  {label}: {mean:.2f} ± {ci:.2f} (t 95% CI half-width, n={n})")


def _trace_xy(path: Path, metric: str) -> tuple[np.ndarray, np.ndarray]:
    by_eval: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get(metric) is not None:
            by_eval[int(row["evaluations"])] = float(row[metric])
    xs = np.asarray(sorted(by_eval), dtype=float)
    ys = np.asarray([by_eval[int(item)] for item in xs], dtype=float)
    if metric == "coverage":
        ys = 100.0 * ys
    return xs, ys


def _trace_curve(path: Path, metric: str, grid: np.ndarray) -> np.ndarray:
    xs, ys = _trace_xy(path, metric)
    return np.interp(grid, xs, ys)


def _median_iqr_supported(
    arm_root: Path,
    metric: str,
    seeds: range | list[int],
    *,
    step: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median/IQR on a grid truncated to the arm's common real-eval support.

    No last-observation-carried-forward past each seed's final logged evaluation:
    the shared grid ends at ``min_seed(last_eval)`` so every seed still has data.
    """
    series: list[tuple[np.ndarray, np.ndarray]] = []
    lasts: list[float] = []
    for seed in seeds:
        path = arm_root / f"seed_{seed}" / "archive_trace.jsonl"
        xs, ys = _trace_xy(path, metric)
        if xs.size == 0:
            raise FileNotFoundError(f"empty trace metric={metric}: {path}")
        series.append((xs, ys))
        lasts.append(float(xs[-1]))
    support = float(min(lasts))
    grid = np.arange(0.0, support + 1.0, float(step))
    if grid.size == 0 or grid[-1] != support:
        grid = np.append(grid, support)
    curves = np.vstack([np.interp(grid, xs, ys) for xs, ys in series])
    med = np.median(curves, axis=0)
    q25 = np.quantile(curves, 0.25, axis=0)
    q75 = np.quantile(curves, 0.75, axis=0)
    return grid, med, q25, q75


def _dungeon_terminal_iqr(
    summary_path: Path, condition: str, metric: str, seeds: range | list[int]
) -> tuple[float, float, float]:
    """Median / IQR of terminal summary rows at the fixed proposal budget."""
    wanted = {int(s) for s in seeds}
    xs: list[float] = []
    with summary_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("condition") != condition:
                continue
            if int(row["seed"]) not in wanted:
                continue
            key = "coverage_pct" if metric == "coverage" else metric
            val = float(row[key])
            if metric == "coverage" and val <= 1.5:
                val *= 100.0
            xs.append(val)
    if not xs:
        msg = f"no terminal rows for {condition}/{metric} in {summary_path}"
        raise KeyError(msg)
    arr = np.asarray(xs, dtype=float)
    return (
        float(np.median(arr)),
        float(np.quantile(arr, 0.25)),
        float(np.quantile(arr, 0.75)),
    )


def fig_b4_dungeon_anytime(out_dir: Path | None = None) -> None:
    """Dungeon anytime (cropped at AUC horizon) + fixed-proposal terminal panel."""
    root = ROOT / "artifacts/experiments/q1-v4-dungeon-rerun"
    out_dir = out_dir or FIG_DIR
    summary_path = root / "summary.csv"
    # Okabe–Ito-ish colorblind-safe palette + distinct linestyles/markers.
    arms = (
        ("genetic", "#009E73", "-", "o"),
        ("genetic_filter", "#56B4E9", "--", "s"),
        ("llm_stub", "#E69F00", "-.", "^"),
        ("llm_hints", "#CC79A7", ":", "D"),
        ("llm_hints_filter", "#0072B2", (0, (3, 1, 1, 1)), "v"),
    )
    stats_path = root / "v4_dungeon_statistics.json"
    auc_horizon = int(
        json.loads(stats_path.read_text(encoding="utf-8"))["common_evaluation_budget"]
    )
    seeds = range(10)
    for metric, ylabel, stem in (
        ("coverage", "Coverage (%)", "fig_b4_anytime_coverage"),
        ("qd_score", "QD-score", "fig_b4_anytime_qd_score"),
    ):
        fig, (ax_inf, ax_term) = plt.subplots(
            1, 2, figsize=(10.6, 4.4), gridspec_kw={"width_ratios": [1.55, 1.0]}
        )
        for label, color, ls, marker in arms:
            grid, med, q25, q75 = _median_iqr_supported(
                root / label, metric, seeds, step=50
            )
            keep = grid <= float(auc_horizon) + 1e-9
            g, m, lo, hi = grid[keep], med[keep], q25[keep], q75[keep]
            markevery = max(1, len(g) // 8)
            ax_inf.plot(
                g,
                m,
                label=label,
                color=color,
                linewidth=2,
                linestyle=ls,
                marker=marker,
                markevery=markevery,
                markersize=5,
            )
            ax_inf.fill_between(g, lo, hi, color=color, alpha=0.14)
        ax_inf.axvline(
            auc_horizon,
            color="0.35",
            linestyle="--",
            linewidth=1.2,
        )
        ax_inf.set_xlabel("Real evaluations")
        ax_inf.set_ylabel(ylabel)
        ax_inf.set_xlim(0, auc_horizon)
        ax_inf.set_title(f"Primary inference (crop @ {auc_horizon:,})")
        ax_inf.grid(True, alpha=0.3)
        ax_inf.legend(fontsize=7, loc="lower right")

        x = np.arange(len(arms))
        for idx, (label, color, _ls, marker) in enumerate(arms):
            med, lo, hi = _dungeon_terminal_iqr(summary_path, label, metric, seeds)
            ax_term.errorbar(
                [idx],
                [med],
                yerr=[[med - lo], [hi - med]],
                fmt=marker,
                color=color,
                ecolor=color,
                capsize=3,
                markersize=8,
            )
            ax_term.scatter([idx], [med], color=color, s=36, zorder=3, marker=marker)
        ax_term.set_xticks(x, [a[0] for a in arms], rotation=25, ha="right")
        ax_term.set_ylabel(ylabel)
        ax_term.set_title("Fixed-proposal terminal (5k slots)")
        ax_term.grid(True, axis="y", alpha=0.3)
        ax_term.set_xlim(-0.5, len(arms) - 0.5)

        title_metric = "Coverage (%)" if metric == "coverage" else "QD-score"
        fig.suptitle(
            f"Dungeon — median {title_metric} (n=10; IQR; no LOCF). "
            "Filter arms stop early on the eval axis because skips cut sims, not because runs are incomplete.",
            fontsize=10,
        )
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / f"{stem}.png"
        pdf = out_dir / f"{stem}.pdf"
        fig.savefig(png, dpi=200, bbox_inches="tight")
        fig.savefig(pdf, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {png}")
        print(f"Wrote {pdf}")


def fig08_anytime_ladder(out: Path) -> None:
    """Coverage vs evaluations: vanilla / hints / cma_me (q1-anytime-ladder, n=5)."""
    grid = np.arange(0, 32_501, 500, dtype=float)
    arm_paths = [
        ("vanilla", ROOT / "artifacts/experiments/q1-anytime-ladder/vanilla"),
        ("hints", ROOT / "artifacts/experiments/q1-anytime-ladder/hints"),
        ("cma_me", ROOT / "artifacts/experiments/q1-anytime-ladder/cma_me"),
    ]
    colors = {"vanilla": "#9467bd", "hints": "#2ca02c", "cma_me": "#d62728"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, arm_root in arm_paths:
        curves = []
        for seed in range(5):
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
    ax.set_title("Anytime ladder: vanilla vs hints vs cma_me (n=5, IQR)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def fig06_acquisition(out: Path) -> None:
    grid = np.arange(0, 20_001, 500, dtype=float)
    arm_paths = [
        (
            "uniform",
            ROOT / "artifacts/experiments/q1-v3-genetic-me-uniform/genetic_me_uniform",
        ),
        (
            "filter",
            ROOT / "artifacts/experiments/q1-v3-genetic-me-filter/genetic_me_filter",
        ),
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
    ax.set_title(
        "H2 (after-generation): genetic_me_uniform vs genetic_me_filter (n=10, IQR)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _coverage_pct(summary_path: Path, condition: str, seed: int) -> float:
    with summary_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] == condition and int(row["seed"]) == seed:
                return float(row["coverage_pct"])
    msg = f"no {condition} row for seed {seed} in {summary_path}"
    raise KeyError(msg)


def _load_fitness_pivot(archive_path: Path) -> np.ndarray:
    import sys

    repo = ROOT
    dash = repo / "dashboard"
    for entry in (str(repo), str(dash)):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    from dashboard.components.archive_loader import load_archive_bundle
    from dashboard.utils.config import load_config

    cfg = load_config()
    bundle = load_archive_bundle(archive_path, archive_path.stat().st_mtime, cfg)
    if bundle.archive_type != "grid":
        msg = f"expected grid archive, got {bundle.archive_type}: {archive_path}"
        raise ValueError(msg)
    return bundle.pivots["fitness"]


# §3.11 picks by |hints−cma_me| Δcov: largest (4), mid (6), smallest (1).
HEATMAP_PROTOCOL_SEEDS = (4, 6, 1)


def _archive_pair_paths(
    seed: int, *, left: str = "hints", right: str = "cma_me"
) -> tuple[Path, Path, Path, Path, str, str]:
    """Return (left_archive, right_archive, left_summary, right_summary, left_cond, right_cond)."""
    roots = {
        "hints": (
            ROOT
            / f"artifacts/experiments/q1-full/hints/seed_{seed}/map_elites_archive.jsonl",
            ROOT / "artifacts/experiments/q1-full/summary.csv",
            "hints",
        ),
        "filter": (
            ROOT
            / f"artifacts/experiments/q1-full/filter/seed_{seed}/map_elites_archive.jsonl",
            ROOT / "artifacts/experiments/q1-full/summary.csv",
            "filter",
        ),
        "cma_me": (
            ROOT
            / f"artifacts/experiments/q1-v3-pyribs/cma_me/seed_{seed}/map_elites_archive.jsonl",
            ROOT / "artifacts/experiments/q1-v3-pyribs/summary.csv",
            "cma_me",
        ),
    }
    if left not in roots or right not in roots:
        msg = f"unknown arm in pair ({left}, {right}); expected one of {sorted(roots)}"
        raise KeyError(msg)
    la, ls, lc = roots[left]
    ra, rs, rc = roots[right]
    return la, ra, ls, rs, lc, rc


def _draw_heatmap_row(
    axes: np.ndarray,
    *,
    seed: int,
    left: str,
    right: str,
    cmap,
) -> AxesImage:
    la, ra, ls, rs, lc, rc = _archive_pair_paths(seed, left=left, right=right)
    for path in (la, ra):
        if not path.is_file():
            raise FileNotFoundError(path)
    panels = [
        (lc, la, _coverage_pct(ls, lc, seed)),
        (rc, ra, _coverage_pct(rs, rc, seed)),
    ]
    grids = [_load_fitness_pivot(path) for _, path, _ in panels]
    im: AxesImage | None = None
    for ax, (label, _, cov), grid in zip(axes, panels, grids, strict=True):
        masked = np.ma.masked_invalid(grid)
        im = ax.imshow(
            masked,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            aspect="equal",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(f"{label} (seed {seed}, coverage {cov:.1f}%)", fontsize=10)
        ax.set_xlabel("Diversity →")
        ax.set_ylabel("Stability →")
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
    if im is None:
        raise RuntimeError(f"no heatmap panels drawn for seed {seed}")
    return im


def fig07_archive_heatmaps(
    out: Path,
    *,
    seed: int = 4,
    left: str = "hints",
    right: str = "cma_me",
) -> None:
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dddddd")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.8), layout="constrained")
    im = _draw_heatmap_row(axes, seed=seed, left=left, right=right, cmap=cmap)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Elite fitness (unitless)")
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fig.suptitle(
        "Archive fitness in behaviour space (collapsed warm-start; gray = empty, not infeasible)",
        fontsize=12,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    png_out = out.with_suffix(".png")
    fig.savefig(png_out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"Wrote {png_out}")

    alias_dir = FIG_DIR / "archive_heatmaps"
    alias_dir.mkdir(parents=True, exist_ok=True)
    alias = alias_dir / f"seed{seed}_{left}_vs_{right}.png"
    alias.write_bytes(png_out.read_bytes())
    print(f"Wrote {alias}")


def fig07_archive_heatmaps_panel(
    out: Path,
    *,
    seeds: tuple[int, ...] = HEATMAP_PROTOCOL_SEEDS,
    left: str = "hints",
    right: str = "cma_me",
) -> None:
    """Multi-seed side-by-side panel for protocol §3.11 (|Δcov| large / mid / small)."""
    if not seeds:
        raise ValueError("seeds must be non-empty")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dddddd")
    n = len(seeds)
    fig, axes = plt.subplots(n, 2, figsize=(9.8, 3.9 * n), layout="constrained")
    if n == 1:
        axes = np.asarray([axes])
    im: AxesImage | None = None
    for row, seed in enumerate(seeds):
        im = _draw_heatmap_row(axes[row], seed=seed, left=left, right=right, cmap=cmap)
    assert im is not None
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.55)
    cbar.set_label("Elite fitness (unitless)")
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fig.suptitle(
        "Archive fitness across paired seeds (collapsed warm-start; gray = empty, not infeasible)",
        fontsize=12,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    png_out = out.with_suffix(".png")
    fig.savefig(png_out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"Wrote {png_out}")
    alias_dir = FIG_DIR / "archive_heatmaps"
    alias_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = "_".join(str(s) for s in seeds)
    alias = alias_dir / f"panel_seeds{seed_tag}_{left}_vs_{right}.png"
    alias.write_bytes(png_out.read_bytes())
    print(f"Wrote {alias}")


def _fitness_grid_for_arm(arm: str, seed: int) -> np.ndarray:
    la, ra, _ls, _rs, _lc, _rc = _archive_pair_paths(seed, left="hints", right="cma_me")
    path = la if arm == "hints" else ra
    return _load_fitness_pivot(path)


def fig07_occupancy_n10(out: Path) -> None:
    """n=10 occupancy probability, E[f|occupied], and occupancy-difference maps."""
    seeds = range(10)
    hints = np.stack([_fitness_grid_for_arm("hints", s) for s in seeds], axis=0)
    cma = np.stack([_fitness_grid_for_arm("cma_me", s) for s in seeds], axis=0)
    p_h = np.mean(np.isfinite(hints), axis=0)
    p_c = np.mean(np.isfinite(cma), axis=0)
    d_p = p_c - p_h
    n_h = np.sum(np.isfinite(hints), axis=0)
    n_c = np.sum(np.isfinite(cma), axis=0)
    e_h = np.divide(
        np.nansum(hints, axis=0),
        n_h,
        out=np.full(n_h.shape, np.nan, dtype=float),
        where=n_h > 0,
    )
    e_c = np.divide(
        np.nansum(cma, axis=0),
        n_c,
        out=np.full(n_c.shape, np.nan, dtype=float),
        where=n_c > 0,
    )

    cmap_fit = plt.get_cmap("viridis").copy()
    cmap_fit.set_bad("#dddddd")
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.4), layout="constrained")
    occ_kw = {
        "origin": "lower",
        "extent": (0.0, 1.0, 0.0, 1.0),
        "aspect": "equal",
        "interpolation": "nearest",
        "vmin": 0.0,
        "vmax": 1.0,
    }
    im_h = axes[0, 0].imshow(p_h, cmap="Blues", **occ_kw)
    axes[0, 0].set_title(r"$P$(occupied)  hints")
    im_c = axes[0, 1].imshow(p_c, cmap="Blues", **occ_kw)
    axes[0, 1].set_title(r"$P$(occupied)  cma_me")
    v = float(np.nanmax(np.abs(d_p))) or 1.0
    im_d = axes[0, 2].imshow(
        d_p,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        aspect="equal",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-v,
        vmax=v,
    )
    axes[0, 2].set_title(r"$\Delta P$  (cma_me $-$ hints)")

    im_eh = axes[1, 0].imshow(
        np.ma.masked_invalid(e_h),
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        aspect="equal",
        interpolation="nearest",
        cmap=cmap_fit,
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 0].set_title(r"$\mathbb{E}[f\mid\mathrm{occ}]$  hints")
    im_ec = axes[1, 1].imshow(
        np.ma.masked_invalid(e_c),
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        aspect="equal",
        interpolation="nearest",
        cmap=cmap_fit,
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 1].set_title(r"$\mathbb{E}[f\mid\mathrm{occ}]$  cma_me")
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.55,
        "n=10 paired seeds\n"
        "gray = never occupied\n"
        "across those seeds\n\n"
        r"$\Delta P>0$: CMA fills"
        "\nmore often than hints",
        transform=axes[1, 2].transAxes,
        fontsize=9,
        va="center",
    )

    for ax in (*axes[0], axes[1, 0], axes[1, 1]):
        ax.set_xlabel("Diversity →")
        ax.set_ylabel("Stability →")
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])

    fig.colorbar(im_h, ax=axes[0, 0], fraction=0.046, pad=0.04, ticks=[0, 0.5, 1])
    fig.colorbar(im_c, ax=axes[0, 1], fraction=0.046, pad=0.04, ticks=[0, 0.5, 1])
    fig.colorbar(im_d, ax=axes[0, 2], fraction=0.046, pad=0.04)
    fig.colorbar(im_eh, ax=axes[1, 0], fraction=0.046, pad=0.04, ticks=[0, 0.5, 1])
    fig.colorbar(im_ec, ax=axes[1, 1], fraction=0.046, pad=0.04, ticks=[0, 0.5, 1])
    fig.suptitle(
        "Archive occupancy across seeds 0–9 (hints vs cma_me; collapsed warm-start)",
        fontsize=12,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    png_out = out.with_suffix(".png")
    fig.savefig(png_out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"Wrote {png_out}")


def fig_budget_axes(out: Path) -> None:
    """Conceptual three-budget schematic (Methods; not a results ranking)."""
    from matplotlib.patches import FancyBboxPatch

    # Okabe–Ito: bluish = better on that axis, vermillion = worse, gray = axis.
    better = "#0072B2"
    worse = "#D55E00"
    ink = "#222222"
    panels = (
        (
            "Proposal slots",
            "matched ask count",
            "filter worse",
            "−0.83 pp terminal\n@ 32,500 slots",
            worse,
        ),
        (
            "Real simulator\nevaluations",
            "matched run_world calls",
            "filter better",
            "+3.65 pp coverage\n@ 20,000 evals",
            better,
        ),
        (
            "Wall / LLM calls",
            "clock time, not invoices",
            "filter slower",
            "~20 vs ~7 min\nsame skip gate",
            worse,
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.85))
    for ax, (title, sub, verdict, detail, color) in zip(axes, panels):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.04, 0.06),
                0.92,
                0.88,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                linewidth=1.6,
                edgecolor=color,
                facecolor="#FAFAFA",
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.text(
            0.5,
            0.82,
            title,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=ink,
            transform=ax.transAxes,
            linespacing=1.15,
        )
        ax.text(
            0.5,
            0.58,
            sub,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#555555",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.38,
            verdict,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=color,
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.16,
            detail,
            ha="center",
            va="center",
            fontsize=8.5,
            color=ink,
            transform=ax.transAxes,
            linespacing=1.25,
        )
    fig.subplots_adjust(wspace=0.08, left=0.01, right=0.99, top=0.98, bottom=0.02)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(
        out.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fig",
        type=int,
        choices=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        action="append",
        dest="figs",
        help="Which figure(s) to export (repeatable; 9 = appendix B4; 10 = budget axes)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export Fig. 1–8 (not appendix B4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        dest="seeds",
        help=(
            "Paired seed(s) for Fig. 7 (repeatable). "
            f"Default protocol set: {','.join(map(str, HEATMAP_PROTOCOL_SEEDS))}"
        ),
    )
    parser.add_argument(
        "--pair",
        type=str,
        default="hints,cma_me",
        help="Comma pair of arms for Fig. 7 (default: hints,cma_me; also filter,cma_me)",
    )
    parser.add_argument(
        "--panel",
        action="store_true",
        help="Also write multi-seed panel PDF/PNG for all requested Fig. 7 seeds",
    )
    parser.add_argument(
        "--fig2-seed",
        type=int,
        default=FIG02_DEFAULT_SEED,
        help=f"Seed for Fig. 2 elite triptychs (default: {FIG02_DEFAULT_SEED})",
    )
    args = parser.parse_args()
    figs = args.figs or ([1, 2, 3, 4, 5, 6, 7, 8] if args.all else [])
    if not figs:
        parser.error("pass --fig N and/or --all")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    exporters = {
        1: (fig01_pipeline, FIG_DIR / "fig01_pipeline.pdf"),
        2: (fig02_elite_worlds, FIG_DIR / "fig02_elite_worlds.png"),
        3: (fig03_surrogate_flow, FIG_DIR / "fig03_surrogate_flow.pdf"),
        4: (fig04_rq1_rq0, FIG_DIR / "fig04_rq1_rq0.pdf"),
        5: (fig05_ladder, FIG_DIR / "fig05_ladder.pdf"),
        6: (fig06_acquisition, FIG_DIR / "fig06_acquisition_anytime.pdf"),
        7: (fig07_archive_heatmaps, FIG_DIR / "fig07_archive_heatmaps_seed4.pdf"),
        8: (fig08_anytime_ladder, FIG_DIR / "fig08_anytime_ladder.pdf"),
        9: (fig_b4_dungeon_anytime, FIG_DIR),
        10: (fig_budget_axes, FIG_DIR / "fig_budget_axes.pdf"),
    }
    left, right = (p.strip() for p in args.pair.split(",", 1))
    seeds: tuple[int, ...] = (
        tuple(int(s) for s in args.seeds) if args.seeds else HEATMAP_PROTOCOL_SEEDS
    )
    for number in figs:
        fn, path = exporters[number]
        if number == 7:
            for seed in seeds:
                if (left, right) == ("hints", "cma_me"):
                    out_path = FIG_DIR / f"fig07_archive_heatmaps_seed{seed}.pdf"
                else:
                    out_path = (
                        FIG_DIR
                        / f"fig07_archive_heatmaps_seed{seed}_{left}_vs_{right}.pdf"
                    )
                fn(out_path, seed=seed, left=left, right=right)
            if args.panel or (len(seeds) > 1 and (left, right) == ("hints", "cma_me")):
                tag = "_".join(str(s) for s in seeds)
                if (left, right) == ("hints", "cma_me"):
                    panel_out = FIG_DIR / f"fig07_archive_heatmaps_panel_seeds{tag}.pdf"
                else:
                    panel_out = (
                        FIG_DIR
                        / f"fig07_archive_heatmaps_panel_seeds{tag}_{left}_vs_{right}.pdf"
                    )
                fig07_archive_heatmaps_panel(
                    panel_out, seeds=seeds, left=left, right=right
                )
            if (left, right) == ("hints", "cma_me"):
                fig07_occupancy_n10(FIG_DIR / "fig07_occupancy_n10.pdf")
        else:
            if number == 2:
                fig02_elite_worlds(path, seed=args.fig2_seed)
            elif number == 9:
                fig_b4_dungeon_anytime(path)
            else:
                fn(path)


if __name__ == "__main__":
    main()

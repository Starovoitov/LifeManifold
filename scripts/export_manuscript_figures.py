#!/usr/bin/env python3
"""Export manuscript figures (Fig. 1–7; Fig. 1/3 via mermaid-cli)."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage

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
    low_idx = max(1, len(elites) // 10)
    picks = (
        ("high", elites[-1]),
        ("median", elites[len(elites) // 2]),
        ("low", elites[low_idx]),
    )
    for _label, elite in picks:
        if elite.world_spec is None:
            msg = f"elite missing world_spec in {archive_path}"
            raise ValueError(msg)
    return picks


def _fig02_render_triptych(elite: ArchiveElite, tmp_path: Path) -> np.ndarray:
    from PIL import Image

    from worldspace.simulator import run_world
    from worldspace.visualizer.diagnostics import plot_elite_triptych

    if elite.world_spec is None:
        msg = "elite missing world_spec"
        raise ValueError(msg)
    result = run_world(elite.world_spec)
    plot_elite_triptych(result, tmp_path, dpi=120)
    return np.array(Image.open(tmp_path))


def fig02_elite_worlds(
    out: Path,
    *,
    seed: int = FIG02_DEFAULT_SEED,
    condition: str = FIG02_DEFAULT_CONDITION,
) -> None:
    archive_path = _fig02_archive_path(seed=seed, condition=condition)
    picks = _fig02_pick_elites(archive_path)
    tmp_dir = FIG_DIR / "_fig02_cache"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    panel_rows: list[tuple[str, np.ndarray]] = []
    for label, elite in picks:
        fit = float(elite.fitness)
        tmp_path = tmp_dir / f"seed{seed}_{label}.png"
        arr = _fig02_render_triptych(elite, tmp_path)
        panel_rows.append((f"{label.capitalize()} fitness ({fit:.2f})", arr))

    max_h = max(arr.shape[0] for _, arr in panel_rows)
    padded: list[tuple[str, np.ndarray]] = []
    for title, arr in panel_rows:
        if arr.shape[0] < max_h:
            pad = max_h - arr.shape[0]
            arr = np.pad(arr, ((0, pad), (0, 0), (0, 0)), mode="constant")
        padded.append((title, arr))

    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2))
    for ax, (title, arr) in zip(axes, padded, strict=True):
        ax.imshow(arr)
        ax.set_title(title, loc="left", fontsize=11, pad=6)
        ax.axis("off")

    fig.tight_layout(h_pad=0.25)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    meta = {
        "seed": seed,
        "condition": condition,
        "archive": str(archive_path.relative_to(ROOT)),
        "panels": [
            {"tier": label, "fitness": float(elite.fitness), "bin": elite.bin}
            for label, elite in picks
        ],
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
    delta_cov = np.array(
        [hints[s]["coverage_pct"] - stub[s]["coverage_pct"] for s in seeds]
    )
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
    labels: list[str] = []
    means: list[float] = []
    cis: list[float] = []
    ns: list[int] = []
    for label, path in specs:
        cov = [row["coverage_pct"] for row in _load_summary(path, label).values()]
        n = len(cov)
        mean = float(np.mean(cov))
        sd = float(np.std(cov, ddof=1)) if n > 1 else 0.0
        # Half-width of normal approx. 95% CI on the mean (same for every bar).
        ci = 1.96 * sd / np.sqrt(n) if n > 1 else 0.0
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
        "genetic_me_filter": "genetic_me_filter\n(H2, terminal)",
        "filter": "LLM+filter\n(H3, terminal)",
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
        f"mean ± 95% CI; n={ns[0]})"
    )
    ax.text(
        0.02,
        0.98,
        "H2 claim = eval-indexed (per sim), not bar height\n"
        "H3 = LLM+filter stack (descriptive / blocked prod.)",
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
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    for label, mean, ci, n in zip(labels, means, cis, ns):
        print(f"  {label}: {mean:.2f} ± {ci:.2f} (95% CI half-width, n={n})")


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


# §3.11 picks: smallest gap (1), largest gap (4), mid-large (6).
HEATMAP_PROTOCOL_SEEDS = (1, 4, 6)


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
            aspect="equal",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(f"{label} (seed {seed}, coverage {cov:.1f}%)", fontsize=10)
        ax.set_xlabel("Diversity bin")
        ax.set_ylabel("Stability bin")
        ax.set_xlim(-0.5, grid.shape[1] - 0.5)
        ax.set_ylim(-0.5, grid.shape[0] - 0.5)
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
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="Fitness")
    fig.suptitle(
        "Archive fitness in behaviour space (collapsed warm-start archives)",
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
    """Multi-seed side-by-side panel for protocol §3.11 (small / large / mid Δcov)."""
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
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.55, label="Fitness")
    fig.suptitle(
        "Archive fitness across paired seeds (collapsed warm-start; gray = empty)",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fig",
        type=int,
        choices=(1, 2, 3, 4, 5, 6, 7, 8),
        action="append",
        dest="figs",
        help="Which figure(s) to export (repeatable)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export Fig. 1–8",
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
        else:
            if number == 2:
                fig02_elite_worlds(path, seed=args.fig2_seed)
            else:
                fn(path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Blocked analysis for the contemporaneous RQ1 policy×scalar experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldspace.illuminators.archive import (  # noqa: E402
    load_and_collapse_jsonl,
    merge_archives,
)
from worldspace.illuminators.archive_trace import qd_score_from_archive  # noqa: E402

ARMS = (
    "stub_minfit",
    "live_minfit",
    "stub_uniform",
    "live_uniform",
    "shuffled_uniform",
)
TARGET_FILLED = 971
N_CELLS = 2500
DEFAULT_ROOT = _ROOT / "artifacts" / "experiments" / "q1-rq1-contemporaneous"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def archive_metrics(floor_path: Path, run_archive_path: Path) -> dict[str, float]:
    floor = load_and_collapse_jsonl(floor_path, resolution=50)
    run = load_and_collapse_jsonl(run_archive_path, resolution=50)
    merge_archives(floor, run)
    filled = floor.filled_count()
    qd = qd_score_from_archive(floor)
    return {
        "filled_cells": float(filled),
        "coverage_pct": 100.0 * filled / N_CELLS,
        "new_coverage_pp": 100.0 * (filled - TARGET_FILLED) / N_CELLS,
        "qd_score": float(qd),
        "mean_elite_fitness": float(qd / filled) if filled else np.nan,
    }


def collect(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    floors = root / "floors"
    blocks = root / "blocks"
    for archive_dir in sorted(blocks.glob("archive_*")):
        archive_index = int(archive_dir.name.split("_")[-1])
        floor = floors / f"archive_{archive_index:02d}.jsonl"
        for continuation_dir in sorted(archive_dir.glob("continuation_*")):
            continuation_index = int(continuation_dir.name.split("_")[-1])
            for arm in ARMS:
                run_dir = continuation_dir / arm
                summary_path = run_dir / "nightly_run_summary.json"
                archive_path = run_dir / "map_elites_archive.jsonl"
                if not summary_path.is_file() or not archive_path.is_file():
                    raise FileNotFoundError(
                        f"incomplete cell: archive={archive_index} "
                        f"continuation={continuation_index} arm={arm}"
                    )
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "archive": archive_index,
                        "continuation": continuation_index,
                        "seed": int(summary["seed"]),
                        "arm": arm,
                        **archive_metrics(floor, archive_path),
                    }
                )
    frame = pd.DataFrame(rows)
    expected = 5 * 2 * len(ARMS)
    if len(frame) != expected:
        raise RuntimeError(f"expected {expected} cells, found {len(frame)}")
    return frame


def contrast_series(wide: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "scalar_at_minfit": wide["live_minfit"] - wide["stub_minfit"],
        "scalar_at_uniform": wide["live_uniform"] - wide["stub_uniform"],
        "policy_under_stub": wide["stub_uniform"] - wide["stub_minfit"],
        "policy_under_live": wide["live_uniform"] - wide["live_minfit"],
        "interaction": (
            (wide["live_uniform"] - wide["stub_uniform"])
            - (wide["live_minfit"] - wide["stub_minfit"])
        ),
        "alignment_live_minus_shuffled": (
            wide["live_uniform"] - wide["shuffled_uniform"]
        ),
    }


def cluster_bootstrap_ci(
    values: pd.Series,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    indexed = values.rename("value").reset_index()
    archives = np.array(sorted(indexed["archive"].unique()), dtype=int)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = rng.choice(archives, size=len(archives), replace=True)
        chunks = [
            indexed.loc[indexed["archive"] == archive, "value"].to_numpy()
            for archive in sampled
        ]
        estimates[draw] = float(np.concatenate(chunks).mean())
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return float(lo), float(hi)


def paired_correlations(wide: pd.DataFrame) -> dict[str, float]:
    pairs = {
        "live_vs_stub_minfit": ("live_minfit", "stub_minfit"),
        "live_vs_stub_uniform": ("live_uniform", "stub_uniform"),
        "uniform_vs_minfit_stub": ("stub_uniform", "stub_minfit"),
        "uniform_vs_minfit_live": ("live_uniform", "live_minfit"),
        "live_vs_shuffled_uniform": ("live_uniform", "shuffled_uniform"),
    }
    return {
        name: float(wide[left].corr(wide[right]))
        for name, (left, right) in pairs.items()
    }


def unpaired_sensitivity(wide: pd.DataFrame) -> dict[str, dict[str, float]]:
    pairs = {
        "scalar_at_minfit": ("live_minfit", "stub_minfit"),
        "scalar_at_uniform": ("live_uniform", "stub_uniform"),
        "policy_under_stub": ("stub_uniform", "stub_minfit"),
        "policy_under_live": ("live_uniform", "live_minfit"),
        "alignment_live_minus_shuffled": (
            "live_uniform",
            "shuffled_uniform",
        ),
    }
    out: dict[str, dict[str, float]] = {}
    for name, (left, right) in pairs.items():
        a = wide[left].to_numpy(dtype=float)
        b = wide[right].to_numpy(dtype=float)
        se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
        out[name] = {
            "difference_in_means": float(a.mean() - b.mean()),
            "unpaired_standard_error": se,
        }
    return out


def summarize_metric(
    frame: pd.DataFrame,
    metric: str,
    *,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    wide = frame.pivot(
        index=["archive", "continuation"],
        columns="arm",
        values=metric,
    ).sort_index()
    contrasts = contrast_series(wide)
    raw = pd.DataFrame(contrasts)
    summary: dict[str, object] = {}
    for index, (name, values) in enumerate(contrasts.items()):
        lo, hi = cluster_bootstrap_ci(
            values,
            draws=draws,
            seed=seed + index,
        )
        summary[name] = {
            "mean": float(values.mean()),
            "sd_across_continuation_blocks": float(values.std(ddof=1)),
            "archive_clustered_bootstrap_95ci": [lo, hi],
            "positive_blocks": int((values > 0).sum()),
            "blocks": int(len(values)),
        }
    summary["paired_correlations"] = paired_correlations(wide)
    summary["unpaired_sensitivity"] = unpaired_sensitivity(wide)
    return raw, summary


def main() -> None:
    args = parse_args()
    frame = collect(args.root)
    output_dir = args.root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "cell_metrics.csv", index=False)

    all_summary: dict[str, object] = {
        "design": {
            "archives": 5,
            "continuations_per_archive": 2,
            "outer_resampling_unit": "archive",
            "ci": "archive-clustered percentile bootstrap 95%",
            "bootstrap_draws": args.bootstrap,
        }
    }
    for offset, metric in enumerate(
        ("new_coverage_pp", "coverage_pct", "qd_score", "mean_elite_fitness")
    ):
        raw, summary = summarize_metric(
            frame,
            metric,
            draws=args.bootstrap,
            seed=args.seed + offset * 100,
        )
        raw.to_csv(output_dir / f"raw_block_deltas_{metric}.csv")
        all_summary[metric] = summary
    (output_dir / "factorial_summary.json").write_text(
        json.dumps(all_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

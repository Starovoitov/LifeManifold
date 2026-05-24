"""Compare MAP-Elites runs for Surrogate Acquisition A/B analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs candidate illuminator output directories",
    )
    parser.add_argument(
        "--baseline-dir",
        required=True,
        help="Output directory for baseline (e.g. acquisition off)",
    )
    parser.add_argument(
        "--candidate-dir",
        required=True,
        help="Output directory for candidate (e.g. filter mode)",
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=10,
        help="Archive resolution for filled-cell percentage",
    )
    return parser.parse_args()


class _RunSummary(TypedDict):
    evaluations: float
    filled_cells: float
    filled_cells_pct: float
    mean_best_fitness: float
    recommended_skip_rate: float | None


def main() -> None:
    args = parse_args()
    baseline = _summarize_run(
        Path(args.baseline_dir),
        grid_resolution=args.grid_resolution,
    )
    candidate = _summarize_run(
        Path(args.candidate_dir),
        grid_resolution=args.grid_resolution,
    )
    eval_reduction_pct = 0.0
    if baseline["evaluations"] > 0:
        eval_reduction_pct = (
            100.0
            * (baseline["evaluations"] - candidate["evaluations"])
            / baseline["evaluations"]
        )
    filled_delta_pct = 0.0
    if baseline["filled_cells_pct"] > 0:
        filled_delta_pct = (
            100.0
            * (candidate["filled_cells_pct"] - baseline["filled_cells_pct"])
            / baseline["filled_cells_pct"]
        )
    fitness_delta_pct = 0.0
    if baseline["mean_best_fitness"] > 0:
        fitness_delta_pct = (
            100.0
            * (candidate["mean_best_fitness"] - baseline["mean_best_fitness"])
            / baseline["mean_best_fitness"]
        )
    print("=== Acquisition A/B ===")
    print(f"baseline evaluations: {baseline['evaluations']}")
    print(f"candidate evaluations: {candidate['evaluations']}")
    print(f"eval reduction: {eval_reduction_pct:.1f}%")
    print(
        f"filled cells: baseline {baseline['filled_cells_pct']:.1f}% "
        f"candidate {candidate['filled_cells_pct']:.1f}% "
        f"(delta {filled_delta_pct:+.1f}%)"
    )
    print(
        f"mean best fitness: baseline {baseline['mean_best_fitness']:.4f} "
        f"candidate {candidate['mean_best_fitness']:.4f} "
        f"(delta {fitness_delta_pct:+.1f}%)"
    )
    if candidate.get("recommended_skip_rate") is not None:
        print(
            f"candidate policy skip rate (archive): "
            f"{candidate['recommended_skip_rate']:.1f}%"
        )


def _summarize_run(path: Path, *, grid_resolution: int) -> _RunSummary:
    archive_path = path / "map_elites_archive.jsonl"
    surrogate_path = path / "surrogate_archive.jsonl"
    evaluations = 0
    best_fitness_values: list[float] = []
    if archive_path.is_file():
        for line in archive_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            evaluations += 1
            best_fitness_values.append(float(record.get("fitness", 0.0)))
    filled_cells = len(best_fitness_values)
    total_cells = max(1, grid_resolution * grid_resolution)
    skip_count = 0
    slot_count = 0
    if surrogate_path.is_file():
        for line in surrogate_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            slot_count += 1
            record = json.loads(line)
            if record.get("decision") == "skip":
                skip_count += 1
    return {
        "evaluations": float(evaluations),
        "filled_cells": float(filled_cells),
        "filled_cells_pct": 100.0 * filled_cells / total_cells,
        "mean_best_fitness": (
            float(sum(best_fitness_values) / len(best_fitness_values))
            if best_fitness_values
            else 0.0
        ),
        "recommended_skip_rate": (
            100.0 * skip_count / slot_count if slot_count > 0 else None
        ),
    }


if __name__ == "__main__":
    main()

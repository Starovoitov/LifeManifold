#!/usr/bin/env python3
"""Collect reserved maze design data and train its surrogate ensemble."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.mazes.evaluation import evaluate_maze
from worldspace.mazes.genetics import mutate_maze, random_maze
from worldspace.mazes.surrogate import (
    buffer_row,
    load_buffer,
    save_checkpoint,
    train_checkpoint,
)

RESERVED_DESIGN_SEED = 20000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--buffer",
        type=Path,
        default=Path("artifacts/mazes/surrogate/buffer.jsonl"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/surrogate/checkpoints/maze_v1.pkl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/mazes/surrogate/report.json"),
    )
    parser.add_argument("--collect-count", type=int, default=2000)
    parser.add_argument(
        "--design-seed",
        type=int,
        default=RESERVED_DESIGN_SEED,
        help="Reserved seed block; must not overlap experiment seeds 0-9.",
    )
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--no-collect", action="store_true")
    parser.add_argument("--fitness-threshold", type=float, default=None)
    parser.add_argument("--uncertainty-threshold", type=float, default=None)
    parser.add_argument("--live-shadow-skip-rate", type=float, default=None)
    return parser.parse_args(argv)


def collect_buffer(path: Path, *, count: int, seed: int) -> None:
    if count < 20:
        raise ValueError("collect-count must be at least 20")
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    current = random_maze(rng)
    with path.open("w", encoding="utf-8") as handle:
        while len(seen) < count:
            if len(seen) % 3 == 0:
                current = random_maze(rng)
            else:
                current = mutate_maze(current, rng, edits=6)
            candidate_hash = current.candidate_hash()
            if candidate_hash in seen:
                continue
            seen.add(candidate_hash)
            design_seed = seed + len(seen)
            evaluation = evaluate_maze(current)
            handle.write(
                json.dumps(
                    buffer_row(
                        current,
                        evaluation,
                        design_seed=design_seed,
                    ),
                    ensure_ascii=True,
                )
                + "\n"
            )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.no_collect:
        collect_buffer(args.buffer, count=args.collect_count, seed=args.design_seed)
    features, targets = load_buffer(args.buffer)
    checkpoint, report = train_checkpoint(
        features,
        targets,
        ensemble_size=args.ensemble_size,
        max_iter=args.max_iter,
    )
    if args.fitness_threshold is not None:
        checkpoint.fitness_threshold = float(args.fitness_threshold)
        report["fitness_threshold"] = float(args.fitness_threshold)
        report["threshold_source"] = "live_shadow_override"
    if args.uncertainty_threshold is not None:
        checkpoint.uncertainty_threshold = float(args.uncertainty_threshold)
        report["uncertainty_threshold"] = float(args.uncertainty_threshold)
        report["threshold_source"] = "live_shadow_override"
    if args.live_shadow_skip_rate is not None:
        report["shadow_skip_rate"] = float(args.live_shadow_skip_rate)
        report["shadow_skip_gate_pass"] = (
            0.25 <= float(args.live_shadow_skip_rate) <= 0.45
        )
    save_checkpoint(checkpoint, args.checkpoint)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not report["quality_gate_pass"] or not report["shadow_skip_gate_pass"]:
        raise SystemExit("Maze surrogate gate failed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure maze genetic vs genetic_filter wall time under injected sim cost."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.mazes.evaluation import evaluate_maze
from worldspace.mazes.genetics import random_maze
from worldspace.mazes.runner import load_maze_scheduler, run_maze_qd
from worldspace.mazes.surrogate import MazeSurrogate


@dataclass(frozen=True)
class ArmTiming:
    condition: str
    sim_cost_ms: float
    proposals: int
    evaluations: int
    skipped: int
    skip_rate: float
    elapsed_seconds: float
    seconds_per_proposal: float
    seconds_per_evaluation: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--proposals", type=int, default=250)
    parser.add_argument(
        "--sim-cost-ms",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0],
        help="Injected per-eval delays to sweep (milliseconds).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mazes/walltime/benchmark.json"),
    )
    parser.add_argument(
        "--micro-samples",
        type=int,
        default=500,
        help="Samples for micro t_bfs / t_mlp measurement.",
    )
    return parser.parse_args(argv)


def _measure_micro_timings(samples: int) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    specs = [random_maze(rng) for _ in range(samples)]
    started = time.perf_counter()
    for spec in specs:
        evaluate_maze(spec, sim_cost_ms=0.0)
    t_bfs = (time.perf_counter() - started) / samples

    predictor = MazeSurrogate.load(ROOT / "artifacts/surrogate/checkpoints/maze_v1.pkl")
    started = time.perf_counter()
    for spec in specs:
        predictor.predict(spec)
    t_mlp = (time.perf_counter() - started) / samples
    return t_bfs, t_mlp


def _run_arm(
    *,
    condition: str,
    seed: int,
    proposals: int,
    sim_cost_ms: float,
    output_root: Path,
) -> ArmTiming:
    scheduler = ROOT / f"worldspace/specs/maze_scheduler_{condition}.yaml"
    config = load_maze_scheduler(scheduler)
    config = replace(
        config,
        iterations=proposals // config.batch_size,
        sim_cost_ms=sim_cost_ms,
    )
    predictor = (
        MazeSurrogate.load(Path(config.surrogate_checkpoint))
        if config.surrogate_checkpoint
        else None
    )
    out = output_root / condition / f"seed_{seed}" / f"sim_{sim_cost_ms:g}ms"
    result = run_maze_qd(
        config,
        seed=seed,
        output_dir=out,
        predictor=predictor,
    )
    skip_rate = result.skipped / result.proposals if result.proposals else 0.0
    return ArmTiming(
        condition=result.condition,
        sim_cost_ms=sim_cost_ms,
        proposals=result.proposals,
        evaluations=result.evaluations,
        skipped=result.skipped,
        skip_rate=skip_rate,
        elapsed_seconds=result.elapsed_seconds,
        seconds_per_proposal=result.elapsed_seconds / result.proposals,
        seconds_per_evaluation=(
            result.elapsed_seconds / result.evaluations if result.evaluations else 0.0
        ),
    )


def _break_even_injected_ms(
    *,
    skip_rate: float,
    t_bfs_seconds: float,
    t_mlp_seconds: float,
) -> float | None:
    """Minimum injected ms so filter wall time <= genetic at equal proposals."""
    if skip_rate <= 0.0:
        return None
    # Per proposal: genetic ~ t_sim; filter ~ t_mlp + (1-s)*t_sim, t_sim = t_bfs + inj.
    # Break-even: t_mlp + (1-s)*(t_bfs + inj) = t_bfs + inj  =>  inj = (t_mlp - s*t_bfs)/s
    numerator = t_mlp_seconds - skip_rate * t_bfs_seconds
    if numerator <= 0.0:
        return 0.0
    return 1000.0 * numerator / skip_rate


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.proposals <= 0:
        raise SystemExit("--proposals must be positive")
    output_root = args.output.parent / "runs"
    output_root.mkdir(parents=True, exist_ok=True)

    t_bfs, t_mlp = _measure_micro_timings(args.micro_samples)
    ratio_mlp_over_bfs = t_mlp / t_bfs if t_bfs > 0 else float("inf")

    arms: list[ArmTiming] = []
    for sim_cost_ms in args.sim_cost_ms:
        for condition in ("genetic", "genetic_filter"):
            arms.append(
                _run_arm(
                    condition=condition,
                    seed=args.seed,
                    proposals=args.proposals,
                    sim_cost_ms=sim_cost_ms,
                    output_root=output_root,
                )
            )

    filter_at_zero = next(
        arm
        for arm in arms
        if arm.condition == "genetic_filter" and arm.sim_cost_ms == 0.0
    )
    skip_rate = filter_at_zero.skip_rate
    break_even_ms = _break_even_injected_ms(
        skip_rate=skip_rate,
        t_bfs_seconds=t_bfs,
        t_mlp_seconds=t_mlp,
    )
    break_even_multiplier = (
        (t_bfs + (break_even_ms or 0.0) / 1000.0) / t_bfs if t_bfs > 0 else None
    )

    payload = {
        "schema_version": "maze-walltime-1.0",
        "seed": args.seed,
        "proposals": args.proposals,
        "micro_samples": args.micro_samples,
        "t_bfs_seconds": round(t_bfs, 9),
        "t_mlp_seconds": round(t_mlp, 9),
        "t_mlp_over_t_bfs": round(ratio_mlp_over_bfs, 3),
        "filter_skip_rate_at_zero_cost": round(skip_rate, 4),
        "break_even_injected_ms": (
            None if break_even_ms is None else round(break_even_ms, 3)
        ),
        "break_even_sim_slowdown_vs_bfs": (
            None if break_even_multiplier is None else round(break_even_multiplier, 2)
        ),
        "formula": "filter wins wall when t_mlp < skip_rate * (t_bfs + injected_ms/1000)",
        "arms": [
            {
                "condition": arm.condition,
                "sim_cost_ms": arm.sim_cost_ms,
                "proposals": arm.proposals,
                "evaluations": arm.evaluations,
                "skipped": arm.skipped,
                "skip_rate": round(arm.skip_rate, 4),
                "elapsed_seconds": arm.elapsed_seconds,
                "seconds_per_proposal": round(arm.seconds_per_proposal, 6),
                "seconds_per_evaluation": round(arm.seconds_per_evaluation, 6),
                "filter_faster_than_genetic": (
                    arm.condition == "genetic_filter"
                    and any(
                        other.condition == "genetic"
                        and other.sim_cost_ms == arm.sim_cost_ms
                        and arm.elapsed_seconds < other.elapsed_seconds
                        for other in arms
                    )
                ),
            }
            for arm in arms
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

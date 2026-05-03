from __future__ import annotations

import argparse
import json

from .generators import (
    RandomWalkWorldGenerator,
    RandomWorldGenerator,
    RuleBiasMarkovGenerator,
    TwoStateNoiseMarkovGenerator,
)
from .pipeline import explore_world_space, points_to_dicts, save_points_jsonl


def main() -> None:
    """Parse CLI args, run world-space exploration, and emit/save results."""
    parser = argparse.ArgumentParser(description="MVP explorer for the world-space cellular automata.")
    parser.add_argument("--generator", choices=["random", "random_walk", "markov_noise", "markov_rules"], default="random")
    parser.add_argument("--worlds", type=int, default=30)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--grid", type=int, default=40)
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output path; results are written as JSONL (one record per line).",
    )
    args = parser.parse_args()

    base = RandomWorldGenerator(grid_size=args.grid, steps=args.steps).generate(1)[0]
    if args.generator == "random":
        generator = RandomWorldGenerator(grid_size=args.grid, steps=args.steps)
    elif args.generator == "random_walk":
        generator = RandomWalkWorldGenerator(start_world=base, scale=0.02)
    elif args.generator == "markov_noise":
        generator = TwoStateNoiseMarkovGenerator(start_world=base)
    else:
        generator = RuleBiasMarkovGenerator(start_world=base)

    points = explore_world_space(generator=generator, n_worlds=args.worlds)
    payload = points_to_dicts(points)
    if args.output:
        save_points_jsonl(points, args.output)
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

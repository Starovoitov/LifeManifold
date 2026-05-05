"""Command-line entry for world-space streaming and optional embedding plots."""

from __future__ import annotations

import argparse
from pathlib import Path

from .generators import (
    RandomWalkWorldGenerator,
    RandomWorldGenerator,
    RuleBiasMarkovGenerator,
    TwoStateNoiseMarkovGenerator,
)
from .pipeline import stream_world_space_to_jsonl
from .viz import plot_world_embedding_from_jsonl


def main() -> None:
    """Parse CLI args and run world-space exploration with streaming JSONL output."""
    parser = argparse.ArgumentParser(
        description="MVP explorer for the world-space cellular automata."
    )
    parser.add_argument(
        "--generator",
        choices=["random", "random_walk", "markov_noise", "markov_rules", "neural"],
        default="random",
    )
    parser.add_argument(
        "--neural-spec",
        type=str,
        default="",
        help="YAML spec for --generator neural (default: bundled neural_world_generator.spec).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Torch device for --generator neural (overrides YAML torch.device).",
    )
    parser.add_argument("--worlds", type=int, default=30)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--grid", type=int, default=40)
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional JSONL path (one JSON object per line). Memory use is O(1) vs. number of worlds.",
    )
    parser.add_argument(
        "--echo-lines",
        action="store_true",
        help="When using --output, also print each JSON line to stdout.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="",
        help="Optional path to save PCA embedding scatter (requires --output).",
    )
    args = parser.parse_args()

    base = RandomWorldGenerator(grid_size=args.grid, steps=args.steps).generate(1)[0]
    if args.generator == "random":
        generator = RandomWorldGenerator(grid_size=args.grid, steps=args.steps)
    elif args.generator == "random_walk":
        generator = RandomWalkWorldGenerator(start_world=base, scale=0.02)
    elif args.generator == "markov_noise":
        generator = TwoStateNoiseMarkovGenerator(start_world=base)
    elif args.generator == "markov_rules":
        generator = RuleBiasMarkovGenerator(start_world=base)
    else:
        from .neural_world import NeuralWorldGenerator

        spec_path = Path(args.neural_spec) if args.neural_spec.strip() else None
        dev_kw = None if args.device == "auto" else args.device
        generator = NeuralWorldGenerator(spec_path=spec_path, device=dev_kw)

    out_path = args.output or None
    echo_stdout = (out_path is None) or args.echo_lines
    stream_world_space_to_jsonl(
        generator,
        args.worlds,
        out_path,
        k_clusters=4,
        echo_stdout=echo_stdout,
    )

    if args.plot:
        if not args.output:
            parser.error(
                "--plot requires --output (plot is built from the JSONL file)."
            )
        title_parts = [f"generator={args.generator}", f"worlds={args.worlds}"]
        if args.generator == "neural":
            title_parts.append(f"device={args.device}")
        plot_world_embedding_from_jsonl(
            args.output,
            args.plot,
            title=", ".join(title_parts),
        )


if __name__ == "__main__":
    main()

"""Command-line entry for world-space streaming and optional embedding plots."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from .generators import (
    GeneticWorldGenerator,
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
        choices=[
            "random",
            "random_walk",
            "markov_noise",
            "markov_rules",
            "genetic",
            "neural",
        ],
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
        "--ga-population",
        type=int,
        default=12,
        help="Population size for --generator genetic.",
    )
    parser.add_argument(
        "--ga-elite",
        type=int,
        default=3,
        help="Elite survivors per generation for --generator genetic.",
    )
    parser.add_argument(
        "--ga-mutation-scale",
        type=float,
        default=0.02,
        help="Gaussian mutation scale for --generator genetic.",
    )
    parser.add_argument(
        "--ga-seed",
        type=int,
        default=0,
        help="RNG seed for --generator genetic.",
    )
    parser.add_argument(
        "--genetic-spec",
        type=str,
        default="",
        help="YAML spec for --generator genetic (default: bundled genetic_world_generator.yaml).",
    )
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
        help="Path to save PCA embedding scatter. If set without --output, JSONL is written to a temp file then removed.",
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
    elif args.generator == "genetic":
        genetic_spec = Path(args.genetic_spec) if args.genetic_spec.strip() else None
        generator = GeneticWorldGenerator(
            grid_size=args.grid,
            steps=args.steps,
            population_size=args.ga_population,
            elite_count=args.ga_elite,
            mutation_scale=args.ga_mutation_scale,
            seed=args.ga_seed,
            spec_path=genetic_spec,
        )
    else:
        from .neural_world import NeuralWorldGenerator

        spec_path = Path(args.neural_spec) if args.neural_spec.strip() else None
        dev_kw = None if args.device == "auto" else args.device
        generator = NeuralWorldGenerator(spec_path=spec_path, device=dev_kw)

    out_arg = args.output.strip()
    temp_jsonl: Path | None = None
    if args.plot.strip() and not out_arg:
        fd, tmp = tempfile.mkstemp(suffix=".jsonl", prefix="worldspace-")
        os.close(fd)
        temp_jsonl = Path(tmp)
        out_path: str | Path | None = temp_jsonl
        echo_stdout = args.echo_lines
    else:
        out_path = out_arg or None
        echo_stdout = (out_path is None) or args.echo_lines

    try:
        stream_world_space_to_jsonl(
            generator,
            args.worlds,
            out_path,
            k_clusters=4,
            echo_stdout=echo_stdout,
        )

        if args.plot.strip():
            jsonl_src = out_arg if out_arg else str(temp_jsonl)
            title_parts = [f"generator={args.generator}", f"worlds={args.worlds}"]
            if args.generator == "neural":
                title_parts.append(f"device={args.device}")
            plot_world_embedding_from_jsonl(
                jsonl_src,
                args.plot.strip(),
                title=", ".join(title_parts),
            )
    finally:
        if temp_jsonl is not None:
            try:
                temp_jsonl.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()

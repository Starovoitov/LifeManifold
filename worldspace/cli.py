"""Command-line entry for world-space streaming and optional trace files."""

from __future__ import annotations

import argparse
from pathlib import Path

from .generators import (
    GeneticWorldGenerator,
    HybridGALlmWorldGenerator,
    make_llm_world_generator,
    RandomWalkWorldGenerator,
    RandomWorldGenerator,
    RuleBiasMarkovGenerator,
    TwoStateNoiseMarkovGenerator,
)
from .cli_generator_spec import parse_generator_spec_path, validate_generator_spec_yaml
from .cli_mapelites import add_mapelites_arguments, run_mapelites_cli
from .pipeline import stream_world_space_to_jsonl

_MAPELITES_DEFAULT_STEPS = 300


def main() -> None:
    """Parse CLI args and run world-space exploration with streaming JSONL output."""
    parser = argparse.ArgumentParser(
        description="MVP explorer for the world-space cellular automata."
    )
    add_mapelites_arguments(parser)
    parser.add_argument(
        "--generator",
        choices=[
            "random",
            "random_walk",
            "markov_noise",
            "markov_rules",
            "genetic",
            "hybrid",
            "llm",
            "neural",
        ],
        default="random",
    )
    parser.add_argument(
        "--generator-spec",
        type=str,
        default="",
        help=(
            "YAML spec path for --generator genetic, llm, hybrid, or neural "
            "(default: bundled spec for that generator). Ignored for other generators."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Torch device for --generator neural (overrides YAML torch.device).",
    )
    parser.add_argument("--worlds", type=int, default=30)
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help=(
            "CA steps per world. Legacy default 200. "
            f"For --illuminator mapelites use >= 200 (recommended {_MAPELITES_DEFAULT_STEPS})."
        ),
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=40,
        help="Simulation grid side length (legacy default 40; mapelites often uses 50).",
    )
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
        "--echo-lines",
        action="store_true",
        help=(
            "Print each main JSONL line to stdout as it is produced (same payload as "
            "``--metrics-trace`` when that path is set). "
            "Omit for quiet runs that only write trace files."
        ),
    )
    parser.add_argument(
        "--metrics-trace",
        type=str,
        default="",
        help=(
            "Optional JSONL: one line per world after embedding + k-means "
            "(yield_index, world, metrics, dominant_metric_delta_xy, "
            "dominant_metric_delta_axis_labels, "
            "cluster_id); "
            "suitable for legacy ``python -m worldspace.visualizer`` PNG export; "
            "MAP-Elites archives: use Streamlit dashboard (dashboard/Home.py). "
            "Any --generator."
        ),
    )
    parser.add_argument(
        "--ca-step-trace",
        type=str,
        default="",
        help="Optional JSONL: one line per CA timestep per pipeline run_world (yield_index, ca_step, metrics); any --generator.",
    )
    args = parser.parse_args()

    if args.illuminator == "mapelites":
        run_mapelites_cli(args)
        return

    spec_generators = frozenset({"genetic", "llm", "hybrid", "neural"})
    gen_spec_path = parse_generator_spec_path(args.generator_spec)
    if args.generator_spec.strip() and args.generator not in spec_generators:
        parser.error(
            f"--generator-spec is only valid with --generator "
            f"genetic|llm|hybrid|neural (got {args.generator!r})."
        )
    if gen_spec_path is not None and args.generator in spec_generators:
        try:
            validate_generator_spec_yaml(args.generator, gen_spec_path)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))

    trace_arg = args.metrics_trace.strip()
    metrics_trace_path = Path(trace_arg).expanduser() if trace_arg else None
    ca_arg = args.ca_step_trace.strip()
    ca_step_trace_path = Path(ca_arg).expanduser() if ca_arg else None

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
        generator = GeneticWorldGenerator(
            grid_size=args.grid,
            steps=args.steps,
            population_size=args.ga_population,
            elite_count=args.ga_elite,
            mutation_scale=args.ga_mutation_scale,
            seed=args.ga_seed,
            spec_path=gen_spec_path,
        )
    elif args.generator == "llm":
        generator = make_llm_world_generator(
            grid_size=args.grid,
            steps=args.steps,
            seed=args.ga_seed,
            spec_path=gen_spec_path,
        )
    elif args.generator == "hybrid":
        generator = HybridGALlmWorldGenerator(
            grid_size=args.grid,
            steps=args.steps,
            seed=args.ga_seed,
            spec_path=gen_spec_path,
        )
    else:
        from .generators.neural_world import NeuralWorldGenerator

        dev_kw = None if args.device == "auto" else args.device
        generator = NeuralWorldGenerator(spec_path=gen_spec_path, device=dev_kw)

    echo_stdout = bool(args.echo_lines)

    stream_world_space_to_jsonl(
        generator,
        args.worlds,
        None,
        k_clusters=4,
        echo_stdout=echo_stdout,
        metrics_trace_path=metrics_trace_path,
        ca_step_trace_path=ca_step_trace_path,
    )


if __name__ == "__main__":
    main()

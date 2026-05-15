"""Command-line entry for world-space streaming and optional trace files."""

from __future__ import annotations

import argparse
from pathlib import Path

from .generators import (
    GeneticWorldGenerator,
    HybridGALlmWorldGenerator,
    LLMWorldGenerator,
    RandomWalkWorldGenerator,
    RandomWorldGenerator,
    RuleBiasMarkovGenerator,
    TwoStateNoiseMarkovGenerator,
)
from .pipeline import stream_world_space_to_jsonl


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
            "hybrid",
            "llm",
            "neural",
        ],
        default="random",
    )
    parser.add_argument(
        "--neural-spec",
        type=str,
        default="",
        help="YAML spec for --generator neural (default: bundled neural_world_generator.yaml).",
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
        "--llm-spec",
        type=str,
        default="",
        help="YAML spec for --generator llm (default: bundled llm_world_generator.yaml).",
    )
    parser.add_argument(
        "--hybrid-spec",
        type=str,
        default="",
        help="YAML spec for --generator hybrid (default: bundled hybrid_world_generator.yaml).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help=(
            "Optional JSONL path for full world-space records (embedding + cluster). "
            "Omit to skip writing main JSONL (e.g. metrics-only or CA-step trace runs)."
        ),
    )
    parser.add_argument(
        "--echo-lines",
        action="store_true",
        help=(
            "Print each main JSONL line to stdout. When --output is set, mirrors the "
            "file; when --output is omitted, streams JSONL to stdout only if this flag "
            "is set."
        ),
    )
    parser.add_argument(
        "--metrics-trace",
        type=str,
        default="",
        help="Optional JSONL: one line per yielded world (yield_index, world, metrics) during the first pipeline pass; any --generator.",
    )
    parser.add_argument(
        "--ca-step-trace",
        type=str,
        default="",
        help="Optional JSONL: one line per CA timestep per pipeline run_world (yield_index, ca_step, metrics); any --generator.",
    )
    args = parser.parse_args()

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
    elif args.generator == "llm":
        llm_spec = Path(args.llm_spec) if args.llm_spec.strip() else None
        generator = LLMWorldGenerator(
            grid_size=args.grid,
            steps=args.steps,
            seed=args.ga_seed,
            spec_path=llm_spec,
        )
    elif args.generator == "hybrid":
        hybrid_spec = Path(args.hybrid_spec) if args.hybrid_spec.strip() else None
        generator = HybridGALlmWorldGenerator(
            grid_size=args.grid,
            steps=args.steps,
            seed=args.ga_seed,
            spec_path=hybrid_spec,
        )
    else:
        from .generators.neural_world import NeuralWorldGenerator

        spec_path = Path(args.neural_spec) if args.neural_spec.strip() else None
        dev_kw = None if args.device == "auto" else args.device
        generator = NeuralWorldGenerator(spec_path=spec_path, device=dev_kw)

    out_arg = args.output.strip()
    out_path: str | Path | None = out_arg or None
    echo_stdout = bool(args.echo_lines)

    stream_world_space_to_jsonl(
        generator,
        args.worlds,
        out_path,
        k_clusters=4,
        echo_stdout=echo_stdout,
        metrics_trace_path=metrics_trace_path,
        ca_step_trace_path=ca_step_trace_path,
    )


if __name__ == "__main__":
    main()

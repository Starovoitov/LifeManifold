"""Standalone CLI for ``python -m worldspace.illuminators``."""

from __future__ import annotations

import argparse
from typing import Literal

from worldspace.illuminators.evaluation import ILLUMINATOR_MIN_STEPS
from worldspace.illuminators.illuminator import MapElitesIlluminator, MapElitesRunResult
from worldspace.illuminators.scheduler import DEFAULT_SCHEDULER_PATH

_DEFAULT_GRID = 50
_DEFAULT_OUTPUT = "output"


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for the illuminators package entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run MAP-Elites (quality-diversity illuminator).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global RNG seed for scheduler, bin selection, and emitters.",
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=None,
        help="Override archive grid side length from scheduler YAML.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=_DEFAULT_GRID,
        help="Simulation grid side length per candidate.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=ILLUMINATOR_MIN_STEPS,
        help=f"CA steps per candidate (minimum {ILLUMINATOR_MIN_STEPS}).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override scheduler YAML iterations.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="",
        help=f"Scheduler YAML path (default: {DEFAULT_SCHEDULER_PATH}).",
    )
    parser.add_argument(
        "--load-archive",
        type=str,
        default="",
        help="Optional existing archive JSONL to collapse and resume from.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=_DEFAULT_OUTPUT,
        help="Directory for map_elites_archive.jsonl and surrogate_archive.jsonl.",
    )
    parser.add_argument(
        "--archive-type",
        choices=["grid", "cvt"],
        default=None,
        help="Override scheduler archive.type (requires schema_version 1.3).",
    )
    return parser


def run_illuminator_cli(args: argparse.Namespace) -> MapElitesRunResult:
    """Execute a MAP-Elites run from parsed CLI arguments."""
    if args.steps < ILLUMINATOR_MIN_STEPS:
        raise SystemExit(
            f"--steps must be >= {ILLUMINATOR_MIN_STEPS} for the MAP-Elites illuminator"
        )
    scheduler_path = args.scheduler.strip() or DEFAULT_SCHEDULER_PATH
    load_path = args.load_archive.strip()
    archive_type: Literal["grid", "cvt"] | None = getattr(args, "archive_type", None)
    return MapElitesIlluminator().run(
        scheduler_path=scheduler_path,
        output_dir=args.output_dir,
        seed=args.seed,
        grid_resolution=args.grid_resolution,
        grid_size=args.grid,
        steps=args.steps,
        iterations=args.iterations,
        load_archive_path=load_path or None,
        archive_type=archive_type,
    )


def print_run_summary(result: MapElitesRunResult) -> None:
    """Print a short human-readable summary after a run."""
    print(
        f"MAP-Elites done: {result.evaluations} evaluations, "
        f"{result.filled_cells} occupied bins, "
        f"archive {result.archive_jsonl_path}"
    )
    if result.surrogate_archive_jsonl_path is not None:
        path = result.surrogate_archive_jsonl_path
        lines = 0
        if path.is_file():
            lines = sum(
                1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
            )
        print(f"SurrogateArchive: {path} ({lines} lines)")


def main(argv: list[str] | None = None) -> None:
    """Parse argv and run the illuminator."""
    args = build_parser().parse_args(argv)
    result = run_illuminator_cli(args)
    print_run_summary(result)


if __name__ == "__main__":
    main()

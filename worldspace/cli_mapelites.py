"""CLI for ``python -m worldspace --illuminator mapelites``."""

from __future__ import annotations

import argparse

from worldspace.illuminators.cli import print_run_summary, run_illuminator_cli
from worldspace.illuminators.evaluation import ILLUMINATOR_MIN_STEPS
from worldspace.illuminators.scheduler import DEFAULT_SCHEDULER_PATH


def add_mapelites_arguments(parser: argparse.ArgumentParser) -> None:
    """Register MAP-Elites illuminator flags on ``parser``."""
    parser.add_argument(
        "--illuminator",
        choices=["mapelites"],
        default=None,
        help="Run MAP-Elites quality-diversity search (not legacy --generator).",
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
        default=_DEFAULT_GRID,
        help="Archive grid side length (behavioral bins per axis).",
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
        help="Directory for map_elites_archive.jsonl.",
    )


def run_mapelites_cli(args: argparse.Namespace) -> None:
    """Execute a MAP-Elites run from parsed CLI arguments."""
    if args.steps < ILLUMINATOR_MIN_STEPS:
        raise SystemExit(
            f"--steps must be >= {ILLUMINATOR_MIN_STEPS} for --illuminator mapelites"
        )
    result = run_illuminator_cli(args)
    print_run_summary(result)


_DEFAULT_GRID = 50
_DEFAULT_OUTPUT = "output"

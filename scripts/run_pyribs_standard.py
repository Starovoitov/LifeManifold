#!/usr/bin/env python3
"""CLI for the supplementary pyribs Sphere/Rastrigin benchmarks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.benchmarks.pyribs_standard_runner import (
    DEFAULT_EVALUATIONS,
    PyribsStandardConfig,
    run_pyribs_standard,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a CA-independent pyribs Sphere/Rastrigin QD benchmark.",
    )
    parser.add_argument(
        "--benchmark",
        choices=("sphere", "rastrigin"),
        required=True,
    )
    parser.add_argument(
        "--algo",
        choices=("cma_me", "cma_mae", "me_random"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--evaluations",
        type=int,
        default=DEFAULT_EVALUATIONS,
        help=f"Exact evaluation budget (default: {DEFAULT_EVALUATIONS}).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.benchmark == "rastrigin" and args.algo == "me_random":
        parser.error("me_random is only defined for the sphere benchmark")
    return args


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    config = PyribsStandardConfig(
        benchmark=args.benchmark,
        algo=args.algo,
        seed=args.seed,
        evaluations=args.evaluations,
    )
    result = run_pyribs_standard(config, output_dir=args.output_dir)
    logging.info(
        "Done benchmark=%s algo=%s seed=%s evals=%s elites=%s "
        "coverage=%.4f mean_fit=%s elapsed=%.3fs",
        result.benchmark,
        result.algo,
        result.seed,
        result.evaluations,
        result.filled_cells,
        result.coverage,
        result.mean_best_fitness,
        result.elapsed_seconds,
    )


if __name__ == "__main__":
    main()

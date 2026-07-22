#!/usr/bin/env python3
"""CLI: pyribs CMA-ME / CMA-MAE baseline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.illuminators.pyribs_baseline import (
    DEFAULT_BASELINE_ARCHIVE,
    DEFAULT_EMITTER_BATCH_SIZE,
    DEFAULT_EVALUATIONS,
    DEFAULT_NUM_EMITTERS,
    DEFAULT_SIGMA0,
    PyribsBaselineConfig,
    run_pyribs_baseline,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CMA-ME or CMA-MAE (pyribs) with LifeManifold illuminator eval.",
    )
    parser.add_argument(
        "--algo",
        choices=("cma_me", "cma_mae"),
        required=True,
        help="QD algorithm arm.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--evaluations",
        type=int,
        default=DEFAULT_EVALUATIONS,
        help=f"Exact sim budget (default {DEFAULT_EVALUATIONS}; must divide ask size).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Run output directory (summary + archive JSONL).",
    )
    parser.add_argument(
        "--load-archive",
        type=Path,
        default=DEFAULT_BASELINE_ARCHIVE,
        help="Warm-start MAP-Elites JSONL (default: nightly grid baseline).",
    )
    parser.add_argument(
        "--no-load-archive",
        action="store_true",
        help="Start from an empty pyribs archive.",
    )
    parser.add_argument("--grid-size", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--num-emitters",
        type=int,
        default=DEFAULT_NUM_EMITTERS,
        help=f"T0 default {DEFAULT_NUM_EMITTERS}; override for smoke only.",
    )
    parser.add_argument(
        "--emitter-batch-size",
        type=int,
        default=DEFAULT_EMITTER_BATCH_SIZE,
        help=f"T0 default {DEFAULT_EMITTER_BATCH_SIZE}; override for smoke only.",
    )
    parser.add_argument("--sigma0", type=float, default=DEFAULT_SIGMA0)
    parser.add_argument(
        "--no-parallel-eval",
        action="store_true",
        help="Disable forkserver parallel simulation.",
    )
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument(
        "--decode-mode",
        choices=("rint", "threshold", "bernoulli"),
        default="rint",
        help="Rule-bit decode for CMA continuous genome (default: rint = frozen B2).",
    )
    parser.add_argument(
        "--condition-label",
        default=None,
        help="Summary/aggregate condition name (default: --algo value).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    load_archive = None if args.no_load_archive else Path(args.load_archive)
    config = PyribsBaselineConfig(
        algo=args.algo,
        seed=args.seed,
        evaluations=args.evaluations,
        num_emitters=args.num_emitters,
        emitter_batch_size=args.emitter_batch_size,
        sigma0=args.sigma0,
        grid_size=args.grid_size,
        steps=args.steps,
        load_archive=load_archive,
        parallel_eval=not args.no_parallel_eval,
        parallel_workers=args.parallel_workers,
        decode_mode=args.decode_mode,
        condition_label=args.condition_label,
    )
    result = run_pyribs_baseline(config, output_dir=Path(args.output_dir))
    logging.info(
        "Done algo=%s seed=%s evals=%s elites=%s coverage=%.4f mean_fit=%s elapsed=%.1fs",
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

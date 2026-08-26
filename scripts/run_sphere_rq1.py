#!/usr/bin/env python3
"""Run one Sphere RQ1 / H1 condition and seed (Fontaine D=20).

Usage:
  python scripts/run_sphere_rq1.py train-surrogate [--out PATH]
  python scripts/run_sphere_rq1.py --scheduler YAML --seed N --output-dir DIR
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.benchmarks.sphere_llm import MockSphereLlmEmitter, SphereLlmEmitter
from worldspace.benchmarks.sphere_rq1 import (
    DEFAULT_H1_CHECKPOINT,
    load_sphere_h1_surrogate,
    load_sphere_scheduler,
    run_sphere_qd,
    save_sphere_h1_surrogate,
    train_sphere_h1_surrogate,
)
from worldspace.generators.llm_call_log import (
    configure_llm_call_log,
    resolve_llm_call_log_path,
)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] == "train-surrogate":
        _train(_parse_train(args_list[1:]))
        return
    _run(_parse_run(args_list))


def _parse_train(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit the frozen Sphere H1 ensemble")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=20_000)
    parser.add_argument("--n-members", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_H1_CHECKPOINT)
    return parser.parse_args(argv)


def _parse_run(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--proposals",
        type=int,
        default=None,
        help="Override exact proposal budget; must divide scheduler batch size.",
    )
    parser.add_argument(
        "--llm-spec",
        type=Path,
        default=Path("worldspace/specs/llm_world_generator_rq1_fixed_openai.yaml"),
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use Gaussian local mutations instead of a remote LLM.",
    )
    return parser.parse_args(argv)


def _train(args: argparse.Namespace) -> None:
    sur = train_sphere_h1_surrogate(
        seed=args.seed,
        n_train=args.n_train,
        n_members=args.n_members,
    )
    save_sphere_h1_surrogate(sur, args.out)
    logging.info(
        "Wrote %s  members=%s n_train=%s MAE=%.4f",
        args.out,
        sur.n_members,
        sur.n_train,
        sur.train_mae,
    )


def _run(args: argparse.Namespace) -> None:
    config = load_sphere_scheduler(args.scheduler)
    if args.proposals is not None:
        if args.proposals <= 0 or args.proposals % config.batch_size:
            raise SystemExit(
                f"--proposals must be positive and divisible by {config.batch_size}"
            )
        config = replace(config, iterations=args.proposals // config.batch_size)
    predictor = (
        load_sphere_h1_surrogate(Path(config.surrogate_checkpoint))
        if config.surrogate_checkpoint
        else None
    )
    llm_emitter = None
    if "llm" in config.emitters:
        if args.mock_llm:
            llm_emitter = MockSphereLlmEmitter(sigma=config.sigma)
        else:
            llm_emitter = SphereLlmEmitter(
                prompt_mode=config.llm_prompt_mode,
                llm_spec_path=args.llm_spec,
                sigma=config.sigma,
            )
    configure_llm_call_log(resolve_llm_call_log_path(output_dir=args.output_dir))
    result = run_sphere_qd(
        config,
        seed=args.seed,
        output_dir=args.output_dir,
        predictor=predictor,
        llm_emitter=llm_emitter,
    )
    logging.info(
        "Done condition=%s seed=%s proposals=%s evals=%s coverage=%.4f",
        result.condition,
        result.seed,
        result.proposals,
        result.evaluations,
        result.coverage,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run one self-contained dungeon QD condition and seed."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.dungeons.runner import (
    load_dungeon_scheduler,
    run_dungeon_qd,
)
from worldspace.dungeons.llm_emitter import DungeonLlmEmitter
from worldspace.dungeons.surrogate import DungeonSurrogate
from worldspace.generators.llm_call_log import (
    configure_llm_call_log,
    resolve_llm_call_log_path,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        default=Path("worldspace/specs/llm_world_generator_qwen.yaml"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    config = load_dungeon_scheduler(args.scheduler)
    if args.proposals is not None:
        if args.proposals <= 0 or args.proposals % config.batch_size:
            raise SystemExit(
                f"--proposals must be positive and divisible by {config.batch_size}"
            )
        config = replace(
            config,
            iterations=args.proposals // config.batch_size,
        )
    predictor = (
        DungeonSurrogate.load(Path(config.surrogate_checkpoint))
        if config.surrogate_checkpoint
        else None
    )
    llm_emitter = (
        DungeonLlmEmitter(
            prompt_mode=config.llm_prompt_mode,
            llm_spec_path=args.llm_spec,
        )
        if "llm" in config.emitters
        else None
    )
    configure_llm_call_log(resolve_llm_call_log_path(output_dir=args.output_dir))
    result = run_dungeon_qd(
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

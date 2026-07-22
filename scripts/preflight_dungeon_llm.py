#!/usr/bin/env python3
"""Run and gate a small live dungeon-LLM preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.dungeons.archive import DungeonArchive, DungeonElite
from worldspace.dungeons.emitters import select_uniform_frontier
from worldspace.dungeons.evaluation import evaluate_dungeon
from worldspace.dungeons.genetics import random_dungeon
from worldspace.dungeons.llm_emitter import DungeonLlmEmitter
from worldspace.dungeons.surrogate import DungeonSurrogate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/dungeons/surrogate/checkpoint.pkl"),
    )
    parser.add_argument(
        "--llm-spec",
        type=Path,
        default=Path("worldspace/specs/llm_world_generator_qwen.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/dungeons/llm_preflight.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    predictor = DungeonSurrogate.load(args.checkpoint)
    emitter = DungeonLlmEmitter(
        prompt_mode="hints",
        llm_spec_path=args.llm_spec,
    )
    archive = DungeonArchive(30)
    initial = random_dungeon(rng)
    initial_eval = evaluate_dungeon(initial, seed=args.seed)
    archive.try_insert(
        DungeonElite(
            bin=archive.bin_for_measures(initial_eval.measures),
            fitness=initial_eval.fitness,
            measures=initial_eval.measures,
            spec=initial,
            candidate_id="preflight-parent",
            parent_id=None,
            emitter_type="random",
        )
    )
    for index in range(args.calls):
        target = select_uniform_frontier(archive, rng)
        prediction = (
            predictor.predict(target.parent.spec) if target.parent is not None else None
        )
        emitted = emitter.emit(
            target=target,
            archive=archive,
            rng=rng,
            prediction=prediction,
        )
        evaluation = evaluate_dungeon(emitted.spec, seed=args.seed + index + 1)
        archive.try_insert(
            DungeonElite(
                bin=archive.bin_for_measures(evaluation.measures),
                fitness=evaluation.fitness,
                measures=evaluation.measures,
                spec=emitted.spec,
                candidate_id=f"preflight-{index}",
                parent_id=emitted.parent_id,
                emitter_type=emitted.emitter_type,
            )
        )
    report = emitter.audit.to_dict()
    parse_success_rate = report["parse_success_rate"]
    mean_tile_distance = report["mean_tile_distance"]
    repair_collapse_rate = report["repair_collapse_rate"]
    assert isinstance(parse_success_rate, (int, float))
    assert isinstance(mean_tile_distance, (int, float))
    assert isinstance(repair_collapse_rate, (int, float))
    report.update(
        {
            "schema_version": "dungeon-llm-preflight-1.0",
            "prompt_version": emitter.prompt_version,
            "parse_gate_pass": float(parse_success_rate) >= 0.95,
            "distance_gate_pass": float(mean_tile_distance) > 0.0,
            "repair_gate_pass": float(repair_collapse_rate) < 0.10,
        }
    )
    report["all_gates_pass"] = all(
        report[key]
        for key in ("parse_gate_pass", "distance_gate_pass", "repair_gate_pass")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not report["all_gates_pass"]:
        raise SystemExit("Dungeon LLM preflight failed")


if __name__ == "__main__":
    main()

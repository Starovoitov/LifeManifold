#!/usr/bin/env python3
"""Run and gate a small live maze-LLM preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.mazes.archive import MazeArchive, MazeElite
from worldspace.mazes.emitters import select_uniform_frontier
from worldspace.mazes.evaluation import evaluate_maze
from worldspace.mazes.genetics import random_maze
from worldspace.mazes.llm_emitter import MazeLlmEmitter
from worldspace.mazes.surrogate import MazeSurrogate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/surrogate/checkpoints/maze_v1.pkl"),
    )
    parser.add_argument(
        "--llm-spec",
        type=Path,
        default=Path("worldspace/specs/llm_world_generator_qwen.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mazes/llm_preflight.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    predictor = MazeSurrogate.load(args.checkpoint)
    emitter = MazeLlmEmitter(
        prompt_mode="hints",
        llm_spec_path=args.llm_spec,
    )
    archive = MazeArchive(30)
    initial = random_maze(rng)
    initial_eval = evaluate_maze(initial)
    archive.try_insert(
        MazeElite(
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
        evaluation = evaluate_maze(emitted.spec)
        archive.try_insert(
            MazeElite(
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
            "schema_version": "maze-llm-preflight-1.0",
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
        raise SystemExit("Maze LLM preflight failed")


if __name__ == "__main__":
    main()

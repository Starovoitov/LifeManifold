#!/usr/bin/env python3
"""Live Sphere LLM preflight: parse/fallback gates before the 40-run 2×2.

Does not launch Phase B. Protocol: artifacts/Q1_RQ1_SPHERE_DOMAIN.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ribs.archives import GridArchive

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.benchmarks.qd_sphere import (
    DEFAULT_SOLUTION_DIM,
    archive_ranges,
    clip_solution,
    linear_projection_measures,
    sphere_objective,
)
from worldspace.benchmarks.sphere_llm import SphereLlmEmitter
from worldspace.benchmarks.sphere_rq1 import select_target_cell


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--llm-spec",
        type=Path,
        default=Path("worldspace/specs/llm_world_generator_rq1_fixed_openai.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/experiments/q1-rq1-sphere-factorial/llm_preflight.json"
        ),
    )
    return parser.parse_args()


def _seeded_archive(rng: np.random.Generator) -> GridArchive:
    archive = GridArchive(
        solution_dim=DEFAULT_SOLUTION_DIM,
        dims=(10, 10),
        ranges=archive_ranges(DEFAULT_SOLUTION_DIM),
        seed=0,
        learning_rate=1.0,
    )
    for _ in range(30):
        child = clip_solution(rng.uniform(-5.12, 5.12, size=DEFAULT_SOLUTION_DIM))
        archive.add(
            child[np.newaxis, :],
            np.asarray([float(sphere_objective(child))]),
            linear_projection_measures(child)[np.newaxis, :],
        )
    return archive


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    archive = _seeded_archive(rng)
    emitter = SphereLlmEmitter(
        prompt_mode="stub",
        llm_spec_path=args.llm_spec,
    )
    for _ in range(args.calls):
        target = select_target_cell(archive, rng, target_selection="uniform_frontier")
        emitter.emit(target=target, rng=rng, prediction=None)
    report = emitter.audit.to_dict()
    parse_success_rate = float(report["parse_success_rate"])
    fallback_rate = float(report["fallback_rate"])
    mean_l2 = float(report["mean_l2"])
    report.update(
        {
            "schema_version": "sphere-llm-preflight-1.0",
            "prompt_version": emitter.prompt_version,
            "max_retries": emitter.max_retries,
            "calls": args.calls,
            "llm_spec": str(args.llm_spec),
            "parse_gate_pass": parse_success_rate >= 0.90,
            "fallback_gate_pass": fallback_rate <= 0.10,
            "distance_gate_pass": mean_l2 >= 0.1,
        }
    )
    report["all_gates_pass"] = all(
        report[key]
        for key in (
            "parse_gate_pass",
            "fallback_gate_pass",
            "distance_gate_pass",
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not report["all_gates_pass"]:
        raise SystemExit("Sphere LLM preflight failed")


if __name__ == "__main__":
    main()

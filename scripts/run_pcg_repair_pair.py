#!/usr/bin/env python3
"""Paired PCG smoke: identity vs structural_counts. No LLM.

Does not overwrite pcg_smoke.json or identity bin edges.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.pcg.pin_smoke import run_problem
from worldspace.pcg.repair import RepairKind
from worldspace.pcg.smoke import (
    RESERVED_PAIR_SOKOBAN_SEED,
    RESERVED_PAIR_ZELDA_SEED,
    SMOKE_EVALUATIONS,
)
from worldspace.pcg.spec import SOKOBAN_V0, ZELDA_V0, PcgTask

DEFAULT_OUT = ROOT / "artifacts/controlled_attribution/pcg"
PAIR_KINDS: tuple[RepairKind, ...] = ("identity", "structural_counts")
ArmReport = dict[str, object]
PairArms = dict[RepairKind, ArmReport]


def _arm(
    task: PcgTask,
    *,
    seed: int,
    out_dir: Path,
    evaluations: int,
    repair_kind: RepairKind,
) -> ArmReport:
    dump = repair_kind == "structural_counts"
    filename = None
    if dump:
        filename = (
            f"{task.problem_name.replace('-', '_')}_bin_edges_structural_counts.json"
        )
    return run_problem(
        task,
        seed=seed,
        out_dir=out_dir,
        evaluations=evaluations,
        repair_kind=repair_kind,
        dump_edges=dump,
        edges_filename=filename,
        edges_stage="pcg_repair_pair",
    )


def _arm_summary(payload: ArmReport) -> dict[str, object]:
    gates = payload["gates"]  # type: ignore[index]
    return {
        "repair_kind": payload["repair_kind"],
        "selector_niche_jaccard": payload["selector_niche_jaccard"],
        "selector_jaccard_ok": gates["selector_jaccard"],  # type: ignore[index]
        "coverage_headroom": gates["coverage_headroom"],  # type: ignore[index]
        "random_playable": payload["random_playable"],
        "genetic_uniform_playable": payload["genetic_uniform"]["playable"],  # type: ignore[index]
        "genetic_min_fitness_playable": payload["genetic_min_fitness"]["playable"],  # type: ignore[index]
        "random_quality_min": payload["random_quality_min"],
        "random_quality_max": payload["random_quality_max"],
        "random_measures_min": payload["random_measures_min"],
        "random_measures_max": payload["random_measures_max"],
        "measure0_collapsed": payload["measure0_collapsed"],
        "random_astar_eligible": payload["random_astar_eligible"],
        "random_tiles_changed_mean": payload["random_tiles_changed_mean"],
        "repair_identity_gate": gates["repair_identity"],  # type: ignore[index]
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--evaluations", type=int, default=SMOKE_EVALUATIONS)
    parser.add_argument("--skip-zelda", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sokoban: PairArms = {}
    for kind in PAIR_KINDS:
        sokoban[kind] = _arm(
            SOKOBAN_V0,
            seed=RESERVED_PAIR_SOKOBAN_SEED,
            out_dir=args.output_dir,
            evaluations=args.evaluations,
            repair_kind=kind,
        )
    zelda: PairArms | None = None
    zelda_skipped = bool(args.skip_zelda)
    if not zelda_skipped:
        zelda = {}
        for kind in PAIR_KINDS:
            zelda[kind] = _arm(
                ZELDA_V0,
                seed=RESERVED_PAIR_ZELDA_SEED,
                out_dir=args.output_dir,
                evaluations=args.evaluations,
                repair_kind=kind,
            )
    report: dict[str, object] = {
        "stage": "pcg_repair_pair",
        "llm": False,
        "family": "pcg_benchmark",
        "one_family_not_two_public_tasks": True,
        "pair": list(PAIR_KINDS),
        "same_reserved_seeds_both_arms": True,
        "did_not_overwrite_pcg_smoke_or_identity_edges": True,
        "seeds": {
            "sokoban": RESERVED_PAIR_SOKOBAN_SEED,
            "zelda": None if zelda_skipped else RESERVED_PAIR_ZELDA_SEED,
        },
        "sokoban": sokoban,
        "zelda": zelda,
        "zelda_skipped": zelda_skipped,
        "summary": {
            "sokoban": {kind: _arm_summary(sokoban[kind]) for kind in PAIR_KINDS},
            "zelda": (
                None
                if zelda is None
                else {kind: _arm_summary(zelda[kind]) for kind in PAIR_KINDS}
            ),
        },
    }
    out = args.output_dir / "pcg_repair_pair.json"
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

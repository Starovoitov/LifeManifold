#!/usr/bin/env python3
"""PCG Benchmark pin + random/genetic smoke (no LLM, repair identity)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.pcg.pin_smoke import run_problem
from worldspace.pcg.smoke import (
    RESERVED_SOKOBAN_SEED,
    RESERVED_ZELDA_SEED,
    SMOKE_EVALUATIONS,
)
from worldspace.pcg.spec import SOKOBAN_V0, ZELDA_V0

DEFAULT_OUT = ROOT / "artifacts/controlled_attribution/pcg"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--evaluations", type=int, default=SMOKE_EVALUATIONS)
    parser.add_argument("--skip-zelda", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "stage": "pcg_smoke",
        "llm": False,
        "repair": "identity",
        "family": "pcg_benchmark",
        "one_family_not_two_public_tasks": True,
        "sokoban": run_problem(
            SOKOBAN_V0,
            seed=RESERVED_SOKOBAN_SEED,
            out_dir=args.output_dir,
            evaluations=args.evaluations,
            repair_kind="identity",
            dump_edges=True,
            edges_stage="pcg_smoke",
        ),
    }
    sokoban_crash_or_license = not all(
        report["sokoban"]["gates"][key]  # type: ignore[index]
        for key in ("pinned_env", "license")
    )
    if args.skip_zelda or sokoban_crash_or_license:
        report["zelda"] = None
        report["zelda_skipped"] = True
    else:
        report["zelda"] = run_problem(
            ZELDA_V0,
            seed=RESERVED_ZELDA_SEED,
            out_dir=args.output_dir,
            evaluations=args.evaluations,
            repair_kind="identity",
            dump_edges=True,
            edges_stage="pcg_smoke",
        )
        report["zelda_skipped"] = False
    out = args.output_dir / "pcg_smoke.json"
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "sokoban": report["sokoban"]["gates"],  # type: ignore[index]
                "zelda_skipped": report["zelda_skipped"],
            },
            indent=2,
        )
    )
    if report.get("zelda"):
        print(json.dumps({"zelda": report["zelda"]["gates"]}, indent=2))  # type: ignore[index]
    print(f"wrote {out}")
    sokoban_ok = all(report["sokoban"]["gates"].values())  # type: ignore[union-attr]
    zelda_ok = True
    if report.get("zelda"):
        zelda_ok = all(report["zelda"]["gates"].values())  # type: ignore[union-attr]
    return 0 if sokoban_ok and zelda_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

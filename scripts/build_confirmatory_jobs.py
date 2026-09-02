#!/usr/bin/env python3
"""Emit confirmatory selector/generator/allocation job plans without launching."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from worldspace.attribution.job_builder import write_job_plan
from worldspace.attribution.confirmatory_studies import (
    DomainId,
    allocation_study,
    build_confirmatory_job_plan,
    generator_study,
    selector_channel_study,
)

BUILDERS = {
    "selector-channel": selector_channel_study,
    "generator": generator_study,
    "allocation": allocation_study,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build confirmatory job plans for nas201 / pcg_sokoban. "
            "Never launches LLM or native search."
        )
    )
    parser.add_argument("--domain", choices=("nas201", "pcg_sokoban"), required=True)
    parser.add_argument(
        "--study",
        choices=("selector-channel", "generator", "allocation", "all"),
        default="all",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/attribution-jobs/confirmatory-draft"),
    )
    parser.add_argument(
        "--write-study-json",
        action="store_true",
        help="Also write study_manifest.json next to each job plan",
    )
    args = parser.parse_args(argv)

    studies = (
        ("selector-channel", "generator", "allocation")
        if args.study == "all"
        else (args.study,)
    )
    domain: DomainId = args.domain
    for study_kind in studies:
        study = BUILDERS[study_kind](domain)
        plan = build_confirmatory_job_plan(
            study, output_root=str(args.output_root / study.study_id)
        )
        if plan.launched:
            print("refusing to write a launched plan", file=sys.stderr)
            return 2
        out = args.output_root / study.study_id
        write_job_plan(plan, study_root=out)
        if args.write_study_json:
            out.mkdir(parents=True, exist_ok=True)
            (out / "study_manifest.json").write_text(
                study.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )
        print(
            json.dumps(
                {
                    "study_id": study.study_id,
                    "runs": len(plan.runs),
                    "launched": plan.launched,
                    "output": str(out),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

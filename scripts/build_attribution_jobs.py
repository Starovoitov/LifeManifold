#!/usr/bin/env python3
"""Write a frozen attribution job plan to disk without launching runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from worldspace.attribution import (
    InitialArchiveRef,
    JobBuildContext,
    StudyManifest,
    build_factorial_job_plan,
    current_domain_capabilities,
    write_job_plan,
)
from worldspace.attribution.hashing import canonical_sha256

DEFAULT_ARTIFACTS = (
    "run_manifest.json",
    "run_summary.json",
    "proposal_events.jsonl",
    "budget_ledger.jsonl",
)


def _context_from_study(
    study: StudyManifest,
    *,
    output_root: str,
    unit_monetary_cost: float,
) -> JobBuildContext:
    empty_hash = canonical_sha256([])
    archives = {
        instance_id: InitialArchiveRef(
            archive_id=f"empty-{instance_id}",
            archive_hash=empty_hash,
        )
        for instance_id in study.replication.domain_instance_ids
    }
    return JobBuildContext(
        output_root=output_root,
        initial_archives=archives,
        dependency_hashes={"lock": empty_hash},
        expected_artifacts=DEFAULT_ARTIFACTS,
        unit_monetary_cost=unit_monetary_cost,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Expand a study manifest into run manifests and a design matrix. "
            "Does not launch jobs."
        )
    )
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--unit-monetary-cost", type=float, default=0.0)
    args = parser.parse_args(argv)

    payload: dict[str, Any] = json.loads(args.study.read_text(encoding="utf-8"))
    study = StudyManifest.model_validate(payload)
    capabilities = current_domain_capabilities().get(study.domain_id)
    if capabilities is None:
        print(f"unknown domain_id {study.domain_id!r}", file=sys.stderr)
        return 2
    plan = build_factorial_job_plan(
        study,
        capabilities,
        _context_from_study(
            study,
            output_root=str(args.output_root),
            unit_monetary_cost=args.unit_monetary_cost,
        ),
    )
    if plan.launched:
        print("job builder marked the plan launched", file=sys.stderr)
        return 1
    study_root = write_job_plan(plan)
    print(f"Wrote unlaunched job plan to {study_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

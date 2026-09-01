#!/usr/bin/env python3
"""Isolated sokoban-v0 LLM with named structural_counts repair.

Prompt scan runs before any API call. Does not overwrite identity isolated
artifacts (pcg_isolated.json and the matching proposal/call-log files).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.generators.llm_call_log import configure_llm_call_log
from worldspace.pcg.descriptors import load_frozen_bin_edges
from worldspace.pcg.env import (
    PINNED_COMMIT,
    PINNED_LICENSE,
    PINNED_VERSION,
    BenchmarkPcgEnv,
)
from worldspace.pcg.isolated import (
    ISOLATED_PROPOSALS,
    RESERVED_REPAIR_ISOLATED_SEED,
    records_as_dicts,
    run_isolated_batch,
)
from worldspace.pcg.llm_emitter import DEFAULT_LLM_SPEC, PcgSokobanLlmEmitter
from worldspace.pcg.prompt_scan import SokobanPromptError, assert_prompt_templates
from worldspace.pcg.spec import SOKOBAN_V0

DEFAULT_EDGES = (
    ROOT
    / "artifacts/controlled_attribution/pcg/sokoban_v0_bin_edges_structural_counts.json"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts/controlled_attribution/pcg/pcg_isolated_structural_counts.json"
)
DEFAULT_PROPOSALS = (
    ROOT
    / "artifacts/controlled_attribution/pcg/pcg_isolated_structural_counts_proposals.jsonl"
)
DEFAULT_CALL_LOG = (
    ROOT
    / "artifacts/controlled_attribution/pcg/pcg_isolated_structural_counts_llm_call_log.jsonl"
)
REPAIR_KIND = "structural_counts"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    parser.add_argument("--llm-spec", type=Path, default=DEFAULT_LLM_SPEC)
    parser.add_argument("--n", type=int, default=ISOLATED_PROPOSALS)
    parser.add_argument("--seed", type=int, default=RESERVED_REPAIR_ISOLATED_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--call-log", type=Path, default=DEFAULT_CALL_LOG)
    parser.add_argument("--prompt-scan-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        templates = assert_prompt_templates()
    except SokobanPromptError as error:
        print(str(error), file=sys.stderr)
        return 3
    prompt_sha = hashlib.sha256(
        f"{templates['system']}\n---\n{templates['user']}".encode()
    ).hexdigest()
    print(
        json.dumps(
            {
                "prompt_scan": "pass",
                "prompt_sha256": prompt_sha,
                "repair": REPAIR_KIND,
                "parents": "structural_counts",
            },
            indent=2,
        )
    )
    if args.prompt_scan_only:
        return 0

    identity_outputs = {
        ROOT / "artifacts/controlled_attribution/pcg/pcg_isolated.json",
        ROOT / "artifacts/controlled_attribution/pcg/pcg_isolated_proposals.jsonl",
        ROOT / "artifacts/controlled_attribution/pcg/pcg_isolated_llm_call_log.jsonl",
    }
    if args.output.resolve() in {path.resolve() for path in identity_outputs}:
        print(
            "refusing to overwrite identity isolated artifacts",
            file=sys.stderr,
        )
        return 2

    edges = load_frozen_bin_edges(args.edges)
    if edges.problem_name != SOKOBAN_V0.problem_name:
        print(
            f"frozen edges are {edges.problem_name}, expected {SOKOBAN_V0.problem_name}",
            file=sys.stderr,
        )
        return 2

    configure_llm_call_log(args.call_log)
    env = BenchmarkPcgEnv(SOKOBAN_V0.problem_name, seed=args.seed)
    emitter = PcgSokobanLlmEmitter(llm_spec_path=args.llm_spec)
    records, summary = run_isolated_batch(
        env,
        edges,
        emitter,
        n_proposals=args.n,
        seed=args.seed,
        repair_kind=REPAIR_KIND,
    )
    report = {
        "stage": "pcg_isolated_structural_counts",
        "llm": True,
        "problem_name": SOKOBAN_V0.problem_name,
        "family": "pcg_benchmark",
        "one_family_not_two_public_tasks": True,
        "repair": REPAIR_KIND,
        "parents": "structural_counts",
        "few_shot": False,
        "pinned_commit": PINNED_COMMIT,
        "pinned_version": PINNED_VERSION,
        "pinned_license": PINNED_LICENSE,
        "reserved_seed": args.seed,
        "prompt_scan": "pass",
        "prompt_sha256": prompt_sha,
        "prompt_version": emitter.prompt_version,
        "llm_spec": _rel(args.llm_spec),
        "active_provider": emitter.config.active_provider,
        "requested_model": emitter.config.providers[emitter.config.active_provider].get(
            "model"
        ),
        "frozen_bin_path": _rel(args.edges),
        "frozen_bin_n_samples": edges.n_samples,
        "frozen_bin_stage": "pcg_repair_pair",
        "did_not_overwrite_identity_isolated": True,
        "selector_jaccard_not_reread": True,
        "zelda_llm": False,
        **summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with args.proposals.open("w", encoding="utf-8") as handle:
        for row in records_as_dicts(records):
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    print(json.dumps(report["gates"], indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {args.proposals}")
    return 0 if all(report["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

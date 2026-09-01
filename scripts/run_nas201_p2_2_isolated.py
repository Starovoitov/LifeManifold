#!/usr/bin/env python3
"""P2.2 isolated NAS LLM proposals. N4 prompt scan runs before any API call."""

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
from worldspace.nas201.descriptors import load_frozen_bin_edges
from worldspace.nas201.isolated import (
    ISOLATED_PROPOSALS,
    RESERVED_ISOLATED_SEED,
    records_as_dicts,
    run_isolated_batch,
)
from worldspace.nas201.llm_emitter import DEFAULT_LLM_SPEC, Nas201LlmEmitter
from worldspace.nas201.n4 import PromptN4Error, assert_n4_templates
from worldspace.nas201.table import CompactNas201Table

DEFAULT_JSONL = (
    ROOT / "artifacts/controlled_attribution/nas201/nas201_compact_cifar10_valid.jsonl"
)
DEFAULT_META = (
    ROOT
    / "artifacts/controlled_attribution/nas201/nas201_compact_cifar10_valid.meta.json"
)
DEFAULT_EDGES = ROOT / "artifacts/controlled_attribution/nas201/nas201_bin_edges.json"
DEFAULT_OUTPUT = ROOT / "artifacts/controlled_attribution/nas201/p2_2_isolated.json"
DEFAULT_PROPOSALS = (
    ROOT / "artifacts/controlled_attribution/nas201/p2_2_proposals.jsonl"
)
DEFAULT_CALL_LOG = (
    ROOT / "artifacts/controlled_attribution/nas201/p2_2_llm_call_log.jsonl"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    parser.add_argument("--llm-spec", type=Path, default=DEFAULT_LLM_SPEC)
    parser.add_argument("--n", type=int, default=ISOLATED_PROPOSALS)
    parser.add_argument("--seed", type=int, default=RESERVED_ISOLATED_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--call-log", type=Path, default=DEFAULT_CALL_LOG)
    parser.add_argument("--n4-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        templates = assert_n4_templates()
    except PromptN4Error as error:
        print(str(error), file=sys.stderr)
        return 3
    prompt_sha = hashlib.sha256(
        f"{templates['system']}\n---\n{templates['user']}".encode()
    ).hexdigest()
    print(json.dumps({"n4": "pass", "prompt_sha256": prompt_sha}, indent=2))
    if args.n4_only:
        return 0

    configure_llm_call_log(args.call_log)
    table = CompactNas201Table.from_jsonl(args.jsonl, args.meta)
    edges = load_frozen_bin_edges(args.edges)
    emitter = Nas201LlmEmitter(llm_spec_path=args.llm_spec)
    records, summary = run_isolated_batch(
        table,
        edges,
        emitter,
        n_proposals=args.n,
        seed=args.seed,
    )
    report = {
        "slice": "P2.2",
        "llm": True,
        "reserved_seed": args.seed,
        "n4": "pass",
        "prompt_sha256": prompt_sha,
        "prompt_version": emitter.prompt_version,
        "llm_spec": str(args.llm_spec),
        "active_provider": emitter.config.active_provider,
        "requested_model": emitter.config.providers[emitter.config.active_provider].get(
            "model"
        ),
        "frozen_bin_source_sha256": edges.source_sha256,
        "n_architectures": len(table),
        "search_dataset": table.meta.search_dataset,
        "contains_test_metrics": table.meta.contains_test_metrics,
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

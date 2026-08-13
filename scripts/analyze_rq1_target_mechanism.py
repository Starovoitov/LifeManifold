#!/usr/bin/env python3
"""Target-distribution, proposal-quality, and LLM-fidelity audit for clean RQ1."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldspace.illuminators.emitters.genetics import encode_world  # noqa: E402
from worldspace.specs.spec import WorldSpec  # noqa: E402

DEFAULT_ROOT = _ROOT / "artifacts" / "experiments" / "q1-rq1-contemporaneous"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalized_entropy(values: list[int]) -> float:
    counts = np.array(list(Counter(values).values()), dtype=float)
    if len(counts) <= 1:
        return 0.0
    probabilities = counts / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(len(counts))


def parent_child_edits(row: dict[str, Any]) -> tuple[float, float]:
    parent_raw = row.get("parent_world_spec")
    child_raw = row.get("world_spec")
    if not isinstance(parent_raw, dict) or not isinstance(child_raw, dict):
        return np.nan, np.nan
    parent = encode_world(WorldSpec.from_json_dict(parent_raw))
    child = encode_world(WorldSpec.from_json_dict(child_raw))
    hamming = float(np.count_nonzero(np.rint(parent[:18]) != np.rint(child[:18])))
    scalar_l1 = float(np.abs(parent[18:] - child[18:]).sum())
    return hamming, scalar_l1


def target_error(row: dict[str, Any]) -> float:
    target = row.get("target_bin")
    realized = row.get("realized_bin")
    if not isinstance(target, list) or not isinstance(realized, list):
        return np.nan
    delta = (np.asarray(realized, dtype=float) - np.asarray(target, dtype=float)) / 50.0
    return float(np.linalg.norm(delta))


def run_identity(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    return {
        "archive": int(manifest["archive_index"]),
        "continuation": int(manifest["continuation_index"]),
        "arm": str(manifest["arm"]),
    }


def summarize_group(
    identity: dict[str, Any],
    emitter: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    targets = [int(row["target_cell_id"]) for row in rows]
    counts = Counter(targets)
    edits = np.array([parent_child_edits(row) for row in rows], dtype=float)
    errors = np.array([target_error(row) for row in rows], dtype=float)
    consecutive_repeats = sum(
        left == right for left, right in zip(targets, targets[1:])
    )
    return {
        **identity,
        "emitter_type": emitter,
        "proposals": len(rows),
        "unique_targets": len(counts),
        "unique_target_fraction": len(counts) / len(rows) if rows else np.nan,
        "target_entropy_normalized": normalized_entropy(targets),
        "max_target_share": max(counts.values()) / len(rows) if rows else np.nan,
        "repeat_fraction": 1.0 - len(counts) / len(rows) if rows else np.nan,
        "consecutive_repeat_fraction": (
            consecutive_repeats / (len(rows) - 1) if len(rows) > 1 else np.nan
        ),
        "accepted_rate": np.mean([bool(row["accepted"]) for row in rows]),
        "improved_rate": np.mean([bool(row["improved"]) for row in rows]),
        "fill_empty_rate": np.mean(
            [row.get("outcome") == "fill_empty" for row in rows]
        ),
        "early_extinct_rate": np.mean([bool(row["early_extinct"]) for row in rows]),
        "target_error_mean": float(np.nanmean(errors)),
        "parent_child_rule_hamming_mean": (
            float(np.nanmean(edits[:, 0])) if np.isfinite(edits[:, 0]).any() else np.nan
        ),
        "parent_child_scalar_l1_mean": (
            float(np.nanmean(edits[:, 1])) if np.isfinite(edits[:, 1]).any() else np.nan
        ),
    }


def call_fidelity(
    identity: dict[str, Any],
    proposal_rows: list[dict[str, Any]],
    call_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    llm_proposals = [
        row for row in proposal_rows if str(row["emitter_type"]).startswith("llm")
    ]
    proposal_ids = {
        str(row["llm_call_id"]) for row in llm_proposals if row.get("llm_call_id")
    }
    call_ids = {str(row["call_id"]) for row in call_rows if row.get("call_id")}
    parse_counts = Counter(str(row.get("llm_parse_outcome")) for row in llm_proposals)
    response_models = sorted(
        {str(row["response_model"]) for row in call_rows if row.get("response_model")}
    )
    fingerprints = sorted(
        {
            str(row["system_fingerprint"])
            for row in call_rows
            if row.get("system_fingerprint")
        }
    )
    prompt_alignment: list[float] = []
    for row in llm_proposals:
        prompt = row.get("prompt_prediction")
        source = row.get("source_prediction")
        if isinstance(prompt, dict) and isinstance(source, dict):
            prompt_alignment.append(float(prompt["fitness"]) - float(source["fitness"]))
    return {
        **identity,
        "llm_proposals": len(llm_proposals),
        "llm_calls": len(call_rows),
        "proposal_call_ids": len(proposal_ids),
        "call_ids": len(call_ids),
        "proposal_ids_missing_call": len(proposal_ids - call_ids),
        "call_ids_missing_proposal": len(call_ids - proposal_ids),
        "request_failures": sum(not bool(row.get("ok")) for row in call_rows),
        "retried_calls": sum(int(row.get("attempts", 0)) > 1 for row in call_rows),
        "response_models": response_models,
        "system_fingerprints": fingerprints,
        "parse_outcomes": dict(sorted(parse_counts.items())),
        "prompt_minus_source_fitness_mean": (
            float(np.mean(prompt_alignment)) if prompt_alignment else np.nan
        ),
        "join_complete": proposal_ids == call_ids,
    }


def main() -> None:
    args = parse_args()
    mechanism_rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    run_dirs = sorted((args.root / "blocks").glob("archive_*/continuation_*/*"))
    for run_dir in run_dirs:
        proposal_path = run_dir / "proposal_log.jsonl"
        call_path = run_dir / "llm_call_log.jsonl"
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        if not proposal_path.is_file() or not call_path.is_file():
            raise FileNotFoundError(f"incomplete audit logs under {run_dir}")
        identity = run_identity(run_dir)
        proposals = read_jsonl(proposal_path)
        calls = read_jsonl(call_path)
        by_emitter: dict[str, list[dict[str, Any]]] = {}
        for row in proposals:
            by_emitter.setdefault(str(row["emitter_type"]), []).append(row)
        for emitter, rows in sorted(by_emitter.items()):
            mechanism_rows.append(summarize_group(identity, emitter, rows))
        fidelity_rows.append(call_fidelity(identity, proposals, calls))

    output = args.root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mechanism_rows).to_csv(
        output / "target_mechanism_by_emitter.csv",
        index=False,
    )
    (output / "llm_fidelity.json").write_text(
        json.dumps(fidelity_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if any(not bool(row["join_complete"]) for row in fidelity_rows):
        raise RuntimeError(
            "LLM call/proposal join is incomplete; inspect llm_fidelity.json"
        )


if __name__ == "__main__":
    main()

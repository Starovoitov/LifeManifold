"""P2.4 isolated LLM proposals for sokoban-v0 (no archive feedback, identity repair)."""

from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

from worldspace.pcg.copy_audit import copy_readme_example
from worldspace.pcg.descriptors import PcgBinEdges
from worldspace.pcg.emitters import random_spec
from worldspace.pcg.evaluation import PcgEnvLike, evaluate_spec
from worldspace.pcg.llm_emitter import PcgLlmProposal, PcgSokobanLlmEmitter
from worldspace.pcg.spec import SOKOBAN_V0

ISOLATED_PROPOSALS = 50
RESERVED_ISOLATED_SEED = 201_401
G3_MIN_SCHEMA_VALID = 0.50
G4_MAX_FALLBACK_AMONG_PARSE_VALID = 0.80

# Illustrative confirmatory horizons for G7 (not a protocol freeze).
G7_SEEDS = 20
G7_BLOCK_A_CELLS = 4
G7_EXAMPLE_HORIZONS = (50, 200, 500)


@dataclass(frozen=True)
class IsolatedProposalRecord:
    index: int
    parent_grid: list[list[int]]
    child_grid: list[list[int]]
    parent_hash: str
    child_hash: str
    emitter_type: str
    schema_valid: bool
    used_fallback: bool
    hamming: int
    exact_duplicate: bool
    copy_readme_example: bool
    fitness: float | None
    playable: bool
    measures: tuple[float, float] | None
    bin: tuple[int, int] | None
    miss_reason: str | None
    api_calls: int
    retries: int
    latency_ms: float
    eval_latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


def run_isolated_batch(
    env: PcgEnvLike,
    edges: PcgBinEdges,
    emitter: PcgSokobanLlmEmitter,
    *,
    n_proposals: int = ISOLATED_PROPOSALS,
    seed: int = RESERVED_ISOLATED_SEED,
) -> tuple[list[IsolatedProposalRecord], dict[str, object]]:
    rng = np.random.default_rng(seed)
    records: list[IsolatedProposalRecord] = []
    for index in range(n_proposals):
        parent = random_spec(SOKOBAN_V0, rng)
        proposal = emitter.emit(parent, rng, proposal_index=index)
        records.append(_record_from_proposal(proposal, env, edges, index))
        if (index + 1) % 10 == 0 or index + 1 == n_proposals:
            print(
                f"  isolated {index + 1}/{n_proposals} "
                f"schema_valid={sum(row.schema_valid for row in records)} "
                f"fallback={sum(row.used_fallback for row in records)} "
                f"copy_readme={sum(row.copy_readme_example for row in records)} "
                f"playable={sum(row.playable for row in records)}",
                file=sys.stderr,
                flush=True,
            )
    return records, summarize_isolated(records, emitter)


def _record_from_proposal(
    proposal: PcgLlmProposal,
    env: PcgEnvLike,
    edges: PcgBinEdges,
    index: int,
) -> IsolatedProposalRecord:
    started = time.perf_counter()
    evaluation = evaluate_spec(proposal.child, env, edges, SOKOBAN_V0)
    eval_latency_ms = (time.perf_counter() - started) * 1000.0
    return IsolatedProposalRecord(
        index=index,
        parent_grid=proposal.parent.to_nested_list(),
        child_grid=proposal.child.to_nested_list(),
        parent_hash=proposal.parent.genotype_sha256(),
        child_hash=proposal.child.genotype_sha256(),
        emitter_type=proposal.emitter_type,
        schema_valid=proposal.schema_valid,
        used_fallback=proposal.used_fallback,
        hamming=proposal.hamming,
        exact_duplicate=proposal.exact_duplicate,
        copy_readme_example=proposal.copy_readme_example
        or copy_readme_example(proposal.child),
        fitness=evaluation.fitness,
        playable=evaluation.playable,
        measures=evaluation.measures,
        bin=evaluation.bin,
        miss_reason=proposal.miss_reason or evaluation.miss_reason,
        api_calls=proposal.api_calls,
        retries=proposal.retries,
        latency_ms=proposal.latency_ms,
        eval_latency_ms=eval_latency_ms,
        prompt_tokens=proposal.prompt_tokens,
        completion_tokens=proposal.completion_tokens,
        total_tokens=proposal.total_tokens,
    )


def summarize_isolated(
    records: list[IsolatedProposalRecord],
    emitter: PcgSokobanLlmEmitter,
) -> dict[str, object]:
    n = len(records)
    parse_valid = [row for row in records if row.schema_valid]
    fallback_among_parse = [row for row in parse_valid if row.used_fallback]
    schema_valid_rate = len(parse_valid) / n if n else 0.0
    fallback_among_parse_rate = (
        len(fallback_among_parse) / len(parse_valid) if parse_valid else 1.0
    )
    mean_hamming = (
        sum(row.hamming for row in parse_valid) / len(parse_valid)
        if parse_valid
        else 0.0
    )
    duplicate_among_parse = (
        sum(row.exact_duplicate for row in parse_valid) / len(parse_valid)
        if parse_valid
        else 0.0
    )
    copy_among_parse = (
        sum(row.copy_readme_example for row in parse_valid) / len(parse_valid)
        if parse_valid
        else 0.0
    )
    playable_among_parse = (
        sum(row.playable for row in parse_valid) / len(parse_valid)
        if parse_valid
        else 0.0
    )
    fitness_rows = [row.fitness for row in parse_valid if row.fitness is not None]
    token_rows = [row for row in records if row.total_tokens is not None]
    latency = [row.latency_ms for row in records]
    eval_latency = [row.eval_latency_ms for row in records]
    mean_total_tokens = (
        sum(row.total_tokens or 0 for row in token_rows) / len(token_rows)
        if token_rows
        else None
    )
    mean_prompt_tokens = (
        sum(row.prompt_tokens or 0 for row in token_rows) / len(token_rows)
        if token_rows
        else None
    )
    mean_completion_tokens = (
        sum(row.completion_tokens or 0 for row in token_rows) / len(token_rows)
        if token_rows
        else None
    )
    mean_latency_ms = sum(latency) / len(latency) if latency else 0.0
    mean_eval_ms = sum(eval_latency) / len(eval_latency) if eval_latency else 0.0
    api_calls = sum(row.api_calls for row in records)
    g3 = schema_valid_rate >= G3_MIN_SCHEMA_VALID
    g4 = (
        bool(parse_valid)
        and fallback_among_parse_rate < G4_MAX_FALLBACK_AMONG_PARSE_VALID
        and mean_hamming > 0.0
    )
    g7 = {
        "block_a_seeds": G7_SEEDS,
        "block_a_cells": G7_BLOCK_A_CELLS,
        "horizon_not_frozen": True,
        "mean_total_tokens_per_proposal": mean_total_tokens,
        "mean_latency_ms_per_proposal": mean_latency_ms,
        "mean_eval_ms_per_proposal": mean_eval_ms,
        "api_calls_this_batch": api_calls,
        "illustrative_orders": {
            str(horizon): {
                "llm_calls": G7_SEEDS * G7_BLOCK_A_CELLS * horizon,
                "tokens": (
                    None
                    if mean_total_tokens is None
                    else G7_SEEDS * G7_BLOCK_A_CELLS * horizon * mean_total_tokens
                ),
                "wall_hours": (
                    G7_SEEDS
                    * G7_BLOCK_A_CELLS
                    * horizon
                    * mean_latency_ms
                    / 3_600_000.0
                ),
                "eval_hours": (
                    G7_SEEDS * G7_BLOCK_A_CELLS * horizon * mean_eval_ms / 3_600_000.0
                ),
            }
            for horizon in G7_EXAMPLE_HORIZONS
        },
        "monetary": None,
    }
    return {
        "n_proposals": n,
        "schema_valid": len(parse_valid),
        "schema_valid_rate": schema_valid_rate,
        "fallback": sum(row.used_fallback for row in records),
        "fallback_rate": sum(row.used_fallback for row in records) / n if n else 0.0,
        "fallback_among_parse_valid_rate": fallback_among_parse_rate,
        "mean_hamming_parse_valid": mean_hamming,
        "exact_duplicate_among_parse_valid_rate": duplicate_among_parse,
        "copy_readme_among_parse_valid_rate": copy_among_parse,
        "copy_readme": sum(row.copy_readme_example for row in records),
        "playable_among_parse_valid_rate": playable_among_parse,
        "playable": sum(row.playable for row in records),
        "fitness_min_parse_valid": min(fitness_rows) if fitness_rows else None,
        "fitness_max_parse_valid": max(fitness_rows) if fitness_rows else None,
        "mean_latency_ms": mean_latency_ms,
        "mean_eval_ms": mean_eval_ms,
        "mean_prompt_tokens": mean_prompt_tokens,
        "mean_completion_tokens": mean_completion_tokens,
        "mean_total_tokens": mean_total_tokens,
        "api_calls": api_calls,
        "prompt_version": emitter.prompt_version,
        "response_model": emitter.last_response_model,
        "emitter_audit": emitter.audit.to_dict(),
        "gates": {
            "G3_parse": g3,
            "G4_emitter_not_fallback": g4,
            "P7_zero_shot": True,
        },
        "g7_cost_order": g7,
    }


def records_as_dicts(records: list[IsolatedProposalRecord]) -> list[dict[str, object]]:
    rows = []
    for row in records:
        payload = asdict(row)
        if payload["measures"] is not None:
            payload["measures"] = list(payload["measures"])
        if payload["bin"] is not None:
            payload["bin"] = list(payload["bin"])
        rows.append(payload)
    return rows

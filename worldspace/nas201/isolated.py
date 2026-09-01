"""Isolated NAS LLM proposals (no archive feedback, no test metrics)."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass

import numpy as np

from worldspace.nas201.descriptors import Nas201BinEdges
from worldspace.nas201.emitters import random_spec
from worldspace.nas201.evaluation import evaluate_spec
from worldspace.nas201.llm_emitter import Nas201LlmEmitter, Nas201LlmProposal
from worldspace.nas201.table import Nas201Lookup

ISOLATED_PROPOSALS = 50
RESERVED_ISOLATED_SEED = 201_101
MIN_SCHEMA_VALID_RATE = 0.50
MAX_FALLBACK_AMONG_PARSE_VALID = 0.80

# Illustrative confirmatory cost-order horizons (not a protocol freeze).
COST_ORDER_SEEDS = 20
COST_ORDER_CELLS = 4
COST_ORDER_HORIZONS = (50, 200, 500)


@dataclass(frozen=True)
class IsolatedProposalRecord:
    index: int
    parent_ops: list[str]
    child_ops: list[str]
    parent_arch: str
    child_arch: str
    emitter_type: str
    schema_valid: bool
    used_fallback: bool
    hamming: int
    exact_duplicate: bool
    lookup_hit: bool
    fitness: float | None
    architecture_index: int | None
    miss_reason: str | None
    api_calls: int
    retries: int
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


def run_isolated_batch(
    table: Nas201Lookup,
    edges: Nas201BinEdges,
    emitter: Nas201LlmEmitter,
    *,
    n_proposals: int = ISOLATED_PROPOSALS,
    seed: int = RESERVED_ISOLATED_SEED,
) -> tuple[list[IsolatedProposalRecord], dict[str, object]]:
    rng = np.random.default_rng(seed)
    records: list[IsolatedProposalRecord] = []
    for index in range(n_proposals):
        parent = random_spec(rng)
        proposal = emitter.emit(parent, rng, proposal_index=index)
        records.append(_record_from_proposal(proposal, table, edges, index))
        if (index + 1) % 10 == 0 or index + 1 == n_proposals:
            print(
                f"  isolated {index + 1}/{n_proposals} "
                f"schema_valid={sum(row.schema_valid for row in records)} "
                f"fallback={sum(row.used_fallback for row in records)}",
                file=sys.stderr,
                flush=True,
            )
    return records, summarize_isolated(records, emitter)


def _record_from_proposal(
    proposal: Nas201LlmProposal,
    table: Nas201Lookup,
    edges: Nas201BinEdges,
    index: int,
) -> IsolatedProposalRecord:
    evaluation = evaluate_spec(proposal.child, table, edges)
    return IsolatedProposalRecord(
        index=index,
        parent_ops=list(proposal.parent.ops),
        child_ops=list(proposal.child.ops),
        parent_arch=proposal.parent.arch_str,
        child_arch=proposal.child.arch_str,
        emitter_type=proposal.emitter_type,
        schema_valid=proposal.schema_valid,
        used_fallback=proposal.used_fallback,
        hamming=proposal.hamming,
        exact_duplicate=proposal.exact_duplicate,
        lookup_hit=evaluation.lookup_hit,
        fitness=evaluation.fitness,
        architecture_index=(
            None if evaluation.record is None else evaluation.record.index
        ),
        miss_reason=proposal.miss_reason or evaluation.miss_reason,
        api_calls=proposal.api_calls,
        retries=proposal.retries,
        latency_ms=proposal.latency_ms,
        prompt_tokens=proposal.prompt_tokens,
        completion_tokens=proposal.completion_tokens,
        total_tokens=proposal.total_tokens,
    )


def summarize_isolated(
    records: list[IsolatedProposalRecord],
    emitter: Nas201LlmEmitter,
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
    lookup_among_parse = (
        sum(row.lookup_hit for row in parse_valid) / len(parse_valid)
        if parse_valid
        else 0.0
    )
    duplicate_among_parse = (
        sum(row.exact_duplicate for row in parse_valid) / len(parse_valid)
        if parse_valid
        else 0.0
    )
    token_rows = [row for row in records if row.total_tokens is not None]
    latency = [row.latency_ms for row in records]
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
    api_calls = sum(row.api_calls for row in records)
    parse_ok = schema_valid_rate >= MIN_SCHEMA_VALID_RATE
    emitter_ok = (
        bool(parse_valid)
        and fallback_among_parse_rate < MAX_FALLBACK_AMONG_PARSE_VALID
        and mean_hamming > 0.0
    )
    cost_order = {
        "block_a_seeds": COST_ORDER_SEEDS,
        "block_a_cells": COST_ORDER_CELLS,
        "horizon_not_frozen": True,
        "mean_total_tokens_per_proposal": mean_total_tokens,
        "mean_latency_ms_per_proposal": mean_latency_ms,
        "api_calls_this_batch": api_calls,
        "illustrative_orders": {
            str(horizon): {
                "llm_calls": COST_ORDER_SEEDS * COST_ORDER_CELLS * horizon,
                "tokens": (
                    None
                    if mean_total_tokens is None
                    else COST_ORDER_SEEDS
                    * COST_ORDER_CELLS
                    * horizon
                    * mean_total_tokens
                ),
                "wall_hours": (
                    COST_ORDER_SEEDS
                    * COST_ORDER_CELLS
                    * horizon
                    * mean_latency_ms
                    / 3_600_000.0
                ),
            }
            for horizon in COST_ORDER_HORIZONS
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
        "lookup_hit_among_parse_valid_rate": lookup_among_parse,
        "mean_latency_ms": mean_latency_ms,
        "mean_prompt_tokens": mean_prompt_tokens,
        "mean_completion_tokens": mean_completion_tokens,
        "mean_total_tokens": mean_total_tokens,
        "api_calls": api_calls,
        "prompt_version": emitter.prompt_version,
        "emitter_audit": emitter.audit.to_dict(),
        "gates": {
            "parse": parse_ok,
            "emitter_not_fallback": emitter_ok,
        },
        "cost_order": cost_order,
    }


def records_as_dicts(records: list[IsolatedProposalRecord]) -> list[dict[str, object]]:
    return [asdict(row) for row in records]

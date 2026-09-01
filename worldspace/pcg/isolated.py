"""Isolated LLM proposals for sokoban-v0 (no archive feedback).

Repair is a named factor. Default identity matches the identity isolated
batch. structural_counts repairs count-valid parents and post-emit children;
Hamming among parse-valid proposals is measured after repair.
"""

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
from worldspace.pcg.repair import RepairKind, apply_repair, sokoban_astar_eligible
from worldspace.pcg.spec import SOKOBAN_V0, hamming_tiles

ISOLATED_PROPOSALS = 50
RESERVED_ISOLATED_SEED = 201_401
RESERVED_REPAIR_ISOLATED_SEED = 201_601
MIN_SCHEMA_VALID_RATE = 0.50
MAX_FALLBACK_AMONG_PARSE_VALID = 0.80

# Illustrative confirmatory cost-order horizons (not a protocol freeze).
COST_ORDER_SEEDS = 20
COST_ORDER_CELLS = 4
COST_ORDER_HORIZONS = (50, 200, 500)


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
    repair_kind: RepairKind
    hamming_before_repair: int
    tiles_changed_by_repair: int
    astar_eligible: bool | None


def run_isolated_batch(
    env: PcgEnvLike,
    edges: PcgBinEdges,
    emitter: PcgSokobanLlmEmitter,
    *,
    n_proposals: int = ISOLATED_PROPOSALS,
    seed: int = RESERVED_ISOLATED_SEED,
    repair_kind: RepairKind = "identity",
) -> tuple[list[IsolatedProposalRecord], dict[str, object]]:
    rng = np.random.default_rng(seed)
    records: list[IsolatedProposalRecord] = []
    for index in range(n_proposals):
        parent = random_spec(SOKOBAN_V0, rng)
        if repair_kind != "identity":
            parent, _meta = apply_repair(parent, repair_kind)
        proposal = emitter.emit(parent, rng, proposal_index=index)
        records.append(
            _record_from_proposal(proposal, env, edges, index, repair_kind=repair_kind)
        )
        if (index + 1) % 10 == 0 or index + 1 == n_proposals:
            print(
                f"  isolated {index + 1}/{n_proposals} repair={repair_kind} "
                f"schema_valid={sum(row.schema_valid for row in records)} "
                f"fallback={sum(row.used_fallback for row in records)} "
                f"copy_readme={sum(row.copy_readme_example for row in records)} "
                f"playable={sum(row.playable for row in records)}",
                file=sys.stderr,
                flush=True,
            )
    return records, summarize_isolated(records, emitter, repair_kind=repair_kind)


def _record_from_proposal(
    proposal: PcgLlmProposal,
    env: PcgEnvLike,
    edges: PcgBinEdges,
    index: int,
    *,
    repair_kind: RepairKind = "identity",
) -> IsolatedProposalRecord:
    child, meta = apply_repair(proposal.child, repair_kind)
    hamming_after = hamming_tiles(proposal.parent, child)
    started = time.perf_counter()
    evaluation = evaluate_spec(child, env, edges, SOKOBAN_V0)
    eval_latency_ms = (time.perf_counter() - started) * 1000.0
    copied = proposal.copy_readme_example or copy_readme_example(child)
    astar_eligible = meta.get("astar_eligible")
    eligible = None if astar_eligible is None else bool(astar_eligible)
    if eligible is None and child.problem_name == SOKOBAN_V0.problem_name:
        eligible = sokoban_astar_eligible(child.to_nested_list())
    return IsolatedProposalRecord(
        index=index,
        parent_grid=proposal.parent.to_nested_list(),
        child_grid=child.to_nested_list(),
        parent_hash=proposal.parent.genotype_sha256(),
        child_hash=child.genotype_sha256(),
        emitter_type=proposal.emitter_type,
        schema_valid=proposal.schema_valid,
        used_fallback=proposal.used_fallback,
        hamming=hamming_after,
        exact_duplicate=hamming_after == 0,
        copy_readme_example=copied,
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
        repair_kind=repair_kind,
        hamming_before_repair=proposal.hamming,
        tiles_changed_by_repair=int(meta["tiles_changed"]),
        astar_eligible=eligible,
    )


def summarize_isolated(
    records: list[IsolatedProposalRecord],
    emitter: PcgSokobanLlmEmitter,
    *,
    repair_kind: RepairKind = "identity",
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
    parse_ok = schema_valid_rate >= MIN_SCHEMA_VALID_RATE
    emitter_ok = (
        bool(parse_valid)
        and fallback_among_parse_rate < MAX_FALLBACK_AMONG_PARSE_VALID
        and mean_hamming > 0.0
    )
    mean_hamming_before = (
        sum(row.hamming_before_repair for row in parse_valid) / len(parse_valid)
        if parse_valid
        else 0.0
    )
    mean_repair_tiles = (
        sum(row.tiles_changed_by_repair for row in records) / n if n else 0.0
    )
    astar_eligible = sum(bool(row.astar_eligible) for row in records)
    measure0 = [row.measures[0] for row in records if row.measures is not None]
    cost_order = {
        "block_a_seeds": COST_ORDER_SEEDS,
        "block_a_cells": COST_ORDER_CELLS,
        "horizon_not_frozen": True,
        "mean_total_tokens_per_proposal": mean_total_tokens,
        "mean_latency_ms_per_proposal": mean_latency_ms,
        "mean_eval_ms_per_proposal": mean_eval_ms,
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
                "eval_hours": (
                    COST_ORDER_SEEDS
                    * COST_ORDER_CELLS
                    * horizon
                    * mean_eval_ms
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
        "mean_hamming_before_repair_parse_valid": mean_hamming_before,
        "mean_tiles_changed_by_repair": mean_repair_tiles,
        "astar_eligible": astar_eligible,
        "measure0_min": min(measure0) if measure0 else None,
        "measure0_max": max(measure0) if measure0 else None,
        "measure0_collapsed": (bool(measure0) and min(measure0) == max(measure0)),
        "repair_kind": repair_kind,
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
            "parse": parse_ok,
            "emitter_not_fallback": emitter_ok,
            "zero_shot_prompt": True,
        },
        "cost_order": cost_order,
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

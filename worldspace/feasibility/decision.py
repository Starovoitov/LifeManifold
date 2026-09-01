"""GO / REVISE / DROP from frozen feasibility thresholds and smoke/isolated reports.

Does not re-run evaluators or LLM. Does not change the frozen numeric thresholds.
"""

from __future__ import annotations

from typing import Any

from worldspace.nas201.isolated import (
    COST_ORDER_CELLS,
    COST_ORDER_HORIZONS,
    COST_ORDER_SEEDS,
    MAX_FALLBACK_AMONG_PARSE_VALID,
    MIN_SCHEMA_VALID_RATE,
)
from worldspace.pcg.smoke import MAX_SELECTOR_JACCARD, MAX_SMOKE_COVERAGE

FROZEN_ON = "2026-09-01"
STAGE = "feasibility_decision"

# Identity-repair quality≈0 is REVISE (repair as a named factor), not a silent
# GO and not a new numeric playable-fraction threshold.
QUALITY_NEAR_ZERO_REVISE = True

NEXT_STAGE_REPLACEMENT_FAMILY = "replacement_family"
NEXT_STAGE_PCG_REVISE = "pcg_revise"


def _parse_ok(report: dict[str, Any]) -> bool:
    return float(report["schema_valid_rate"]) >= MIN_SCHEMA_VALID_RATE


def _emitter_ok(report: dict[str, Any]) -> bool:
    return bool(
        int(report["schema_valid"]) > 0
        and float(report["fallback_among_parse_valid_rate"])
        < MAX_FALLBACK_AMONG_PARSE_VALID
        and float(report["mean_hamming_parse_valid"]) > 0.0
    )


def _cost_order_from_isolated(isolated: dict[str, Any]) -> dict[str, Any]:
    raw = isolated.get("cost_order")
    if not isinstance(raw, dict):
        raise TypeError("isolated report must include a cost_order object")
    return {
        "block_a_seeds": COST_ORDER_SEEDS,
        "block_a_cells": COST_ORDER_CELLS,
        "horizons_not_frozen": list(COST_ORDER_HORIZONS),
        "mean_total_tokens_per_proposal": raw.get("mean_total_tokens_per_proposal"),
        "mean_latency_ms_per_proposal": raw.get("mean_latency_ms_per_proposal"),
        "mean_eval_ms_per_proposal": raw.get("mean_eval_ms_per_proposal"),
        "illustrative_orders": raw.get("illustrative_orders"),
        "monetary": None,
        "declared_cap": None,
        "pass": True,
        "decision": "GO",
        "rule": "not_obviously_unaffordable_vs_future_cap",
        "note": (
            "A later confirmatory protocol must declare the cap. Monetary stays "
            "null until a dated price table. Evaluator cost is lookup (NAS) or "
            "CPU A* (PCG); PCG eval_ms here is a lower bound because "
            "identity-repair levels were unsolved."
        ),
    }


def decide_nas(
    nas_smoke: dict[str, Any],
    nas_isolated: dict[str, Any],
) -> dict[str, Any]:
    smoke_gates = dict(nas_smoke["gates"])
    parse_ok = _parse_ok(nas_isolated)
    emitter_ok = _emitter_ok(nas_isolated)
    selector_ok = bool(smoke_gates["selector_jaccard"])
    coverage_ok = bool(smoke_gates["coverage_headroom"])
    occupancy = dict(nas_smoke["occupancy"])
    cost_order = _cost_order_from_isolated(nas_isolated)
    gates = {
        "license": True,
        "deterministic": True,
        "parse": parse_ok,
        "emitter_not_fallback": emitter_ok,
        "selector_jaccard": selector_ok,
        "coverage_headroom": coverage_ok,
        "cost": True,
        "no_test_feedback": True,
        "full_lookup": bool(smoke_gates["full_lookup"]),
        "unique_canonical_hash": bool(smoke_gates["unique_canonical_hash"]),
        "search_split_only": bool(smoke_gates["search_split_only"]),
        "prompt_scan": nas_isolated.get("prompt_scan") == "pass",
        "no_bin_over_half": bool(smoke_gates["no_bin_over_half"]),
    }
    go = all(gates.values())
    return {
        "family": "nas201",
        "task": "nas-bench-201",
        "decision": "GO" if go else "REVISE",
        "gates": gates,
        "selector_jaccard": None,
        "coverage_uniform": nas_smoke["genetic_uniform"]["coverage"],
        "coverage_min_fitness": nas_smoke["genetic_min_fitness"]["coverage"],
        "full_table_occupied_bins": occupancy["occupied_bins"],
        "full_table_bins": 400,
        "max_bin_fraction": occupancy["max_bin_fraction"],
        "coverage_note": (
            "Full table occupies 21/400 bins (params/FLOPs correlated). "
            "Coverage still passes the frozen <0.95 rule. Not a silent "
            "op-histogram switch."
        ),
        "cost_order": cost_order,
        "schema_valid_rate": nas_isolated["schema_valid_rate"],
        "fallback_among_parse_valid_rate": nas_isolated[
            "fallback_among_parse_valid_rate"
        ],
        "mean_hamming_parse_valid": nas_isolated["mean_hamming_parse_valid"],
        "repair": "identity",
    }


def decide_pcg_sokoban(
    pcg_smoke: dict[str, Any],
    pcg_isolated: dict[str, Any],
) -> dict[str, Any]:
    sokoban = pcg_smoke["sokoban"]
    smoke_gates = dict(sokoban["gates"])
    parse_ok = _parse_ok(pcg_isolated)
    emitter_ok = _emitter_ok(pcg_isolated)
    selector_ok = bool(smoke_gates["selector_jaccard"])
    coverage_ok = bool(smoke_gates["coverage_headroom"])
    playable = (
        int(sokoban["random_playable"])
        + int(sokoban["genetic_uniform"]["playable"])
        + int(sokoban["genetic_min_fitness"]["playable"])
        + int(pcg_isolated["playable"])
    )
    cost_order = _cost_order_from_isolated(pcg_isolated)
    domain_gates = {
        "pinned_env": bool(smoke_gates["pinned_env"]),
        "quality_is_fitness": bool(smoke_gates["quality_is_fitness"]),
        "info_keys": bool(smoke_gates["info_keys"]),
        "repair_identity": bool(smoke_gates["repair_identity"]),
        "zero_shot_prompt": bool(pcg_isolated["gates"]["zero_shot_prompt"])
        and pcg_isolated.get("prompt_scan") == "pass",
        "invalid_skips_search": bool(smoke_gates["invalid_skips_search"]),
    }
    gates = {
        "license": bool(smoke_gates["license"]),
        "deterministic": bool(smoke_gates["deterministic"]),
        "parse": parse_ok,
        "emitter_not_fallback": emitter_ok,
        "selector_jaccard": selector_ok,
        "coverage_headroom": coverage_ok,
        "cost": True,
        "no_test_feedback": bool(smoke_gates["no_holdout_in_quality"]),
        **domain_gates,
    }
    quality_near_zero = playable == 0
    reasons: list[str] = []
    if not selector_ok:
        reasons.append("selector_jaccard_at_least_0.80")
    if quality_near_zero and QUALITY_NEAR_ZERO_REVISE:
        reasons.append("identity_repair_playable_zero")
    decision = "GO" if (all(gates.values()) and not reasons) else "REVISE"
    if not gates["license"]:
        decision = "DROP"
    return {
        "family": "pcg_benchmark",
        "task": "sokoban-v0",
        "decision": decision,
        "gates": gates,
        "selector_jaccard": sokoban["selector_niche_jaccard"],
        "coverage_uniform": sokoban["genetic_uniform"]["coverage"],
        "coverage_min_fitness": sokoban["genetic_min_fitness"]["coverage"],
        "playable_total_observed": playable,
        "quality_near_zero": quality_near_zero,
        "revise_reasons": reasons,
        "cost_order": cost_order,
        "schema_valid_rate": pcg_isolated["schema_valid_rate"],
        "fallback_among_parse_valid_rate": pcg_isolated[
            "fallback_among_parse_valid_rate"
        ],
        "mean_hamming_parse_valid": pcg_isolated["mean_hamming_parse_valid"],
        "copy_readme": pcg_isolated["copy_readme"],
        "repair": "identity",
        "solution_length_collapsed": True,
    }


def decide_pcg_zelda(pcg_smoke: dict[str, Any]) -> dict[str, Any]:
    zelda = pcg_smoke["zelda"]
    smoke_gates = dict(zelda["gates"])
    playable = (
        int(zelda["random_playable"])
        + int(zelda["genetic_uniform"]["playable"])
        + int(zelda["genetic_min_fitness"]["playable"])
    )
    quality_near_zero = playable == 0
    reasons: list[str] = []
    if quality_near_zero and QUALITY_NEAR_ZERO_REVISE:
        reasons.append("identity_repair_playable_zero")
    reasons.append("parse_and_emitter_unread_no_zelda_llm")
    gates = {
        "license": bool(smoke_gates["license"]),
        "deterministic": bool(smoke_gates["deterministic"]),
        "parse": None,
        "emitter_not_fallback": None,
        "selector_jaccard": bool(smoke_gates["selector_jaccard"]),
        "coverage_headroom": bool(smoke_gates["coverage_headroom"]),
        "cost": None,
        "no_test_feedback": bool(smoke_gates["no_holdout_in_quality"]),
        "pinned_env": bool(smoke_gates["pinned_env"]),
        "quality_is_fitness": bool(smoke_gates["quality_is_fitness"]),
        "info_keys": bool(smoke_gates["info_keys"]),
        "repair_identity": bool(smoke_gates["repair_identity"]),
        "invalid_skips_search": bool(smoke_gates["invalid_skips_search"]),
    }
    return {
        "family": "pcg_benchmark",
        "task": "zelda-v0",
        "decision": "REVISE",
        "not_second_public_family": True,
        "gates": gates,
        "selector_jaccard": zelda["selector_niche_jaccard"],
        "coverage_uniform": zelda["genetic_uniform"]["coverage"],
        "coverage_min_fitness": zelda["genetic_min_fitness"]["coverage"],
        "playable_total_observed": playable,
        "quality_near_zero": quality_near_zero,
        "revise_reasons": reasons,
        "repair": "identity",
        "llm_batch": False,
    }


def decide_pcg_family(
    sokoban: dict[str, Any],
    zelda: dict[str, Any],
) -> dict[str, Any]:
    task_decisions = {sokoban["decision"], zelda["decision"]}
    if "DROP" in task_decisions and "GO" not in task_decisions:
        family = "DROP"
    elif "GO" in task_decisions:
        family = "GO"
    else:
        family = "REVISE"
    return {
        "family": "pcg_benchmark",
        "one_family_not_two_public_tasks": True,
        "decision": family,
        "go_requires_one_task_go": True,
        "tasks": {
            "sokoban-v0": sokoban["decision"],
            "zelda-v0": zelda["decision"],
        },
        "zelda_is_not_a_second_family": True,
        "replacement_family_not_needed": family != "DROP",
    }


def decide_shortlist(
    nas: dict[str, Any],
    pcg_family: dict[str, Any],
) -> dict[str, Any]:
    eligible = []
    pending = []
    dropped = []
    if nas["decision"] == "GO":
        eligible.append("nas201")
    elif nas["decision"] == "DROP":
        dropped.append("nas201")
    else:
        pending.append("nas201")
    if pcg_family["decision"] == "GO":
        eligible.append("pcg_benchmark")
    elif pcg_family["decision"] == "DROP":
        dropped.append("pcg_benchmark")
    else:
        pending.append("pcg_benchmark")
    two_tasks_frozen = len(eligible) >= 2 and not pending
    needs_replacement_family = (
        nas["decision"] == "DROP" or pcg_family["decision"] == "DROP"
    )
    if needs_replacement_family:
        next_stage = NEXT_STAGE_REPLACEMENT_FAMILY
        next_stage_is = (
            "DROP family replacement: Feynman or Sodarace feasibility design; "
            "not Zelda as a second public family"
        )
    else:
        next_stage = NEXT_STAGE_PCG_REVISE
        next_stage_is = (
            "dated PCG REVISE: repair as an explicit treatment factor; "
            "re-read genetic selector overlap and coverage; "
            "no silent bin-threshold move; "
            "no Feynman/Sodarace substitution"
        )
    return {
        "confirmatory_tasks_frozen": False,
        "two_public_tasks_selected": two_tasks_frozen,
        "eligible_families": eligible,
        "pending_revise_families": pending,
        "dropped_families": dropped,
        "intended_pair_if_pcg_goes": ["nas201", "pcg_benchmark"],
        "not_sokoban_plus_zelda_as_two_families": True,
        "confirmatory_protocol_may_start": False,
        "needs_replacement_family": needs_replacement_family,
        "next_stage": next_stage,
        "next_stage_is": next_stage_is,
    }


def build_decision(
    *,
    nas_smoke: dict[str, Any],
    nas_isolated: dict[str, Any],
    pcg_smoke: dict[str, Any],
    pcg_isolated: dict[str, Any],
) -> dict[str, Any]:
    if nas_isolated["gates"]["parse"] != _parse_ok(nas_isolated):
        raise ValueError(
            "NAS isolated stored parse gate disagrees with frozen threshold"
        )
    if nas_isolated["gates"]["emitter_not_fallback"] != _emitter_ok(nas_isolated):
        raise ValueError(
            "NAS isolated stored emitter gate disagrees with frozen threshold"
        )
    if pcg_isolated["gates"]["parse"] != _parse_ok(pcg_isolated):
        raise ValueError(
            "PCG isolated stored parse gate disagrees with frozen threshold"
        )
    if pcg_isolated["gates"]["emitter_not_fallback"] != _emitter_ok(pcg_isolated):
        raise ValueError(
            "PCG isolated stored emitter gate disagrees with frozen threshold"
        )
    nas = decide_nas(nas_smoke, nas_isolated)
    sokoban = decide_pcg_sokoban(pcg_smoke, pcg_isolated)
    zelda = decide_pcg_zelda(pcg_smoke)
    pcg = decide_pcg_family(sokoban, zelda)
    shortlist = decide_shortlist(nas, pcg)
    return {
        "stage": STAGE,
        "llm": False,
        "evidence_role": "design_data",
        "thresholds_frozen_on": FROZEN_ON,
        "thresholds": {
            "min_schema_valid_rate": MIN_SCHEMA_VALID_RATE,
            "max_fallback_among_parse_valid": MAX_FALLBACK_AMONG_PARSE_VALID,
            "max_selector_jaccard": MAX_SELECTOR_JACCARD,
            "max_smoke_coverage": MAX_SMOKE_COVERAGE,
            "cost_order_seeds": COST_ORDER_SEEDS,
            "cost_order_cells": COST_ORDER_CELLS,
            "cost_order_monetary": None,
            "quality_near_zero_revise": QUALITY_NEAR_ZERO_REVISE,
        },
        "nas201": nas,
        "pcg_sokoban": sokoban,
        "pcg_zelda": zelda,
        "pcg_family": pcg,
        "shortlist": shortlist,
        "sidecar_adapter": False,
        "repair_not_silently_enabled": True,
    }

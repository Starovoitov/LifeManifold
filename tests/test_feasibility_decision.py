"""Feasibility decisions use frozen thresholds and do not rewrite them after the fact."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from worldspace.feasibility.decision import (
    NEXT_STAGE_PCG_REVISE,
    NEXT_STAGE_REPLACEMENT_FAMILY,
    build_decision,
    decide_pcg_family,
    decide_shortlist,
)
from worldspace.nas201.isolated import MIN_SCHEMA_VALID_RATE
from worldspace.pcg.smoke import MAX_SELECTOR_JACCARD, MAX_SMOKE_COVERAGE

ROOT = Path(__file__).resolve().parents[1]
_ART = ROOT / "artifacts/controlled_attribution"
NAS_SMOKE = _ART / "nas201/nas201_lookup_smoke.json"
NAS_ISOLATED = _ART / "nas201/nas201_isolated.json"
PCG_SMOKE = _ART / "pcg/pcg_smoke.json"
PCG_ISOLATED = _ART / "pcg/pcg_isolated.json"


def _nas_smoke() -> dict[str, Any]:
    return {
        "gates": {
            "full_lookup": True,
            "unique_canonical_hash": True,
            "search_split_only": True,
            "no_bin_over_half": True,
            "selector_jaccard": True,
            "coverage_headroom": True,
        },
        "occupancy": {"occupied_bins": 21, "max_bin_fraction": 0.12},
        "genetic_uniform": {"coverage": 0.10},
        "genetic_min_fitness": {"coverage": 0.11},
    }


def _isolated(
    *,
    schema_valid_rate: float = 0.90,
    schema_valid: int = 45,
    fallback_among_parse_valid_rate: float = 0.10,
    mean_hamming_parse_valid: float = 1.2,
    prompt_scan: str = "pass",
    playable: int = 0,
    copy_readme: int = 0,
    zero_shot_prompt: bool = True,
) -> dict[str, Any]:
    parse_ok = schema_valid_rate >= MIN_SCHEMA_VALID_RATE
    emitter_ok = (
        schema_valid > 0
        and fallback_among_parse_valid_rate < 0.80
        and mean_hamming_parse_valid > 0.0
    )
    return {
        "schema_valid_rate": schema_valid_rate,
        "schema_valid": schema_valid,
        "fallback_among_parse_valid_rate": fallback_among_parse_valid_rate,
        "mean_hamming_parse_valid": mean_hamming_parse_valid,
        "prompt_scan": prompt_scan,
        "playable": playable,
        "copy_readme": copy_readme,
        "gates": {
            "parse": parse_ok,
            "emitter_not_fallback": emitter_ok,
            "zero_shot_prompt": zero_shot_prompt,
        },
        "cost_order": {
            "mean_total_tokens_per_proposal": 120.0,
            "mean_latency_ms_per_proposal": 40.0,
            "mean_eval_ms_per_proposal": 5.0,
            "illustrative_orders": {},
        },
    }


def _pcg_task(
    *,
    selector_jaccard_ok: bool,
    selector_niche_jaccard: float,
    coverage: float = 0.20,
    playable: int = 0,
    license_ok: bool = True,
) -> dict[str, Any]:
    return {
        "gates": {
            "pinned_env": True,
            "quality_is_fitness": True,
            "info_keys": True,
            "repair_identity": True,
            "invalid_skips_search": True,
            "license": license_ok,
            "deterministic": True,
            "selector_jaccard": selector_jaccard_ok,
            "coverage_headroom": True,
            "no_holdout_in_quality": True,
        },
        "selector_niche_jaccard": selector_niche_jaccard,
        "random_playable": playable,
        "genetic_uniform": {"coverage": coverage, "playable": 0},
        "genetic_min_fitness": {"coverage": coverage, "playable": 0},
    }


def _pcg_smoke() -> dict[str, Any]:
    return {
        "sokoban": _pcg_task(
            selector_jaccard_ok=False,
            selector_niche_jaccard=0.91,
        ),
        "zelda": _pcg_task(
            selector_jaccard_ok=True,
            selector_niche_jaccard=0.42,
        ),
    }


class TestFrozenThresholds(unittest.TestCase):
    def test_parse_selector_and_coverage_thresholds(self) -> None:
        self.assertEqual(MIN_SCHEMA_VALID_RATE, 0.50)
        self.assertEqual(MAX_SELECTOR_JACCARD, 0.80)
        self.assertEqual(MAX_SMOKE_COVERAGE, 0.95)


class TestFeasibilityDecisionFromReports(unittest.TestCase):
    def test_nas_go_pcg_revise_no_replacement_family(self) -> None:
        decision = build_decision(
            nas_smoke=_nas_smoke(),
            nas_isolated=_isolated(),
            pcg_smoke=_pcg_smoke(),
            pcg_isolated=_isolated(),
        )
        self.assertEqual(decision["nas201"]["decision"], "GO")
        self.assertEqual(decision["pcg_sokoban"]["decision"], "REVISE")
        self.assertEqual(decision["pcg_zelda"]["decision"], "REVISE")
        self.assertEqual(decision["pcg_family"]["decision"], "REVISE")
        self.assertFalse(decision["shortlist"]["confirmatory_tasks_frozen"])
        self.assertFalse(decision["shortlist"]["needs_replacement_family"])
        self.assertEqual(decision["shortlist"]["eligible_families"], ["nas201"])
        self.assertIn(
            "selector_jaccard_at_least_0.80",
            decision["pcg_sokoban"]["revise_reasons"],
        )
        self.assertIn(
            "identity_repair_playable_zero",
            decision["pcg_sokoban"]["revise_reasons"],
        )
        self.assertTrue(decision["pcg_family"]["zelda_is_not_a_second_family"])
        self.assertGreaterEqual(
            decision["pcg_sokoban"]["selector_jaccard"], MAX_SELECTOR_JACCARD
        )
        self.assertLess(decision["pcg_zelda"]["selector_jaccard"], MAX_SELECTOR_JACCARD)
        self.assertIsNone(decision["pcg_zelda"]["gates"]["parse"])
        self.assertTrue(decision["repair_not_silently_enabled"])
        self.assertEqual(decision["thresholds"]["max_selector_jaccard"], 0.80)
        self.assertTrue(decision["pcg_family"]["replacement_family_not_needed"])
        self.assertEqual(decision["shortlist"]["next_stage"], NEXT_STAGE_PCG_REVISE)


@unittest.skipUnless(
    all(path.is_file() for path in (NAS_SMOKE, NAS_ISOLATED, PCG_SMOKE, PCG_ISOLATED)),
    "NAS/PCG smoke and isolated reports are not on disk",
)
class TestFeasibilityDecisionOnDisk(unittest.TestCase):
    def test_on_disk_reports_use_current_gate_keys(self) -> None:
        nas_smoke = json.loads(NAS_SMOKE.read_text(encoding="utf-8"))
        nas_isolated = json.loads(NAS_ISOLATED.read_text(encoding="utf-8"))
        pcg_smoke = json.loads(PCG_SMOKE.read_text(encoding="utf-8"))
        pcg_isolated = json.loads(PCG_ISOLATED.read_text(encoding="utf-8"))
        self.assertIn("full_lookup", nas_smoke["gates"])
        self.assertIn("parse", nas_isolated["gates"])
        self.assertIn("cost_order", nas_isolated)
        self.assertIn("selector_jaccard", pcg_smoke["sokoban"]["gates"])
        self.assertIn("parse", pcg_isolated["gates"])
        decision = build_decision(
            nas_smoke=nas_smoke,
            nas_isolated=nas_isolated,
            pcg_smoke=pcg_smoke,
            pcg_isolated=pcg_isolated,
        )
        self.assertIn(decision["nas201"]["decision"], {"GO", "REVISE", "DROP"})
        self.assertIn(decision["pcg_family"]["decision"], {"GO", "REVISE", "DROP"})


class TestDropTriggersReplacementFamily(unittest.TestCase):
    def test_pcg_drop_sets_replacement_family_on_shortlist(self) -> None:
        family = decide_pcg_family(
            {"decision": "DROP"},
            {"decision": "REVISE"},
        )
        self.assertEqual(family["decision"], "DROP")
        self.assertFalse(family["replacement_family_not_needed"])
        shortlist = decide_shortlist({"decision": "GO"}, family)
        self.assertTrue(shortlist["needs_replacement_family"])
        self.assertEqual(shortlist["dropped_families"], ["pcg_benchmark"])
        self.assertEqual(shortlist["next_stage"], NEXT_STAGE_REPLACEMENT_FAMILY)

    def test_pcg_revise_does_not_set_replacement_family(self) -> None:
        family = decide_pcg_family(
            {"decision": "REVISE"},
            {"decision": "REVISE"},
        )
        self.assertTrue(family["replacement_family_not_needed"])
        shortlist = decide_shortlist({"decision": "GO"}, family)
        self.assertFalse(shortlist["needs_replacement_family"])
        self.assertEqual(shortlist["next_stage"], NEXT_STAGE_PCG_REVISE)

    def test_nas_drop_sets_replacement_family(self) -> None:
        family = decide_pcg_family(
            {"decision": "GO"},
            {"decision": "REVISE"},
        )
        self.assertTrue(family["replacement_family_not_needed"])
        shortlist = decide_shortlist({"decision": "DROP"}, family)
        self.assertTrue(shortlist["needs_replacement_family"])
        self.assertEqual(shortlist["dropped_families"], ["nas201"])
        self.assertEqual(shortlist["next_stage"], NEXT_STAGE_REPLACEMENT_FAMILY)


if __name__ == "__main__":
    unittest.main()

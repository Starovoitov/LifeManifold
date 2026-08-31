from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_claim_inventory.py"
SPEC = importlib.util.spec_from_file_location("claim_inventory", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
claim_inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(claim_inventory)


def _component(status: str = "matched") -> dict[str, object]:
    return {
        "status": status,
        "focal": "focal",
        "baseline": "baseline",
        "evidence": "fixture",
    }


def _minimal_registries() -> dict[str, list[dict[str, object]]]:
    treatment = {name: _component("unclear") for name in claim_inventory.COMPONENTS}
    treatment["generator"] = _component("changed")
    treatment["evaluator"] = _component("matched")
    papers = [
        {
            "record_type": "paper",
            "paper_id": f"seed-{seed}",
            "title": aliases[0],
            "authors": ["Fixture"],
            "year": 2024,
            "version_status": "peer_reviewed",
            "stratum": "core",
            "identifiers": {},
            "urls": [f"https://example.test/{seed}"],
            "domain": "fixture",
            "task": "fixture",
            "code_available": None,
            "screening_id": "ft-fixture",
        }
        for seed, aliases in claim_inventory.SEED_TITLES.items()
    ]
    return {
        "search_runs": [
            {
                "record_type": "search_run",
                "search_id": "fixture-q1",
                "executed_at": "2026-08-31T00:00:00+00:00",
                "source": "openalex",
                "query_family": "Q1",
                "exact_query": "fixture",
                "filters": {},
                "result_count": 1,
                "formal": True,
                "export_path": "artifacts/controlled_attribution/raw/fixture.json",
            }
        ],
        "screening": [
            {
                "record_type": "screening",
                "screening_id": "ft-fixture",
                "title": papers[0]["title"],
                "discovery_source": "fixture",
                "stage": "full_text",
                "decision": "include",
                "reason_code": None,
                "audited": True,
            }
        ],
        "papers": papers,
        "comparisons": [
            {
                "record_type": "comparison",
                "comparison_id": "seed-openelm-c1",
                "paper_id": "seed-openelm",
                "claim": "Fixture contrast.",
                "focal_arm": "focal",
                "baseline_arm": "baseline",
                "endpoint": "coverage",
                "budget_axes": {
                    "reported": ["proposal"],
                    "omitted": sorted(claim_inventory.BUDGET_AXES - {"proposal"}),
                },
                "sample": {"unit": "run", "n": 1, "seeds": [0]},
                "reported_result": {
                    "direction": "unclear",
                    "effect": None,
                    "focal_value": None,
                    "baseline_value": None,
                    "units": "coverage",
                },
                "treatment_vector": treatment,
                "identified_effects": ["generator"],
                "unidentified_effects": ["initialization"],
                "source": {
                    "url": "https://example.test/openelm",
                    "section": "Results",
                    "quote": "",
                },
            }
        ],
        "internal_claims": [],
        "adjudication": [],
    }


class TestClaimInventory(unittest.TestCase):
    def test_seed_recall_matches_title_aliases(self) -> None:
        papers = [
            {"title": aliases[0]} for aliases in claim_inventory.SEED_TITLES.values()
        ]

        recall = claim_inventory.seed_recall(papers)

        self.assertTrue(all(recall.values()))

    def test_missing_treatment_component_is_rejected(self) -> None:
        registries = _minimal_registries()
        broken = copy.deepcopy(registries)
        broken["comparisons"][0]["treatment_vector"].pop("budget")

        errors = claim_inventory.validate_registries(broken)

        self.assertTrue(
            any("missing components ['budget']" in error for error in errors)
        )

    def test_complete_fixture_validates(self) -> None:
        errors = claim_inventory.validate_registries(_minimal_registries())

        self.assertEqual(errors, [])

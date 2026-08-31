from __future__ import annotations

import hashlib
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "finalize_claim_inventory_screening.py"
SPEC = importlib.util.spec_from_file_location(
    "finalize_claim_inventory_screening", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
finalize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalize)


def _exclusion_audit_id() -> str:
    for index in range(10_000):
        screening_id = f"ta-{index:04d}-fixture"
        digest = int(hashlib.sha256(screening_id.encode()).hexdigest()[:8], 16)
        if digest % 10 == 0:
            return screening_id
    raise AssertionError("no screening_id hashed into the 10% exclusion audit")


class TestAuditTitleAbstractRows(unittest.TestCase):
    def test_include_without_notes_does_not_raise(self) -> None:
        row = {
            "stage": "title_abstract",
            "decision": "include",
            "reason_code": None,
            "audited": False,
            "screening_id": "ta-0001-fixture",
        }

        finalize.audit_title_abstract_rows([row])

        self.assertTrue(row["audited"])
        self.assertEqual(row["notes"], "Full-text eligibility pass completed.")

    def test_existing_notes_are_appended(self) -> None:
        row = {
            "stage": "title_abstract",
            "decision": "unclear",
            "reason_code": None,
            "audited": False,
            "screening_id": "ta-0002-fixture",
            "notes": "Needs full text.",
        }

        finalize.audit_title_abstract_rows([row])

        self.assertEqual(
            row["notes"],
            "Needs full text. Full-text eligibility pass completed.",
        )

    def test_exclusion_audit_without_notes_does_not_raise(self) -> None:
        row = {
            "stage": "title_abstract",
            "decision": "exclude",
            "reason_code": "out_of_scope",
            "audited": False,
            "screening_id": _exclusion_audit_id(),
        }

        finalize.audit_title_abstract_rows([row])

        self.assertTrue(row["audited"])
        self.assertEqual(
            row["notes"],
            "Included in deterministic 10% exclusion consistency audit.",
        )


class TestSupplementaryInclusionEntry(unittest.TestCase):
    def test_position_zero_is_title_not_paper_id(self) -> None:
        supplementary = finalize.SUPPLEMENTARY_INCLUSIONS[0]
        paper_id, title = supplementary[0], supplementary[1]

        entry = finalize.inclusion_entry_from_supplementary(supplementary)

        self.assertEqual(entry[0], title)
        self.assertEqual(entry[1], paper_id)
        self.assertNotEqual(entry[0], entry[1])

    def test_layout_matches_primary_inclusions(self) -> None:
        primary = finalize.INCLUSIONS[0]
        for supplementary in finalize.SUPPLEMENTARY_INCLUSIONS:
            entry = finalize.inclusion_entry_from_supplementary(supplementary)
            self.assertEqual(len(entry), len(primary))
            fragment, paper_id, *_rest = entry
            self.assertEqual(fragment, supplementary[1])
            self.assertEqual(paper_id, supplementary[0])

    def test_make_paper_uses_paper_id_from_position_one(self) -> None:
        supplementary = finalize.SUPPLEMENTARY_INCLUSIONS[0]
        entry = finalize.inclusion_entry_from_supplementary(supplementary)
        candidate = {
            "title": supplementary[1],
            "authors": supplementary[2],
            "year": supplementary[3],
            "venue": supplementary[4],
        }

        paper = finalize.make_paper(candidate, entry, "ft-snowball-fixture")
        comparison = finalize.make_comparison(candidate, entry)

        self.assertEqual(paper["paper_id"], supplementary[0])
        self.assertEqual(paper["title"], supplementary[1])
        self.assertEqual(paper["urls"], [supplementary[11]])
        self.assertEqual(comparison["paper_id"], supplementary[0])
        self.assertEqual(comparison["source"]["url"], supplementary[11])

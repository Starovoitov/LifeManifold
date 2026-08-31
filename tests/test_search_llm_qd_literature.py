from __future__ import annotations

import importlib.util
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "search_llm_qd_literature.py"
SPEC = importlib.util.spec_from_file_location("search_llm_qd_literature", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
search_llm_qd_literature = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(search_llm_qd_literature)


class TestAllocateRawExportDir(unittest.TestCase):
    def test_preferred_stamp_uses_second_precision(self) -> None:
        timestamp = datetime(2026, 8, 31, 19, 30, 12, 123456, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executed_at, stamp, raw_dir = (
                search_llm_qd_literature.allocate_raw_export_dir(
                    root, timestamp=timestamp
                )
            )

            self.assertEqual(executed_at, timestamp.replace(microsecond=0))
            self.assertEqual(stamp, "20260831T193012Z")
            self.assertEqual(raw_dir, root / "raw" / stamp)
            self.assertTrue(raw_dir.is_dir())

    def test_same_second_collision_gets_unique_directory(self) -> None:
        timestamp = datetime(2026, 8, 31, 19, 30, 12, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_at, first_stamp, first_dir = (
                search_llm_qd_literature.allocate_raw_export_dir(
                    root, timestamp=timestamp
                )
            )
            second_at, second_stamp, second_dir = (
                search_llm_qd_literature.allocate_raw_export_dir(
                    root, timestamp=timestamp
                )
            )

            self.assertEqual(first_at, second_at)
            self.assertEqual(first_stamp, "20260831T193012Z")
            self.assertTrue(second_stamp.startswith("20260831T193012Z-"))
            self.assertNotEqual(first_dir, second_dir)
            self.assertTrue(first_dir.is_dir())
            self.assertTrue(second_dir.is_dir())


class TestSearchRunNotes(unittest.TestCase):
    def test_complete_standard_query_does_not_claim_amendment(self) -> None:
        notes = search_llm_qd_literature.search_run_notes(
            returned=4,
            reported=4,
            status="ok",
            error=None,
            amendment_1=False,
        )

        self.assertEqual(notes, "returned=4")
        self.assertNotIn("amendment=1", notes)

    def test_complete_amendment_query_records_the_variant(self) -> None:
        notes = search_llm_qd_literature.search_run_notes(
            returned=4,
            reported=4,
            status="ok",
            error=None,
            amendment_1=True,
        )

        self.assertEqual(notes, "returned=4; amendment=1")

    def test_incomplete_and_failed_notes_are_unchanged(self) -> None:
        self.assertEqual(
            search_llm_qd_literature.search_run_notes(
                returned=2,
                reported=5,
                status="ok",
                error=None,
                amendment_1=False,
            ),
            "incomplete: returned=2 of reported=5",
        )
        self.assertEqual(
            search_llm_qd_literature.search_run_notes(
                returned=0,
                reported=0,
                status="error",
                error="TimeoutError: timed out",
                amendment_1=True,
            ),
            "search failed and is not formal: TimeoutError: timed out",
        )

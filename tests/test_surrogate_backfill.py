"""Tests for surrogate buffer backfill from MAP-Elites archive JSONL."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldspace.surrogate.backfill import backfill_buffer_from_archive
from worldspace.surrogate.training import load_buffer


class TestSurrogateBackfill(unittest.TestCase):
    def test_backfill_smoke_archive(self) -> None:
        archive = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "map_elites_smoke"
            / "map_elites_archive.jsonl"
        )
        if not archive.is_file():
            self.skipTest("smoke archive missing")
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            stats = backfill_buffer_from_archive(archive, buffer_path)
            self.assertGreater(stats["buffer_rows_written"], 0)
            features, targets = load_buffer(buffer_path)
            self.assertEqual(features.shape[0], stats["buffer_rows_written"])
            self.assertEqual(features.shape[0], len(targets["stability"]))
            first_line = json.loads(
                buffer_path.read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(first_line["metadata"]["source"], "archive_backfill")


if __name__ == "__main__":
    unittest.main()

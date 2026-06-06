"""Tests for surrogate buffer backfill from MAP-Elites archive JSONL."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from worldspace.illuminators.evaluation import apply_canonical_seed
from worldspace.specs.spec import WorldSpec
from worldspace.surrogate.backfill import backfill_buffer_from_archive
from worldspace.surrogate.feature_extractor import extract
from worldspace.surrogate.genome_features import FEATURE_DIM
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
            self.assertEqual(features.shape[1], FEATURE_DIM)
            self.assertEqual(features.shape[0], len(targets["stability"]))
            first_line = json.loads(
                buffer_path.read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(first_line["metadata"]["source"], "archive_backfill")
            self.assertEqual(first_line["feature_schema_version"], "2.0")
            self.assertIn("world_spec", first_line)
            restored = WorldSpec.from_json_dict(first_line["world_spec"])
            apply_canonical_seed(restored)
            np.testing.assert_allclose(
                extract(restored),
                np.asarray(first_line["features"], dtype=float),
            )


if __name__ == "__main__":
    unittest.main()

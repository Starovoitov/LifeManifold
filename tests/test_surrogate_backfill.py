"""Tests for surrogate buffer backfill from MAP-Elites archive JSONL."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from worldspace.illuminators.archive import elite_to_archive_record
from worldspace.illuminators.evaluation import apply_canonical_seed
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec
from tests.test_map_elites_archive import (
    _example_metrics,
    _minimal_elite,
    _write_jsonl,
)
from worldspace.surrogate.backfill import (
    backfill_buffer_from_archive,
    buffer_has_archive_backfill_rows,
    buffer_has_live_eval_rows,
)
from worldspace.surrogate.buffer import buffer_record, world_spec_dict_for_buffer
from worldspace.surrogate.model import TARGET_KEYS
from worldspace.surrogate.feature_extractor import extract
from worldspace.surrogate.genome_features import FEATURE_DIM_V21
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
            self.assertEqual(features.shape[1], FEATURE_DIM_V21)
            self.assertEqual(features.shape[0], len(targets["stability"]))
            first_line = json.loads(
                buffer_path.read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(first_line["metadata"]["source"], "archive_backfill")
            self.assertEqual(first_line["feature_schema_version"], "2.1")
            self.assertIn("world_spec", first_line)
            restored = WorldSpec.from_json_dict(first_line["world_spec"])
            apply_canonical_seed(restored)
            np.testing.assert_allclose(
                extract(restored),
                np.asarray(first_line["features"], dtype=float),
            )

    def test_backfill_append_preserves_existing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "archive.jsonl"
            _write_jsonl(
                archive,
                [
                    elite_to_archive_record(
                        replace(
                            _minimal_elite((0, 0), 0.7, elite_id="elite-a"),
                            metrics=_example_metrics(),
                        )
                    ),
                    elite_to_archive_record(
                        replace(
                            _minimal_elite((1, 1), 0.8, elite_id="elite-b"),
                            metrics=_example_metrics(density_mean=0.4),
                        )
                    ),
                ],
            )
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            spec = WorldSpec(
                birth=[3],
                survival=[2, 3],
                noise=0.1,
                resource_regen=0.2,
                predation=0.05,
                cell_types=list(CANONICAL_CELL_TYPES),
                grid_size=30,
                steps=220,
                seed=0,
            )
            apply_canonical_seed(spec)
            live_row = buffer_record(
                features=extract(spec),
                targets={key: 0.5 for key in TARGET_KEYS},
                emitter_type="genetic",
                world_spec=world_spec_dict_for_buffer(spec),
                metadata={"source": "live_eval"},
            )
            buffer_path.write_text(
                json.dumps(live_row, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            stats = backfill_buffer_from_archive(archive, buffer_path, overwrite=False)
            lines = buffer_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1 + stats["buffer_rows_written"])
            self.assertTrue(buffer_has_live_eval_rows(buffer_path))
            self.assertTrue(buffer_has_archive_backfill_rows(buffer_path))
            self.assertEqual(json.loads(lines[0])["metadata"]["source"], "live_eval")


if __name__ == "__main__":
    unittest.main()

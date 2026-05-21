"""Unit tests for dashboard archive flattening and canonical hash."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_ARCHIVE = (
    _REPO_ROOT / "artifacts" / "map_elites_smoke" / "map_elites_archive.jsonl"
)


class TestDashboardDataProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        line = _SMOKE_ARCHIVE.read_text(encoding="utf-8").splitlines()[0]
        cls.record = json.loads(line)

    def test_flatten_archive_record_bins(self) -> None:
        from dashboard.utils.data_processing import flatten_archive_record

        row = flatten_archive_record(self.record)
        self.assertEqual(row["bin_x"], int(self.record["bin"][0]))
        self.assertEqual(row["bin_y"], int(self.record["bin"][1]))
        self.assertEqual(row["fitness"], float(self.record["fitness"]))

    def test_canonical_hash_stable_under_key_permutation(self) -> None:
        from dashboard.utils.data_processing import canonical_world_spec_hash

        spec = dict(self.record["world_spec"])
        shuffled = {key: spec[key] for key in reversed(list(spec.keys()))}
        self.assertEqual(
            canonical_world_spec_hash(spec),
            canonical_world_spec_hash(shuffled),
        )

    def test_try_flatten_skips_missing_required_fields(self) -> None:
        from dashboard.utils.data_processing import try_flatten_archive_record

        valid = try_flatten_archive_record(self.record)
        self.assertIsNotNone(valid)

        missing_fitness = dict(self.record)
        del missing_fitness["fitness"]
        self.assertIsNone(try_flatten_archive_record(missing_fitness))

        missing_bin = dict(self.record)
        del missing_bin["bin"]
        self.assertIsNone(try_flatten_archive_record(missing_bin))

    def test_canonical_hash_ignores_runtime_seed(self) -> None:
        from dashboard.utils.data_processing import canonical_world_spec_hash

        base = dict(self.record["world_spec"])
        mutated = dict(base)
        mutated["seed"] = int(base.get("seed", 0)) + 999
        self.assertEqual(
            canonical_world_spec_hash(base),
            canonical_world_spec_hash(mutated),
        )


if __name__ == "__main__":
    unittest.main()

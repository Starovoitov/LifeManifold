"""Unit tests for dashboard world simulation cache helpers."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_ARCHIVE = (
    _REPO_ROOT / "artifacts" / "map_elites_smoke" / "map_elites_archive.jsonl"
)


class TestDashboardWorldRenderer(unittest.TestCase):
    def test_run_world_for_spec_dict_returns_grids(self) -> None:
        from dashboard.components.world_renderer import run_world_for_spec_dict
        from dashboard.utils.data_processing import flatten_archive_record

        record = json.loads(_SMOKE_ARCHIVE.read_text(encoding="utf-8").splitlines()[0])
        row = flatten_archive_record(record)
        result = run_world_for_spec_dict(row["world_spec"])
        life = result.final_life
        food = result.final_food
        assert life is not None
        assert food is not None
        self.assertEqual(life.shape, food.shape)
        self.assertIsNotNone(result.metrics)

    def test_maps_from_result_shapes_match_life(self) -> None:
        from dashboard.components.world_renderer import (
            maps_from_result,
            run_world_for_spec_dict,
        )
        from dashboard.utils.data_processing import flatten_archive_record

        record = json.loads(_SMOKE_ARCHIVE.read_text(encoding="utf-8").splitlines()[0])
        row = flatten_archive_record(record)
        result = run_world_for_spec_dict(row["world_spec"])
        maps = maps_from_result(result)
        life = result.final_life
        food = result.final_food
        assert life is not None
        assert food is not None
        self.assertEqual(maps.boundary.shape, life.shape)
        self.assertEqual(maps.heterogeneity.shape, life.shape)
        self.assertEqual(maps.food_neighbor.shape, food.shape)

    def test_prepare_world_spec_for_run_raises_steps_floor(self) -> None:
        from dataclasses import replace

        from dashboard.components.world_renderer import prepare_world_spec_for_run
        from dashboard.utils.data_processing import (
            flatten_archive_record,
            world_spec_from_dict,
        )
        from worldspace.illuminators.evaluation import ILLUMINATOR_MIN_STEPS

        record = json.loads(_SMOKE_ARCHIVE.read_text(encoding="utf-8").splitlines()[0])
        row = flatten_archive_record(record)
        spec = replace(world_spec_from_dict(row["world_spec"]), steps=1)
        prepared = prepare_world_spec_for_run(spec)
        self.assertGreaterEqual(prepared.steps, ILLUMINATOR_MIN_STEPS)

    def test_canonical_hash_stable_for_same_spec(self) -> None:
        from dashboard.utils.data_processing import (
            canonical_world_spec_hash,
            flatten_archive_record,
        )

        record = json.loads(_SMOKE_ARCHIVE.read_text(encoding="utf-8").splitlines()[0])
        row = flatten_archive_record(record)
        spec = row["world_spec"]
        self.assertEqual(
            canonical_world_spec_hash(spec),
            canonical_world_spec_hash(dict(spec)),
        )


if __name__ == "__main__":
    unittest.main()

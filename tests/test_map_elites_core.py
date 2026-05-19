"""Unit tests for MAP-Elites core (TZ v1.2 §11.1 — E1.x)."""

from __future__ import annotations

import json
import unittest

from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec

_CANONICAL_JSON_KWARGS = {"sort_keys": True, "separators": (",", ":")}


def _canonical_json(spec: WorldSpec) -> str:
    return json.dumps(spec.to_canonical_dict(), **_CANONICAL_JSON_KWARGS)


class TestWorldSpecCanonicalDict(unittest.TestCase):
    def test_canonical_dict_excludes_seed(self) -> None:
        spec = WorldSpec(
            birth=[1],
            survival=[2, 3],
            noise=0.01,
            resource_regen=0.1,
            predation=0.2,
            cell_types=["life", "food"],
            seed=999,
        )
        canonical = spec.to_canonical_dict()
        self.assertNotIn("seed", canonical)
        self.assertEqual(spec.seed, 999)

    def test_canonical_dict_stable_json(self) -> None:
        a = WorldSpec(
            birth=[3, 1, 2],
            survival=[5, 4],
            noise=0.10000004,
            resource_regen=0.05,
            predation=0.25,
            cell_types=["empty", "life", "food"],
            grid_size=40,
            steps=250,
            seed=1,
        )
        b = WorldSpec(
            birth=[2, 3, 1],
            survival=[4, 5],
            noise=0.10000005,
            resource_regen=0.05,
            predation=0.25,
            cell_types=["food", "life", "empty"],
            grid_size=40,
            steps=250,
            seed=99,
        )
        self.assertEqual(_canonical_json(a), _canonical_json(b))

    def test_canonical_dict_sorts_rule_lists(self) -> None:
        spec = WorldSpec(
            birth=[3, 1, 2],
            survival=[8, 2, 5],
            noise=0.0,
            resource_regen=0.0,
            predation=0.0,
            cell_types=["life", "food"],
        )
        canonical = spec.to_canonical_dict()
        self.assertEqual(canonical["birth"], [1, 2, 3])
        self.assertEqual(canonical["survival"], [2, 5, 8])

    def test_canonical_dict_rounds_floats(self) -> None:
        a = WorldSpec(
            birth=[0],
            survival=[1],
            noise=0.10000004,
            resource_regen=0.20000004,
            predation=0.30000004,
            cell_types=["life", "food"],
        )
        b = WorldSpec(
            birth=[0],
            survival=[1],
            noise=0.10000005,
            resource_regen=0.20000005,
            predation=0.30000005,
            cell_types=["life", "food"],
        )
        self.assertEqual(_canonical_json(a), _canonical_json(b))
        canonical = a.to_canonical_dict()
        self.assertEqual(canonical["noise"], 0.1)
        self.assertEqual(canonical["resource_regen"], 0.2)
        self.assertEqual(canonical["predation"], 0.3)

    def test_canonical_dict_normalizes_cell_types(self) -> None:
        spec = WorldSpec(
            birth=[0],
            survival=[1],
            noise=0.0,
            resource_regen=0.0,
            predation=0.0,
            cell_types=["empty", "life", "food"],
        )
        self.assertEqual(
            spec.to_canonical_dict()["cell_types"], list(CANONICAL_CELL_TYPES)
        )

    def test_canonical_dict_differs_from_to_json_dict(self) -> None:
        spec = WorldSpec(
            birth=[2, 1],
            survival=[3],
            noise=0.05,
            resource_regen=0.1,
            predation=0.15,
            cell_types=["empty", "life", "food"],
            seed=42,
        )
        canonical = spec.to_canonical_dict()
        raw = spec.to_json_dict()
        self.assertNotIn("seed", canonical)
        self.assertIn("seed", raw)
        self.assertEqual(canonical["cell_types"], ["life", "food"])
        self.assertEqual(raw["cell_types"], ["empty", "life", "food"])
        self.assertEqual(canonical["birth"], [1, 2])
        self.assertEqual(raw["birth"], [2, 1])


if __name__ == "__main__":
    unittest.main()

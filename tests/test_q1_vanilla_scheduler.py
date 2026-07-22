"""Load tests for Q1 v3 B1 / RQ0 vanilla MAP-Elites scheduler."""

from __future__ import annotations

import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SPECS = _REPO_ROOT / "worldspace" / "specs"
_VANILLA = _SPECS / "map_elites_scheduler_nightly_vanilla.yaml"


class TestQ1VanillaScheduler(unittest.TestCase):
    def test_nightly_vanilla_scheduler_load(self) -> None:
        from worldspace.illuminators.scheduler import (
            load_scheduler,
            resolve_emitter_kind,
        )

        config = load_scheduler(_VANILLA)
        self.assertEqual(config.schema_version, "1.2")
        self.assertEqual(config.archive_type, "grid")
        self.assertEqual(config.iterations, 650)
        self.assertEqual(config.batch_size, 50)
        self.assertEqual(len(config.batch_emitters), 50)
        self.assertEqual(config.batch_emitters.count("random"), 50)
        self.assertEqual(config.batch_emitters.count("genetic"), 0)
        self.assertEqual(config.batch_emitters.count("llm"), 0)
        self.assertFalse(config.llm_enabled)
        self.assertFalse(config.surrogate_enabled)
        self.assertTrue(config.performance.parallel_eval)
        self.assertFalse(config.performance.llm_parallel_emit)

        # After initial fill, every YAML slot stays random (no genetic/LLM).
        for slot in config.batch_emitters:
            kind = resolve_emitter_kind(
                config,
                slot_emitter=slot,
                candidates_evaluated=config.initial_random_candidates,
            )
            self.assertEqual(kind, "random")

    def test_aggregate_infers_vanilla_condition(self) -> None:
        from scripts.aggregate_experiment_runs import _infer_condition

        path = (
            _REPO_ROOT
            / "artifacts"
            / "experiments"
            / "q1-v3-vanilla"
            / "vanilla"
            / "seed_0"
        )
        self.assertEqual(_infer_condition(path), "vanilla")


if __name__ == "__main__":
    unittest.main()

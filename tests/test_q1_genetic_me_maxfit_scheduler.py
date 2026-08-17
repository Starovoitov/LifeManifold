"""Load tests for the genetic ME max-fitness-frontier policy arm."""

from __future__ import annotations

import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCHEDULER = (
    _REPO_ROOT
    / "worldspace"
    / "specs"
    / "map_elites_scheduler_nightly_genetic_me_maxfit.yaml"
)


class TestQ1GeneticMEMaxfitScheduler(unittest.TestCase):
    def test_scheduler_is_matched_no_surrogate_maxfit_arm(self) -> None:
        from worldspace.illuminators.scheduler import load_scheduler

        config = load_scheduler(_SCHEDULER)

        self.assertEqual(config.schema_version, "1.2")
        self.assertEqual(config.iterations, 650)
        self.assertEqual(config.batch_size, 50)
        self.assertEqual(config.target_selection, "max_fitness_frontier")
        self.assertEqual(config.batch_emitters.count("random"), 20)
        self.assertEqual(config.batch_emitters.count("genetic"), 30)
        self.assertEqual(config.batch_emitters.count("llm"), 0)
        self.assertFalse(config.llm_enabled)
        self.assertFalse(config.surrogate_enabled)
        self.assertTrue(config.performance.log_iteration_timing)

    def test_aggregate_infers_condition(self) -> None:
        from scripts.aggregate_experiment_runs import _infer_condition

        run_dir = (
            _REPO_ROOT
            / "artifacts"
            / "experiments"
            / "q1-v3-genetic-me-maxfit"
            / "genetic_me_maxfit"
            / "seed_0"
        )
        self.assertEqual(_infer_condition(run_dir), "genetic_me_maxfit")


if __name__ == "__main__":
    unittest.main()

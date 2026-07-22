"""Tests for dungeon archive, schedulers, runner, and artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldspace.dungeons.archive import DungeonArchive, DungeonElite
from worldspace.dungeons.genetics import random_dungeon
from worldspace.dungeons.runner import (
    DungeonSchedulerConfig,
    load_dungeon_scheduler,
    run_dungeon_qd,
)
from worldspace.dungeons.surrogate import DungeonPrediction
from worldspace.surrogate.acquisition_config import AcquisitionConfig


class _LowPredictor:
    def predict(self, spec: object) -> DungeonPrediction:
        return DungeonPrediction({}, {"path_length": 0.0, "branching": 0.0}, 0.0, 0.0)


class TestDungeonRunner(unittest.TestCase):
    def test_locked_baseline_schedulers_load(self) -> None:
        root = Path(__file__).resolve().parents[1] / "worldspace/specs"
        random = load_dungeon_scheduler(root / "dungeon_scheduler_random.yaml")
        genetic = load_dungeon_scheduler(root / "dungeon_scheduler_genetic.yaml")
        self.assertEqual(random.emitters.count("random"), 50)
        self.assertEqual(genetic.emitters.count("random"), 20)
        self.assertEqual(genetic.emitters.count("genetic"), 30)
        self.assertEqual(genetic.archive_resolution, 30)
        expected = {
            "genetic_filter": ("genetic", "filter"),
            "llm_stub": ("llm", "off"),
            "llm_hints": ("llm", "off"),
            "llm_hints_filter": ("llm", "filter"),
        }
        for condition, (emitter, acquisition) in expected.items():
            with self.subTest(condition=condition):
                config = load_dungeon_scheduler(
                    root / f"dungeon_scheduler_{condition}.yaml"
                )
                self.assertEqual(config.emitters.count(emitter), 30)
                self.assertEqual(config.acquisition.mode, acquisition)

    def test_archive_replaces_only_with_better_fitness(self) -> None:
        import numpy as np

        spec = random_dungeon(np.random.default_rng(1))
        archive = DungeonArchive(4)
        low = DungeonElite((0, 0), 0.2, (0.1, 0.1), spec, "a", None, "random")
        high = DungeonElite((0, 0), 0.3, (0.1, 0.1), spec, "b", None, "random")
        self.assertTrue(archive.try_insert(low).accepted)
        self.assertFalse(archive.try_insert(low).accepted)
        self.assertTrue(archive.try_insert(high).improved)
        self.assertEqual(archive.get_cell(0), high)

    @staticmethod
    def _config() -> DungeonSchedulerConfig:
        return DungeonSchedulerConfig(
            condition="genetic",
            iterations=2,
            batch_size=5,
            archive_resolution=8,
            initial_random_candidates=5,
            emitters=("random", "random", "genetic", "genetic", "genetic"),
        )

    def test_run_writes_complete_artifact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = run_dungeon_qd(self._config(), seed=3, output_dir=out)
            self.assertEqual(result.proposals, 10)
            self.assertEqual(result.evaluations, 10)
            self.assertGreater(result.filled_cells, 0)
            summary = json.loads(
                (out / "nightly_run_summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue(summary["dungeon_benchmark"])
            self.assertEqual(summary["proposals"], 10)
            self.assertTrue((out / "dungeon_archive.jsonl").is_file())
            self.assertTrue((out / "surrogate_archive.jsonl").is_file())
            traces = (out / "archive_trace.jsonl").read_text().splitlines()
            self.assertEqual(len(traces), 3)
            self.assertEqual(json.loads(traces[-1])["proposals"], 10)
            with self.assertRaises(FileExistsError):
                run_dungeon_qd(self._config(), seed=3, output_dir=out)

    def test_same_seed_reproduces_metrics(self) -> None:
        with (
            tempfile.TemporaryDirectory() as first_dir,
            tempfile.TemporaryDirectory() as second_dir,
        ):
            first = run_dungeon_qd(self._config(), seed=9, output_dir=Path(first_dir))
            second = run_dungeon_qd(self._config(), seed=9, output_dir=Path(second_dir))
            self.assertEqual(first.filled_cells, second.filled_cells)
            self.assertEqual(first.coverage, second.coverage)
            self.assertEqual(first.mean_best_fitness, second.mean_best_fitness)
            self.assertEqual(first.qd_score, second.qd_score)

    def test_filter_skips_low_predictions_but_explores_empty_targets(self) -> None:
        config = DungeonSchedulerConfig(
            condition="genetic_filter",
            iterations=3,
            batch_size=5,
            archive_resolution=8,
            initial_random_candidates=0,
            emitters=("genetic",) * 5,
            surrogate_checkpoint="dummy.pkl",
            acquisition=AcquisitionConfig(
                mode="filter",
                min_predicted_fitness=0.5,
                max_uncertainty_to_skip=1.0,
                never_skip_empty_bin=True,
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = run_dungeon_qd(
                config,
                seed=2,
                output_dir=Path(tmp),
                predictor=_LowPredictor(),  # type: ignore[arg-type]
            )
            self.assertGreater(result.evaluations, 0)
            self.assertGreater(result.skipped, 0)
            self.assertEqual(result.proposals, 15)


if __name__ == "__main__":
    unittest.main()

"""Tests for maze archive, schedulers, runner, and artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from worldspace.mazes.archive import MazeArchive, MazeElite
from worldspace.mazes.genetics import random_maze
from worldspace.mazes.runner import (
    MazeSchedulerConfig,
    load_maze_scheduler,
    run_maze_qd,
)


class TestMazeRunner(unittest.TestCase):
    def test_locked_baseline_schedulers_load(self) -> None:
        root = Path(__file__).resolve().parents[1] / "worldspace/specs"
        random = load_maze_scheduler(root / "maze_scheduler_random.yaml")
        genetic = load_maze_scheduler(root / "maze_scheduler_genetic.yaml")
        self.assertEqual(random.emitters.count("random"), 50)
        self.assertEqual(genetic.emitters.count("random"), 20)
        self.assertEqual(genetic.emitters.count("genetic"), 30)
        self.assertEqual(genetic.archive_resolution, 30)

    def test_archive_replaces_only_with_better_fitness(self) -> None:
        spec = random_maze(np.random.default_rng(1))
        archive = MazeArchive(4)
        low = MazeElite((0, 0), 0.2, (0.1, 0.1), spec, "a", None, "random")
        high = MazeElite((0, 0), 0.3, (0.1, 0.1), spec, "b", None, "random")
        self.assertTrue(archive.try_insert(low).accepted)
        self.assertFalse(archive.try_insert(low).accepted)
        self.assertTrue(archive.try_insert(high).improved)
        self.assertEqual(archive.get_cell(0), high)

    @staticmethod
    def _config() -> MazeSchedulerConfig:
        return MazeSchedulerConfig(
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
            result = run_maze_qd(self._config(), seed=3, output_dir=out)
            self.assertEqual(result.proposals, 10)
            self.assertEqual(result.evaluations, 10)
            self.assertGreater(result.filled_cells, 0)
            summary = json.loads(
                (out / "nightly_run_summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue(summary["maze_benchmark"])
            self.assertEqual(summary["proposals"], 10)
            self.assertTrue((out / "maze_archive.jsonl").is_file())
            self.assertTrue((out / "surrogate_archive.jsonl").is_file())
            traces = (out / "archive_trace.jsonl").read_text().splitlines()
            self.assertEqual(len(traces), 3)
            self.assertEqual(json.loads(traces[-1])["proposals"], 10)
            with self.assertRaises(FileExistsError):
                run_maze_qd(self._config(), seed=3, output_dir=out)

    def test_same_seed_reproduces_metrics(self) -> None:
        with (
            tempfile.TemporaryDirectory() as first_dir,
            tempfile.TemporaryDirectory() as second_dir,
        ):
            first = run_maze_qd(self._config(), seed=9, output_dir=Path(first_dir))
            second = run_maze_qd(self._config(), seed=9, output_dir=Path(second_dir))
            self.assertEqual(first.filled_cells, second.filled_cells)
            self.assertEqual(first.coverage, second.coverage)
            self.assertEqual(first.qd_score, second.qd_score)


if __name__ == "__main__":
    unittest.main()

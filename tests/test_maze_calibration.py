"""Tests for maze filter threshold calibration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from worldspace.mazes.calibration import (
    ReplayBatch,
    load_surrogate_archive,
    replay_skip_rate,
    search_fitness_threshold,
)


class TestMazeCalibration(unittest.TestCase):
    def test_replay_respects_empty_bin_force_eval(self) -> None:
        batch = ReplayBatch(
            name="synthetic",
            fitness=np.array([0.2, 0.2, 0.8]),
            uncertainty=np.array([0.01, 0.01, 0.01]),
            target_was_empty=np.array([True, False, False]),
        )
        rate = replay_skip_rate(
            batch,
            min_predicted_fitness=0.5,
            max_uncertainty_to_skip=0.02,
        )
        self.assertAlmostEqual(rate, 1.0 / 3.0)

    def test_search_finds_band_center_on_synthetic_live_batches(self) -> None:
        batch = ReplayBatch(
            name="live",
            fitness=np.linspace(0.2, 0.9, 500),
            uncertainty=np.full(500, 0.01),
            target_was_empty=np.zeros(500, dtype=bool),
        )
        chosen = search_fitness_threshold((batch,), target_skip=0.35)
        self.assertGreaterEqual(chosen.min_predicted_fitness, 0.45)
        self.assertAlmostEqual(chosen.per_source["live"], chosen.mean_skip_rate)
        self.assertGreaterEqual(chosen.mean_skip_rate, 0.25)
        self.assertLessEqual(chosen.mean_skip_rate, 0.45)

    def test_load_surrogate_archive_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "surrogate_archive.jsonl"
            row = {
                "target_was_empty": False,
                "prediction": {"fitness": 0.4, "uncertainty": 0.01},
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            batch = load_surrogate_archive(path, name="test")
            self.assertEqual(batch.n_rows, 1)
            self.assertAlmostEqual(batch.fitness[0], 0.4)

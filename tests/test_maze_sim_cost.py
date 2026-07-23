"""Tests for injected maze simulator wall-time cost (M6 / M3)."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from worldspace.mazes.evaluation import evaluate_maze
from worldspace.mazes.genetics import random_maze
from worldspace.mazes.runner import MazeSchedulerConfig, run_maze_qd


class TestMazeSimCost(unittest.TestCase):
    def test_sim_cost_does_not_change_fitness(self) -> None:
        maze = random_maze(np.random.default_rng(11))
        baseline = evaluate_maze(maze, sim_cost_ms=0.0)
        delayed = evaluate_maze(maze, sim_cost_ms=50.0)
        self.assertEqual(baseline, delayed)

    def test_sim_cost_adds_wall_time(self) -> None:
        maze = random_maze(np.random.default_rng(12))
        started = time.perf_counter()
        evaluate_maze(maze, sim_cost_ms=25.0)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.assertGreaterEqual(elapsed_ms, 20.0)

    def test_runner_records_sim_cost_in_summary(self) -> None:
        config = MazeSchedulerConfig(
            condition="genetic",
            iterations=1,
            batch_size=5,
            archive_resolution=8,
            initial_random_candidates=5,
            emitters=("random", "random", "genetic", "genetic", "genetic"),
            sim_cost_ms=3.5,
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run_maze_qd(config, seed=1, output_dir=out)
            payload = json.loads((out / "nightly_run_summary.json").read_text())
            self.assertEqual(payload["sim_cost_ms"], 3.5)

    def test_filter_skip_rate_unchanged_with_sim_cost(self) -> None:
        from worldspace.mazes.runner import load_maze_scheduler
        from worldspace.mazes.surrogate import MazeSurrogate

        root = Path(__file__).resolve().parents[1]
        base = load_maze_scheduler(
            root / "worldspace/specs/maze_scheduler_genetic_filter.yaml"
        )
        config = replace(base, iterations=2, batch_size=5, initial_random_candidates=5)
        genetic_emitters = ("random", "random", "genetic", "genetic", "genetic")
        config = replace(config, emitters=genetic_emitters, archive_resolution=8)
        predictor = MazeSurrogate.load(
            root / "artifacts/surrogate/checkpoints/maze_v1.pkl"
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_zero = Path(tmp) / "zero"
            out_cost = Path(tmp) / "cost"
            zero = run_maze_qd(
                replace(config, sim_cost_ms=0.0),
                seed=5,
                output_dir=out_zero,
                predictor=predictor,
            )
            cost = run_maze_qd(
                replace(config, sim_cost_ms=10.0),
                seed=5,
                output_dir=out_cost,
                predictor=predictor,
            )
            self.assertEqual(zero.skipped, cost.skipped)
            self.assertEqual(zero.evaluations, cost.evaluations)
            self.assertGreater(cost.elapsed_seconds, zero.elapsed_seconds)

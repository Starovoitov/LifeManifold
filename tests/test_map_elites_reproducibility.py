"""Integration tests for MAP-Elites cold-start reproducibility (mini scheduler)."""

from __future__ import annotations

import unittest

import numpy as np

from worldspace.illuminators.archive import GridArchive
from worldspace.illuminators.emitters.stub import StubCandidateEmitter
from worldspace.illuminators.loop import run_scheduler
from worldspace.illuminators.scheduler import (
    DEFAULT_MINI_SCHEDULER_PATH,
    SchedulerConfig,
    load_scheduler,
)

_MINI_GRID_SIZE = 8
_MINI_STEPS = 200
_MINI_SEED = 42


def _archive_snapshot(archive: GridArchive) -> dict[tuple[int, int], float]:
    """Map occupied bins to elite fitness for structural comparison."""
    resolution = archive.resolution
    snapshot: dict[tuple[int, int], float] = {}
    for i in range(resolution):
        for j in range(resolution):
            elite = archive.get(i, j)
            if elite is not None:
                snapshot[(i, j)] = elite.fitness
    return snapshot


def _run_cold_start(
    *,
    seed: int,
    config: SchedulerConfig,
    grid_size: int = _MINI_GRID_SIZE,
    steps: int = _MINI_STEPS,
) -> dict[tuple[int, int], float]:
    archive = GridArchive(config.grid_resolution)
    rng = np.random.default_rng(seed)
    run_scheduler(
        config,
        archive,
        rng,
        StubCandidateEmitter(),
        grid_size=grid_size,
        steps=steps,
    )
    return _archive_snapshot(archive)


class TestLoadMiniScheduler(unittest.TestCase):
    def test_mini_scheduler_contract(self) -> None:
        config = load_scheduler(DEFAULT_MINI_SCHEDULER_PATH)
        self.assertEqual(config.schema_version, "1.2")
        self.assertEqual(config.iterations, 20)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(len(config.batch_emitters), 4)
        self.assertFalse(config.llm_enabled)
        self.assertEqual(config.grid_resolution, 10)


class TestColdStartReproducibility(unittest.TestCase):
    def test_same_seed_identical_occupied_bins_and_fitness(self) -> None:
        config = load_scheduler(DEFAULT_MINI_SCHEDULER_PATH)
        snap_a = _run_cold_start(seed=_MINI_SEED, config=config)
        snap_b = _run_cold_start(seed=_MINI_SEED, config=config)
        self.assertEqual(snap_a, snap_b)
        self.assertGreater(len(snap_a), 0)

    def test_different_seeds_differ(self) -> None:
        config = load_scheduler(DEFAULT_MINI_SCHEDULER_PATH)
        snap_a = _run_cold_start(seed=1, config=config)
        snap_b = _run_cold_start(seed=2, config=config)
        self.assertNotEqual(snap_a, snap_b)


if __name__ == "__main__":
    unittest.main()

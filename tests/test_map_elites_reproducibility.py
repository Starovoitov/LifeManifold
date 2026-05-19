"""Integration tests for MAP-Elites cold-start reproducibility (mini scheduler)."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from worldspace.illuminators.archive import GridArchive

from worldspace.illuminators.emitters.base import EmitterOutput, MapElitesEmitter
from worldspace.illuminators.emitters.llm_emitter import LlmEmitter
from worldspace.illuminators.emitters.stub import StubCandidateEmitter
from worldspace.illuminators.loop import run_scheduler
from worldspace.illuminators.scheduler import (
    DEFAULT_MINI_SCHEDULER_PATH,
    SchedulerConfig,
    TargetBin,
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


class _FailingLlmEmitter(LlmEmitter):
    """Raises if ``emit`` is invoked (used when LLM slots are disabled)."""

    emit_calls = 0

    def emit(
        self,
        *,
        target: TargetBin,
        archive: GridArchive,
        rng: np.random.Generator,
        grid_size: int,
        steps: int,
    ) -> EmitterOutput:
        del target, archive, rng, grid_size, steps
        _FailingLlmEmitter.emit_calls += 1
        raise AssertionError("LlmEmitter.emit must not run when llm.enabled is false")


class TestLlmDisabledIntegration(unittest.TestCase):
    def setUp(self) -> None:
        _FailingLlmEmitter.emit_calls = 0

    def test_mini_scheduler_never_calls_llm_emitter(self) -> None:
        config = replace(
            load_scheduler(DEFAULT_MINI_SCHEDULER_PATH),
            initial_random_candidates=0,
            iterations=2,
        )
        self.assertFalse(config.llm_enabled)
        archive = GridArchive(config.grid_resolution)
        rng = np.random.default_rng(_MINI_SEED)
        emitter = MapElitesEmitter(
            mutation_scale=config.genetic_mutation_scale,
            scheduler=config,
            llm_emitter=_FailingLlmEmitter(
                grid_resolution=config.grid_resolution,
                surrogate_mean=0.5,
                surrogate_uncertainty=1.0,
            ),
        )
        run_scheduler(
            config,
            archive,
            rng,
            emitter,
            grid_size=_MINI_GRID_SIZE,
            steps=_MINI_STEPS,
        )
        self.assertEqual(_FailingLlmEmitter.emit_calls, 0)
        self.assertGreater(archive.filled_count(), 0)


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

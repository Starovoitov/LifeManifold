"""Integration tests for MAP-Elites cold-start reproducibility (mini scheduler)."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from worldspace.illuminators.archive import GridArchive
from worldspace.illuminators.archive_factory import (
    archive_factory_config_from_scheduler,
    create_archive,
)
from worldspace.illuminators.archive_protocol import ArchiveProtocol
from worldspace.illuminators.cvt import centroids_path_for_output, load_centroids
from worldspace.illuminators.cvt_archive import CvtArchive

from worldspace.illuminators.emitters.base import EmitterOutput, MapElitesEmitter
from worldspace.illuminators.emitters.llm_emitter import LlmEmitter
from worldspace.illuminators.emitters.stub import StubCandidateEmitter
from worldspace.illuminators.illuminator import MapElitesIlluminator, archive_jsonl_path
from worldspace.illuminators.loop import run_scheduler
from worldspace.illuminators.scheduler import (
    DEFAULT_MINI_CVT_SCHEDULER_PATH,
    DEFAULT_MINI_SCHEDULER_PATH,
    SchedulerConfig,
    TargetCell,
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
        target: TargetCell,
        archive: ArchiveProtocol,
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


def _archive_snapshot_cvt(archive: CvtArchive) -> dict[int, float]:
    """Map occupied CVT niches to elite fitness for structural comparison."""
    snapshot: dict[int, float] = {}
    for cell_id in range(archive.n_cells):
        elite = archive.get(cell_id)
        if elite is not None:
            snapshot[cell_id] = elite.fitness
    return snapshot


def _run_cold_start_cvt(
    *,
    seed: int,
    config: SchedulerConfig,
    grid_size: int = _MINI_GRID_SIZE,
    steps: int = _MINI_STEPS,
) -> dict[int, float]:
    archive = create_archive(archive_factory_config_from_scheduler(config))
    rng = np.random.default_rng(seed)
    run_scheduler(
        config,
        archive,
        rng,
        StubCandidateEmitter(),
        grid_size=grid_size,
        steps=steps,
    )
    assert isinstance(archive, CvtArchive)
    return _archive_snapshot_cvt(archive)


def _normalize_jsonl_records(path: Path) -> list[dict[str, object]]:
    """Strip nondeterministic fields for JSONL comparison."""
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        metadata = dict(record.get("metadata", {}))
        metadata.pop("id", None)
        metadata.pop("timestamp", None)
        record["metadata"] = metadata
        records.append(record)
    return records


class TestColdStartReproducibility(unittest.TestCase):
    def test_same_seed_identical_occupied_bins_and_fitness(self) -> None:
        config = load_scheduler(DEFAULT_MINI_SCHEDULER_PATH)
        snap_a = _run_cold_start(seed=_MINI_SEED, config=config)
        snap_b = _run_cold_start(seed=_MINI_SEED, config=config)
        self.assertEqual(snap_a, snap_b)
        self.assertGreater(len(snap_a), 0)

    def test_parallel_eval_matches_sequential(self) -> None:
        base = load_scheduler(DEFAULT_MINI_SCHEDULER_PATH)
        sequential = replace(
            base,
            performance=replace(base.performance, parallel_eval=False),
        )
        parallel = replace(
            base,
            performance=replace(
                base.performance,
                parallel_eval=True,
                parallel_workers=2,
            ),
        )
        snap_seq = _run_cold_start(seed=_MINI_SEED, config=sequential)
        snap_par = _run_cold_start(seed=_MINI_SEED, config=parallel)
        self.assertEqual(snap_seq, snap_par)

    def test_different_seeds_differ(self) -> None:
        config = load_scheduler(DEFAULT_MINI_SCHEDULER_PATH)
        snap_a = _run_cold_start(seed=1, config=config)
        snap_b = _run_cold_start(seed=2, config=config)
        self.assertNotEqual(snap_a, snap_b)


class TestCvtColdStartReproducibility(unittest.TestCase):
    def test_same_seed_identical_occupied_cells_and_fitness(self) -> None:
        config = load_scheduler(DEFAULT_MINI_CVT_SCHEDULER_PATH)
        snap_a = _run_cold_start_cvt(seed=_MINI_SEED, config=config)
        snap_b = _run_cold_start_cvt(seed=_MINI_SEED, config=config)
        self.assertEqual(snap_a, snap_b)
        self.assertGreater(len(snap_a), 0)

    def test_different_seeds_differ(self) -> None:
        config = load_scheduler(DEFAULT_MINI_CVT_SCHEDULER_PATH)
        snap_a = _run_cold_start_cvt(seed=1, config=config)
        snap_b = _run_cold_start_cvt(seed=2, config=config)
        self.assertNotEqual(snap_a, snap_b)

    def test_same_seed_identical_centroids_and_jsonl(self) -> None:
        config = load_scheduler(DEFAULT_MINI_CVT_SCHEDULER_PATH)
        config = replace(config, iterations=2)

        def run_once(seed: int) -> tuple[bytes, list[dict[str, object]]]:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                MapElitesIlluminator().run(
                    scheduler_path=DEFAULT_MINI_CVT_SCHEDULER_PATH,
                    output_dir=out,
                    seed=seed,
                    grid_size=_MINI_GRID_SIZE,
                    steps=_MINI_STEPS,
                    iterations=2,
                )
                centroids_bytes = centroids_path_for_output(out).read_bytes()
                jsonl_records = _normalize_jsonl_records(archive_jsonl_path(out))
                return centroids_bytes, jsonl_records

        centroids_a, jsonl_a = run_once(_MINI_SEED)
        centroids_b, jsonl_b = run_once(_MINI_SEED)
        self.assertEqual(centroids_a, centroids_b)
        self.assertEqual(jsonl_a, jsonl_b)
        self.assertGreater(len(jsonl_a), 0)

    def test_centroids_match_cvt_seed(self) -> None:
        config = load_scheduler(DEFAULT_MINI_CVT_SCHEDULER_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            MapElitesIlluminator().run(
                scheduler_path=DEFAULT_MINI_CVT_SCHEDULER_PATH,
                output_dir=out,
                seed=_MINI_SEED,
                grid_size=_MINI_GRID_SIZE,
                steps=_MINI_STEPS,
                iterations=1,
            )
            centroids_path = centroids_path_for_output(out)
            loaded = load_centroids(centroids_path)
            archive = create_archive(
                archive_factory_config_from_scheduler(config),
                output_dir=out,
            )
            np.testing.assert_allclose(loaded, archive.centroids)


if __name__ == "__main__":
    unittest.main()

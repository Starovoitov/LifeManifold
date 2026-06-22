"""Unit and integration tests for the MAP-Elites iteration loop."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from worldspace.illuminators.archive import GridArchive, new_elite_metadata
from worldspace.illuminators.archive_protocol import ArchiveProtocol
from worldspace.illuminators.emitters.base import EmitterOutput
from worldspace.illuminators.emitters.stub import StubCandidateEmitter
from worldspace.illuminators.loop import run_iteration, run_scheduler
from worldspace.illuminators.scheduler import (
    RunCounters,
    SchedulerConfig,
    TargetCell,
)
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec

_MINI_CONFIG = SchedulerConfig(
    schema_version="1.2",
    iterations=2,
    batch_size=4,
    grid_resolution=5,
    early_extinction_step=200,
    min_steps=200,
    batch_emitters=("random", "genetic", "genetic", "llm"),
    initial_random_candidates=100,
    llm_enabled=True,
    surrogate_enabled=False,
    surrogate_model_type="lightgbm",
    surrogate_checkpoint="artifacts/surrogate/checkpoints/latest.pkl",
    surrogate_buffer_path="artifacts/surrogate/buffer.jsonl",
    surrogate_stub_mean=0.5,
    surrogate_stub_uncertainty=1.0,
    genetic_mutation_scale=0.02,
)

_FIXED_SPEC = WorldSpec(
    birth=[1, 3],
    survival=[2, 3],
    noise=0.02,
    resource_regen=0.05,
    predation=0.1,
    cell_types=CANONICAL_CELL_TYPES.copy(),
    grid_size=8,
    steps=200,
    seed=0,
)


class FixedSpecEmitter:
    """Deterministic emitter for tie-break and reproducibility tests."""

    def __init__(
        self, world_spec: WorldSpec, *, elite_id_prefix: str = "fixed"
    ) -> None:
        self._spec = world_spec
        self._prefix = elite_id_prefix
        self._calls = 0

    def emit(
        self,
        *,
        emitter_kind: str,
        target: TargetCell,
        archive: ArchiveProtocol,
        rng: np.random.Generator,
        grid_size: int,
        steps: int,
    ) -> EmitterOutput:
        del target, archive, rng
        elite_id = f"{self._prefix}-{self._calls}"
        self._calls += 1
        spec = replace(
            self._spec,
            grid_size=grid_size,
            steps=steps,
            seed=0,
        )
        metadata = new_elite_metadata(
            generated_by=emitter_kind,
            emitter_type=emitter_kind,
            parent_id=None,
            elite_id=elite_id,
        )
        return EmitterOutput(world_spec=spec, metadata=metadata)


class TestRunIteration(unittest.TestCase):
    def test_evaluates_batch_size_candidates(self) -> None:
        archive = GridArchive(_MINI_CONFIG.grid_resolution)
        counters = RunCounters()
        stats, outcomes = run_iteration(
            _MINI_CONFIG,
            archive,
            np.random.default_rng(1),
            counters,
            StubCandidateEmitter(),
            iteration_index=1,
            grid_size=8,
            steps=200,
        )
        self.assertEqual(stats.evaluations, 4)
        self.assertEqual(counters.candidates_evaluated, 4)
        self.assertEqual(len(outcomes), 4)

    def test_candidate_id_order(self) -> None:
        _, outcomes = run_iteration(
            _MINI_CONFIG,
            GridArchive(5),
            np.random.default_rng(0),
            RunCounters(),
            StubCandidateEmitter(),
            iteration_index=1,
            grid_size=8,
            steps=200,
        )
        self.assertEqual([o.candidate_id for o in outcomes], [0, 1, 2, 3])

    def test_jsonl_only_on_accept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "archive.jsonl"
            config = replace(_MINI_CONFIG, batch_size=2)
            run_iteration(
                config,
                GridArchive(5),
                np.random.default_rng(3),
                RunCounters(),
                StubCandidateEmitter(),
                iteration_index=1,
                grid_size=8,
                steps=200,
                jsonl_path=path,
            )
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(lines), 1)
            for line in lines:
                record = json.loads(line)
                self.assertEqual(record["schema_version"], "1.2")

    def test_run_scheduler_iterations_accumulate_counters(self) -> None:
        config = replace(_MINI_CONFIG, iterations=3)
        counters = run_scheduler(
            config,
            GridArchive(5),
            np.random.default_rng(9),
            StubCandidateEmitter(),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(counters.candidates_evaluated, 3 * config.batch_size)


class TestBatchEqualFitness(unittest.TestCase):
    def test_second_slot_rejected_same_bin_same_fitness(self) -> None:
        config = replace(_MINI_CONFIG, batch_size=2, initial_random_candidates=0)
        archive = GridArchive(config.grid_resolution)
        counters = RunCounters()
        stats, outcomes = run_iteration(
            config,
            archive,
            np.random.default_rng(0),
            counters,
            FixedSpecEmitter(_FIXED_SPEC),
            iteration_index=1,
            grid_size=8,
            steps=200,
        )
        self.assertEqual(stats.evaluated, 2)
        self.assertEqual(stats.accepted, 1)
        self.assertEqual(stats.rejected, 1)
        assert outcomes[0].insert is not None and outcomes[1].insert is not None
        assert (
            outcomes[0].eval_result is not None and outcomes[1].eval_result is not None
        )
        self.assertTrue(outcomes[0].insert.accepted)
        self.assertTrue(outcomes[1].insert.rejected)
        self.assertEqual(outcomes[0].eval_result.bin, outcomes[1].eval_result.bin)
        self.assertEqual(
            outcomes[0].eval_result.fitness,
            outcomes[1].eval_result.fitness,
        )
        stored = archive.get(*outcomes[0].eval_result.bin)
        assert stored is not None and stored.metadata is not None
        self.assertEqual(stored.metadata.id, "fixed-0")

    def test_jsonl_single_line_on_equal_fitness_tie(self) -> None:
        config = replace(_MINI_CONFIG, batch_size=2, initial_random_candidates=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "archive.jsonl"
            run_iteration(
                config,
                GridArchive(config.grid_resolution),
                np.random.default_rng(0),
                RunCounters(),
                FixedSpecEmitter(_FIXED_SPEC),
                iteration_index=1,
                grid_size=8,
                steps=200,
                jsonl_path=path,
            )
            lines = [
                line for line in path.read_text(encoding="utf-8").splitlines() if line
            ]
            self.assertEqual(len(lines), 1)


if __name__ == "__main__":
    unittest.main()

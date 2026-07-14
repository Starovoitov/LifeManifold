"""Unit and integration tests for the MAP-Elites iteration loop."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from worldspace.illuminators.archive import GridArchive, new_elite_metadata
from worldspace.illuminators.archive_protocol import ArchiveProtocol
from worldspace.illuminators.emitters.base import EmitterOutput, MapElitesEmitter
from worldspace.illuminators.emitters.llm_emitter import LlmEmitter
from worldspace.illuminators.emitters.stub import StubCandidateEmitter
from worldspace.illuminators.loop import run_iteration, run_scheduler
from worldspace.illuminators.scheduler import (
    RunCounters,
    SchedulerConfig,
    TargetCell,
)
from worldspace.simulator_perf import DEFAULT_SIMULATOR_PERFORMANCE
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

_LLM_MOCK_RESPONSE = json.dumps(
    {
        "world_spec": {
            "birth": [2],
            "survival": [2, 3],
            "noise": 0.04,
            "resource_regen": 0.06,
            "predation": 0.12,
            "cell_types": ["life", "food"],
            "neighborhood": "moore",
        },
    }
)

_LLM_PARALLEL_CONFIG = replace(
    _MINI_CONFIG,
    batch_emitters=("random", "llm", "llm", "genetic"),
    initial_random_candidates=0,
    performance=replace(DEFAULT_SIMULATOR_PERFORMANCE, llm_parallel_emit=True),
)


def _llm_mock_emitter(*, sleep_s: float = 0.0) -> MapElitesEmitter:
    def mock_llm(**_: object) -> str:
        if sleep_s:
            time.sleep(sleep_s)
        return _LLM_MOCK_RESPONSE

    return MapElitesEmitter(
        mutation_scale=_LLM_PARALLEL_CONFIG.genetic_mutation_scale,
        llm_emitter=LlmEmitter(
            grid_resolution=_LLM_PARALLEL_CONFIG.grid_resolution,
            call_llm_text=mock_llm,
        ),
    )


def _draft_specs(
    config: SchedulerConfig,
    *,
    llm_parallel: bool,
    seed: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    from worldspace.illuminators.loop import _emit_iteration_drafts

    perf = replace(
        config.performance,
        llm_parallel_emit=llm_parallel,
    )
    cfg = replace(config, performance=perf)
    archive = GridArchive(cfg.grid_resolution)
    rng = np.random.default_rng(seed)
    emitter = _llm_mock_emitter()
    drafts = _emit_iteration_drafts(
        cfg,
        archive,
        rng,
        emitter,
        grid_size=8,
        steps=200,
        counters=RunCounters(candidates_evaluated=100),
    )
    return [(d.spec.birth, d.spec.survival) for d in drafts]


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


class TestMiniCvtLoop(unittest.TestCase):
    def test_run_scheduler_with_map_elites_emitter_fills_archive(self) -> None:
        from worldspace.illuminators.archive_factory import (
            archive_factory_config_from_scheduler,
            create_archive,
        )
        from worldspace.illuminators.cvt import centroids_path_for_output
        from worldspace.illuminators.emitters.base import MapElitesEmitter
        from worldspace.illuminators.scheduler import (
            DEFAULT_MINI_CVT_SCHEDULER_PATH,
            load_scheduler,
        )

        config = load_scheduler(DEFAULT_MINI_CVT_SCHEDULER_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            archive = create_archive(
                archive_factory_config_from_scheduler(config),
                output_dir=out,
            )
            emitter = MapElitesEmitter(
                mutation_scale=config.genetic_mutation_scale,
                scheduler=config,
            )
            counters = run_scheduler(
                config,
                archive,
                np.random.default_rng(42),
                emitter,
                grid_size=8,
                steps=200,
            )
            self.assertEqual(
                counters.candidates_evaluated,
                config.iterations * config.batch_size,
            )
            self.assertGreater(archive.filled_count(), 0)
            self.assertLessEqual(archive.filled_count(), config.n_centroids)
            self.assertTrue(centroids_path_for_output(out).is_file())


class TestParallelLlmEmit(unittest.TestCase):
    def test_parallel_llm_emit_matches_sequential_with_deterministic_mock(self) -> None:
        seq = _draft_specs(_LLM_PARALLEL_CONFIG, llm_parallel=False, seed=7)
        par = _draft_specs(_LLM_PARALLEL_CONFIG, llm_parallel=True, seed=7)
        self.assertEqual(seq, par)

    def test_parallel_llm_emit_faster_than_sequential(self) -> None:
        sleep_s = 0.04

        def timed(parallel: bool) -> float:
            from worldspace.illuminators.loop import _emit_iteration_drafts

            perf = replace(
                _LLM_PARALLEL_CONFIG.performance,
                llm_parallel_emit=parallel,
            )
            cfg = replace(_LLM_PARALLEL_CONFIG, performance=perf)

            def mock_llm(**_: object) -> str:
                time.sleep(sleep_s)
                return _LLM_MOCK_RESPONSE

            emitter = MapElitesEmitter(
                mutation_scale=cfg.genetic_mutation_scale,
                llm_emitter=LlmEmitter(
                    grid_resolution=cfg.grid_resolution,
                    call_llm_text=mock_llm,
                ),
            )
            t0 = time.monotonic()
            _emit_iteration_drafts(
                cfg,
                GridArchive(cfg.grid_resolution),
                np.random.default_rng(11),
                emitter,
                grid_size=8,
                steps=200,
                counters=RunCounters(candidates_evaluated=100),
            )
            return time.monotonic() - t0

        sequential_s = timed(False)
        parallel_s = timed(True)
        self.assertGreater(sequential_s, sleep_s * 1.5)
        self.assertLess(parallel_s, sequential_s * 0.75)

    def test_llm_parallel_disabled_uses_sequential_path(self) -> None:
        with patch(
            "worldspace.illuminators.loop._emit_iteration_drafts_parallel_llm",
        ) as parallel_mock:
            _draft_specs(_LLM_PARALLEL_CONFIG, llm_parallel=False, seed=5)
        parallel_mock.assert_not_called()


class TestLlmEmitCounters(unittest.TestCase):
    def test_run_iteration_records_llm_emit_and_fallback_counters(self) -> None:
        config = replace(
            _MINI_CONFIG,
            batch_size=2,
            batch_emitters=("llm", "llm"),
            initial_random_candidates=0,
        )
        call_count = 0

        def mock_llm(**_: object) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _LLM_MOCK_RESPONSE
            return ""

        emitter = MapElitesEmitter(
            mutation_scale=config.genetic_mutation_scale,
            llm_emitter=LlmEmitter(
                grid_resolution=config.grid_resolution,
                call_llm_text=mock_llm,
            ),
        )
        counters = RunCounters()
        run_iteration(
            config,
            GridArchive(config.grid_resolution),
            np.random.default_rng(0),
            counters,
            emitter,
            iteration_index=1,
            grid_size=8,
            steps=200,
        )
        self.assertEqual(counters.llm_emit_attempts, 2)
        self.assertEqual(counters.llm_emit_fallbacks, 1)
        self.assertGreater(counters.emit_llm_seconds, 0.0)
        self.assertGreater(counters.eval_seconds, 0.0)

    def test_iteration_timing_jsonl_written_when_enabled(self) -> None:
        config = replace(
            _MINI_CONFIG,
            llm_enabled=False,
            batch_emitters=("random", "random"),
            batch_size=2,
            initial_random_candidates=0,
            iterations=2,
            performance=replace(
                DEFAULT_SIMULATOR_PERFORMANCE,
                log_iteration_timing=True,
            ),
        )
        emitter = FixedSpecEmitter(_FIXED_SPEC)
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "archive.jsonl"
            run_scheduler(
                config,
                GridArchive(config.grid_resolution),
                np.random.default_rng(0),
                emitter,
                grid_size=8,
                steps=200,
                jsonl_path=jsonl_path,
            )
            timing_path = Path(tmp) / "iteration_timing.jsonl"
            self.assertTrue(timing_path.is_file())
            lines = [
                line
                for line in timing_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(lines), config.iterations)
            first = json.loads(lines[0])
            self.assertIn("emit_s", first)
            self.assertIn("eval_s", first)
            trace_path = Path(tmp) / "archive_trace.jsonl"
            self.assertTrue(trace_path.is_file())
            trace_lines = [
                line
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(trace_lines), config.iterations + 1)
            trace_first = json.loads(trace_lines[0])
            self.assertEqual(trace_first["iteration"], 0)
            self.assertIn("coverage", trace_first)
            self.assertIn("filled_cells", trace_first)
            trace_last = json.loads(trace_lines[-1])
            self.assertEqual(trace_last["iteration"], config.iterations)
            self.assertEqual(
                trace_last["evaluations"], config.iterations * config.batch_size
            )


if __name__ == "__main__":
    unittest.main()

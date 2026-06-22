"""Integration tests for acquisition modes in the illuminator loop."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from worldspace.illuminators.archive import GridArchive
from worldspace.illuminators.emitters.stub import StubCandidateEmitter
from worldspace.illuminators.loop import run_iteration
from worldspace.illuminators.scheduler import RunCounters, SchedulerConfig
from worldspace.surrogate.acquisition_config import AcquisitionConfig, RetrainConfig
from worldspace.surrogate.surrogate import StubSurrogate
from worldspace.surrogate.surrogate_archive import SurrogateArchiveWriter
from worldspace.surrogate.types import SurrogatePrediction
from worldspace.specs.spec import WorldSpec

_LOW_COMPONENTS = {
    "stability": 0.1,
    "diversity": 0.1,
    "oscillation_score": 0.1,
    "topology_interface_index": 0.1,
    "topology_window_heterogeneity": 0.1,
    "final_density": 0.1,
    "early_extinction_prob": 0.1,
}


class _LowFitnessSurrogate:
    def predict(self, world_spec: WorldSpec) -> SurrogatePrediction:
        _ = world_spec
        return SurrogatePrediction(
            components=dict(_LOW_COMPONENTS),
            measures={"stability": 0.1, "diversity": 0.1},
            fitness=0.1,
            uncertainty=0.1,
        )


def _filter_config(*, batch_size: int = 4) -> SchedulerConfig:
    return SchedulerConfig(
        schema_version="1.2",
        iterations=1,
        batch_size=batch_size,
        grid_resolution=5,
        early_extinction_step=200,
        min_steps=200,
        batch_emitters=("random",) * batch_size,
        initial_random_candidates=100,
        llm_enabled=False,
        surrogate_enabled=True,
        surrogate_model_type="lightgbm",
        surrogate_checkpoint="artifacts/surrogate/checkpoints/micro.pkl",
        surrogate_buffer_path="artifacts/surrogate/buffer.jsonl",
        surrogate_stub_mean=0.5,
        surrogate_stub_uncertainty=1.0,
        genetic_mutation_scale=0.02,
        acquisition=AcquisitionConfig(
            mode="filter",
            min_predicted_fitness=0.25,
            max_uncertainty_to_skip=0.40,
            never_skip_empty_bin=False,
        ),
        retrain=RetrainConfig(enabled=False),
    )


class TestLoopAcquisition(unittest.TestCase):
    def test_filter_skips_without_eval_or_buffer(self) -> None:
        config = _filter_config(batch_size=4)
        counters = RunCounters()
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            archive_path = Path(tmpdir) / "surrogate_archive.jsonl"
            from worldspace.surrogate.buffer import SurrogateBuffer

            buffer = SurrogateBuffer(path=buffer_path, flush_every=32)
            archive_writer = SurrogateArchiveWriter(
                path=archive_path,
                run_id="test-run",
                flush_every=32,
            )
            stats, outcomes = run_iteration(
                config,
                GridArchive(5),
                np.random.default_rng(0),
                counters,
                StubCandidateEmitter(),
                iteration_index=1,
                grid_size=8,
                steps=200,
                surrogate_buffer=buffer,
                surrogate=_LowFitnessSurrogate(),
                surrogate_archive=archive_writer,
            )
            buffer.flush()
            archive_writer.flush()
            archive_writer.close()
            self.assertEqual(stats.evaluated, 0)
            self.assertEqual(stats.skipped, 4)
            self.assertEqual(stats.evaluated + stats.skipped, config.batch_size)
            self.assertEqual(counters.candidates_evaluated, 0)
            self.assertFalse(buffer_path.is_file())
            self.assertTrue(archive_path.is_file())
            self.assertEqual(len(archive_path.read_text().strip().splitlines()), 4)

    def test_shadow_always_evaluates_and_logs_would_skip(self) -> None:
        config = replace(
            _filter_config(batch_size=2),
            acquisition=AcquisitionConfig(
                mode="shadow",
                never_skip_empty_bin=False,
            ),
        )
        counters = RunCounters()
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "surrogate_archive.jsonl"
            archive_writer = SurrogateArchiveWriter(
                path=archive_path,
                run_id="shadow-run",
                flush_every=32,
            )
            stats, outcomes = run_iteration(
                config,
                GridArchive(5),
                np.random.default_rng(1),
                counters,
                StubCandidateEmitter(),
                iteration_index=1,
                grid_size=8,
                steps=200,
                surrogate=_LowFitnessSurrogate(),
                surrogate_archive=archive_writer,
            )
            archive_writer.close()
            self.assertEqual(stats.evaluated, 2)
            self.assertEqual(stats.skipped, 0)
            self.assertEqual(stats.shadow_would_skip, 2)
            self.assertTrue(all(not o.skipped for o in outcomes))

    def test_skips_do_not_advance_initial_random_phase(self) -> None:
        config = replace(
            _filter_config(batch_size=4),
            initial_random_candidates=4,
            iterations=1,
        )
        counters = RunCounters()
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            from worldspace.surrogate.buffer import SurrogateBuffer

            stats, _ = run_iteration(
                config,
                GridArchive(5),
                np.random.default_rng(0),
                counters,
                StubCandidateEmitter(),
                iteration_index=1,
                grid_size=8,
                steps=200,
                surrogate_buffer=SurrogateBuffer(path=buffer_path, flush_every=32),
                surrogate=_LowFitnessSurrogate(),
            )
            self.assertEqual(stats.skipped, 4)
            self.assertEqual(counters.candidates_evaluated, 0)

    def test_off_mode_matches_legacy_eval_count(self) -> None:
        config = replace(
            _filter_config(batch_size=2),
            acquisition=AcquisitionConfig(mode="off"),
        )
        counters = RunCounters()
        stats, _ = run_iteration(
            config,
            GridArchive(5),
            np.random.default_rng(2),
            counters,
            StubCandidateEmitter(),
            iteration_index=1,
            grid_size=8,
            steps=200,
            surrogate=StubSurrogate(mean=0.5, uncertainty=0.85),
        )
        self.assertEqual(stats.evaluated, 2)
        self.assertEqual(stats.skipped, 0)
        self.assertEqual(counters.candidates_evaluated, 2)

    def test_filter_skips_cvt_archive_without_eval(self) -> None:
        from worldspace.illuminators.cvt import generate_centroids
        from worldspace.illuminators.cvt_archive import CvtArchive

        config = replace(
            _filter_config(batch_size=2),
            schema_version="1.3",
            archive_type="cvt",
            n_centroids=9,
            cvt_seed=0,
            lloyd_iterations=5,
        )
        archive = CvtArchive(generate_centroids(9, seed=0, lloyd_iterations=5))
        counters = RunCounters()
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "surrogate_archive.jsonl"
            archive_writer = SurrogateArchiveWriter(
                path=archive_path,
                run_id="cvt-filter-run",
                flush_every=32,
            )
            stats, outcomes = run_iteration(
                config,
                archive,
                np.random.default_rng(3),
                counters,
                StubCandidateEmitter(),
                iteration_index=1,
                grid_size=8,
                steps=200,
                surrogate=_LowFitnessSurrogate(),
                surrogate_archive=archive_writer,
            )
            archive_writer.close()
            self.assertEqual(stats.evaluated, 0)
            self.assertEqual(stats.skipped, 2)
            self.assertTrue(all(o.skipped for o in outcomes))


if __name__ == "__main__":
    unittest.main()

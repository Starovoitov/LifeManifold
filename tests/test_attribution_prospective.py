"""Prospective all-slot capture contract smokes for CA and maze."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from worldspace.attribution import (
    PROSPECTIVE_EVENT_FILENAME,
    ProspectiveEventCapture,
)
from worldspace.attribution.adapters import (
    CaNormalizationAdapter,
    MazeNormalizationAdapter,
    NativeRunInputs,
)
from worldspace.attribution.capabilities import ca_capabilities, maze_capabilities
from worldspace.illuminators.archive import GridArchive
from worldspace.illuminators.emitters.stub import StubCandidateEmitter
from worldspace.illuminators.loop import run_scheduler
from worldspace.illuminators.proposal_log import configure_proposal_log
from worldspace.illuminators.scheduler import (
    DEFAULT_MINI_SCHEDULER_PATH,
    load_scheduler,
)
from worldspace.mazes.runner import MazeSchedulerConfig, run_maze_qd
from worldspace.scripts.run_map_elites_nightly import run_map_elites_nightly

from tests.test_attribution_normalizers import _run_manifest


class TestProspectiveAttributionCapture(unittest.TestCase):
    def test_ca_and_maze_declare_full_prospective_slot_support(self) -> None:
        self.assertTrue(ca_capabilities().supports_full_proposal_log)
        self.assertTrue(maze_capabilities().supports_full_proposal_log)

    def test_ca_all_slots_normalize_as_full_without_metric_drift(self) -> None:
        self.addCleanup(configure_proposal_log, None)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = _run_manifest(
                "ca",
                seed=17,
                selector="min_fitness_frontier",
                generator="random",
            )
            capture = ProspectiveEventCapture(
                manifest,
                run_dir / PROSPECTIVE_EVENT_FILENAME,
            )
            env = {
                "LIFEMANIFOLD_PROPOSAL_LOG_ALL_EMITTERS": "1",
                "LIFEMANIFOLD_LLM_CALL_LOG": "0",
            }
            with patch.dict("os.environ", env, clear=False):
                native = run_map_elites_nightly(
                    scheduler_path=DEFAULT_MINI_SCHEDULER_PATH,
                    output_dir=run_dir,
                    seed=17,
                    grid_size=8,
                    steps=200,
                    iterations=1,
                    attribution_capture=capture,
                )

            bundle = CaNormalizationAdapter().normalize(
                manifest,
                NativeRunInputs(run_dir),
            )

            self.assertEqual(bundle.summary.event_completeness, "full")
            self.assertEqual(len(bundle.events), native.evaluations)
            self.assertEqual(
                sum(event.evaluation.completed for event in bundle.events),
                native.evaluations,
            )
            self.assertTrue(
                all(
                    event.resources.evaluator_seconds is not None
                    and event.resources.evaluator_seconds > 0.0
                    for event in bundle.events
                    if event.evaluation.attempted
                )
            )
            self.assertEqual(
                bundle.summary.final_counters.proposal_slots,
                len(bundle.events),
            )
            self.assertEqual(
                bundle.summary.final_archive.occupied_cells,
                native.filled_cells,
            )
            self.assertAlmostEqual(
                bundle.events[-1].after.raw_qd_score or 0.0,
                bundle.summary.final_archive.raw_qd_score or 0.0,
            )
            self.assertIn(
                "prospective_attribution_events",
                {entry.logical_name for entry in bundle.artifacts.artifacts},
            )

    def test_ca_parallel_capture_preserves_worker_evaluation_timing(self) -> None:
        config = load_scheduler(DEFAULT_MINI_SCHEDULER_PATH)
        config = replace(
            config,
            iterations=1,
            performance=replace(
                config.performance,
                parallel_eval=True,
                parallel_workers=2,
            ),
        )
        manifest = _run_manifest(
            "ca",
            seed=29,
            selector="min_fitness_frontier",
            generator="random",
        )
        capture = ProspectiveEventCapture(manifest)

        run_scheduler(
            config,
            GridArchive(config.grid_resolution),
            np.random.default_rng(29),
            StubCandidateEmitter(),
            grid_size=8,
            steps=200,
            attribution_capture=capture,
        )

        self.assertEqual(len(capture.events), config.batch_size)
        self.assertTrue(
            all(
                event.resources.evaluator_seconds is not None
                and event.resources.evaluator_seconds > 0.0
                for event in capture.events
            )
        )

    def test_maze_capture_includes_evaluated_and_skipped_slots(self) -> None:
        from worldspace.mazes.surrogate import MazePrediction
        from worldspace.surrogate.acquisition_config import AcquisitionConfig

        class LowPredictor:
            def predict(self, spec: object) -> MazePrediction:
                del spec
                return MazePrediction(
                    {},
                    {"path_length": 0.0, "branching": 0.0},
                    0.0,
                    0.0,
                )

        config = MazeSchedulerConfig(
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
            run_dir = Path(tmp)
            manifest = _run_manifest(
                "maze",
                seed=23,
                selector="uniform_frontier",
                generator="genetic",
                gate="filter",
            )
            capture = ProspectiveEventCapture(
                manifest,
                run_dir / PROSPECTIVE_EVENT_FILENAME,
            )
            native = run_maze_qd(
                config,
                seed=23,
                output_dir=run_dir,
                predictor=LowPredictor(),  # type: ignore[arg-type]
                attribution_capture=capture,
            )

            bundle = MazeNormalizationAdapter().normalize(
                manifest,
                NativeRunInputs(run_dir),
            )

            self.assertGreater(native.skipped, 0)
            self.assertEqual(bundle.summary.event_completeness, "full")
            self.assertEqual(len(bundle.events), native.proposals)
            self.assertEqual(
                sum(not event.evaluation.attempted for event in bundle.events),
                native.skipped,
            )
            self.assertEqual(
                sum(event.evaluation.completed for event in bundle.events),
                native.evaluations,
            )
            self.assertEqual(
                bundle.summary.final_counters.valid_proposals,
                native.evaluations,
            )
            self.assertEqual(
                bundle.summary.counter_completeness["evaluator_wall_time"],
                "observed",
            )
            self.assertAlmostEqual(
                bundle.summary.final_counters.evaluator_seconds or 0.0,
                sum(
                    event.resources.evaluator_seconds or 0.0 for event in bundle.events
                ),
            )
            self.assertEqual(
                bundle.summary.final_archive.occupied_cells,
                native.filled_cells,
            )
            self.assertAlmostEqual(
                bundle.summary.final_archive.raw_qd_score or 0.0,
                native.qd_score,
            )


if __name__ == "__main__":
    unittest.main()

"""Prospective all-slot capture contract smokes for CA and maze."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from worldspace.attribution import (
    BUDGET_LEDGER_FILENAME,
    PROSPECTIVE_EVENT_FILENAME,
    ProspectiveEventCapture,
    event_budget_counters,
    reconcile_event_ledger,
)
from worldspace.attribution.adapters import (
    CaNormalizationAdapter,
    MazeNormalizationAdapter,
    NativeRunInputs,
    NormalizationError,
)
from worldspace.attribution.capabilities import ca_capabilities, maze_capabilities
from worldspace.illuminators.archive import GridArchive
from worldspace.illuminators.archive import new_elite_metadata
from worldspace.illuminators.emitters.base import EmitterOutput
from worldspace.illuminators.emitters.stub import StubCandidateEmitter
from worldspace.illuminators.loop import run_scheduler
from worldspace.illuminators.proposal_log import configure_proposal_log
from worldspace.illuminators.scheduler import (
    DEFAULT_MINI_SCHEDULER_PATH,
    load_scheduler,
)
from worldspace.generators.llm_call_log import (
    append_llm_call_record,
    configure_llm_call_log,
)
from worldspace.mazes.runner import MazeSchedulerConfig, run_maze_qd
from worldspace.scripts.run_map_elites_nightly import run_map_elites_nightly
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec

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
            self.assertEqual(len(bundle.checkpoints), len(bundle.events))
            self.assertTrue(
                all(
                    checkpoint.indexed_by == "proposal"
                    for checkpoint in bundle.checkpoints
                )
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
            ledger = reconcile_event_ledger(
                bundle.events,
                bundle.summary,
                llm_applicable=False,
            )
            self.assertEqual(ledger.proposal_slots, native.evaluations)
            self.assertEqual(ledger.evaluator_completions, native.evaluations)
            self.assertIsNone(ledger.total_tokens)
            ledger_path = run_dir / BUDGET_LEDGER_FILENAME
            ledger_rows = [
                json.loads(line)
                for line in ledger_path.read_text(encoding="utf-8").splitlines()
            ]
            ledger_rows[-1]["counters"]["proposal_slots"] += 1
            ledger_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger_rows),
                encoding="utf-8",
            )
            with self.assertRaises(NormalizationError):
                CaNormalizationAdapter().normalize(
                    manifest,
                    NativeRunInputs(run_dir),
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

    def test_ca_prospective_ledger_reconciles_logged_llm_usage(self) -> None:
        class LoggedLlmEmitter:
            def __init__(self) -> None:
                self.index = 0

            def emit(self, **_: object) -> EmitterOutput:
                index = self.index
                self.index += 1
                spec = WorldSpec(
                    birth=[1, 3],
                    survival=[2, 3],
                    noise=0.02,
                    resource_regen=0.05,
                    predation=0.1,
                    cell_types=CANONICAL_CELL_TYPES.copy(),
                    grid_size=8,
                    steps=200,
                    seed=index,
                )
                return EmitterOutput(
                    world_spec=spec,
                    metadata=new_elite_metadata(
                        generated_by="llm",
                        emitter_type="llm",
                        parent_id=None,
                        elite_id=f"llm-{index}",
                    ),
                    llm_call_id=f"fixture-call-{index}",
                    llm_parse_outcome="valid",
                )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self.addCleanup(configure_llm_call_log, None)
            configure_llm_call_log(run_dir / "llm_call_log.jsonl")
            config = load_scheduler(DEFAULT_MINI_SCHEDULER_PATH)
            config = replace(
                config,
                iterations=1,
                initial_random_candidates=0,
                batch_emitters=("llm",) * config.batch_size,
                llm_enabled=True,
            )
            for index in range(config.batch_size):
                append_llm_call_record(
                    {
                        "call_id": f"fixture-call-{index}",
                        "ok": True,
                        "latency_ms": 25.0,
                        "usage": {
                            "prompt_tokens": 7,
                            "completion_tokens": 3,
                            "total_tokens": 10,
                        },
                    }
                )
            manifest = _run_manifest(
                "ca",
                seed=31,
                selector="min_fitness_frontier",
                generator="llm",
            )
            capture = ProspectiveEventCapture(
                manifest,
                run_dir / PROSPECTIVE_EVENT_FILENAME,
            )

            run_scheduler(
                config,
                GridArchive(config.grid_resolution),
                np.random.default_rng(31),
                LoggedLlmEmitter(),  # type: ignore[arg-type]
                grid_size=8,
                steps=200,
                attribution_capture=capture,
            )

            terminal = capture.checkpoints[-1].counters
            self.assertEqual(terminal.llm_attempts, config.batch_size)
            self.assertEqual(terminal.llm_completions, config.batch_size)
            self.assertEqual(terminal.prompt_tokens, 7 * config.batch_size)
            self.assertEqual(terminal.completion_tokens, 3 * config.batch_size)
            self.assertEqual(terminal.total_tokens, 10 * config.batch_size)
            self.assertAlmostEqual(
                terminal.llm_latency_seconds or 0.0,
                0.025 * config.batch_size,
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
            self.assertEqual(len(bundle.checkpoints), native.proposals)
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
            ledger = event_budget_counters(
                bundle.events,
                llm_applicable=False,
            )
            self.assertEqual(ledger.proposal_slots, native.proposals)
            self.assertEqual(ledger.valid_proposals, native.evaluations)
            self.assertEqual(ledger.evaluator_completions, native.evaluations)
            self.assertIsNone(ledger.total_tokens)
            self.assertIn(
                "prospective_budget_ledger",
                {entry.logical_name for entry in bundle.artifacts.artifacts},
            )


if __name__ == "__main__":
    unittest.main()

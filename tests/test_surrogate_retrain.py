"""Unit and integration tests for nested surrogate retrain (SA-8)."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from dataclasses import replace  # used for config and WorldSpec copies
from pathlib import Path
from unittest import mock

import numpy as np

from worldspace.illuminators.archive import GridArchive
from worldspace.illuminators.loop import run_scheduler
from worldspace.illuminators.scheduler import SchedulerConfig
from worldspace.surrogate.acquisition_config import RetrainConfig
from worldspace.surrogate.buffer import count_buffer_rows
from worldspace.surrogate.retrain import (
    RetrainState,
    is_retrain_iteration,
    maybe_retrain_after_iteration,
)
from worldspace.surrogate.surrogate import (
    StubSurrogate,
    SurrogateFacade,
    build_surrogate_facade,
)
from worldspace.surrogate.training_runtime import TrainResult, train_from_buffer
from worldspace.illuminators.emitters.stub import StubCandidateEmitter
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec

_FIXED_SPEC = WorldSpec(
    birth=[1, 3],
    survival=[2, 3],
    noise=0.02,
    resource_regen=0.05,
    predation=0.1,
    cell_types=CANONICAL_CELL_TYPES.copy(),
    grid_size=8,
    steps=200,
    seed=1,
)

_MINI_CONFIG = SchedulerConfig(
    schema_version="1.2",
    iterations=3,
    batch_size=2,
    grid_resolution=5,
    early_extinction_step=200,
    min_steps=200,
    batch_emitters=("random", "random"),
    initial_random_candidates=0,
    llm_enabled=False,
    surrogate_enabled=True,
    surrogate_model_type="lightgbm",
    surrogate_checkpoint="artifacts/surrogate/checkpoints/latest.pkl",
    surrogate_buffer_path="artifacts/surrogate/buffer.jsonl",
    surrogate_stub_mean=0.5,
    surrogate_stub_uncertainty=0.85,
    genetic_mutation_scale=0.1,
)


class TestRetrainHelpers(unittest.TestCase):
    def test_is_retrain_iteration(self) -> None:
        self.assertFalse(is_retrain_iteration(0, 50))
        self.assertFalse(is_retrain_iteration(1, 50))
        self.assertTrue(is_retrain_iteration(50, 50))
        self.assertTrue(is_retrain_iteration(100, 50))

    def test_count_buffer_rows_empty_and_nonempty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.jsonl"
            self.assertEqual(count_buffer_rows(missing), 0)
            path = Path(tmpdir) / "buffer.jsonl"
            path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
            self.assertEqual(count_buffer_rows(path), 2)

    def test_count_buffer_rows_invalid_utf8_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.jsonl"
            path.write_bytes(b"\xff\xfe\n")
            self.assertEqual(count_buffer_rows(path), 0)

    def test_count_buffer_rows_permission_error_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(
                Path,
                "open",
                side_effect=PermissionError("denied"),
            ):
                self.assertEqual(count_buffer_rows(path), 0)


class TestMaybeRetrainAfterIteration(unittest.TestCase):
    def test_skips_when_not_enough_new_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            buffer_path.write_text('{"row": 1}\n', encoding="utf-8")
            config = replace(
                _MINI_CONFIG,
                surrogate_buffer_path=str(buffer_path),
                retrain=RetrainConfig(
                    enabled=True,
                    every_iterations=1,
                    min_new_buffer_rows=500,
                ),
            )
            state = RetrainState(buffer_row_count_at_last_retrain=1)
            facade = build_surrogate_facade(
                _mock_model(),
                uncertainty_fallback=0.5,
            )
            outcome = maybe_retrain_after_iteration(
                config,
                iteration_index=1,
                state=state,
                surrogate=facade,
            )
            self.assertEqual(outcome.status, "skipped_insufficient_buffer_rows")

    def test_skips_stub_surrogate(self) -> None:
        config = replace(
            _MINI_CONFIG,
            retrain=RetrainConfig(
                enabled=True, every_iterations=1, min_new_buffer_rows=0
            ),
        )
        outcome = maybe_retrain_after_iteration(
            config,
            iteration_index=1,
            state=RetrainState(),
            surrogate=StubSurrogate(mean=0.5, uncertainty=0.85),
        )
        self.assertEqual(outcome.status, "skipped_not_facade")

    def test_reload_pickle_error_returns_reload_failed(self) -> None:
        config = replace(
            _MINI_CONFIG,
            retrain=RetrainConfig(
                enabled=True, every_iterations=1, min_new_buffer_rows=0
            ),
        )
        train_result = TrainResult(
            success=True,
            sample_count=100,
            holdout_metrics={"r2_fitness": 0.9},
            quality_passed=True,
            checkpoint_path=Path("x.pkl"),
            summary_path=Path("x.summary.json"),
        )
        facade = build_surrogate_facade(
            _mock_model(),
            uncertainty_fallback=0.5,
        )
        with mock.patch(
            "worldspace.surrogate.retrain.train_from_buffer",
            return_value=train_result,
        ):
            with mock.patch(
                "worldspace.surrogate.surrogate.load_surrogate_checkpoint",
                side_effect=pickle.UnpicklingError("corrupt checkpoint"),
            ):
                outcome = maybe_retrain_after_iteration(
                    config,
                    iteration_index=1,
                    state=RetrainState(),
                    surrogate=facade,
                )
        self.assertEqual(outcome.status, "reload_failed")

    def test_missing_model_dependency_returns_train_failed(self) -> None:
        config = replace(
            _MINI_CONFIG,
            retrain=RetrainConfig(
                enabled=True, every_iterations=1, min_new_buffer_rows=0
            ),
        )
        facade = build_surrogate_facade(
            _mock_model(),
            uncertainty_fallback=0.5,
        )
        with mock.patch(
            "worldspace.surrogate.training_runtime.find_spec",
            return_value=None,
        ):
            outcome = maybe_retrain_after_iteration(
                config,
                iteration_index=1,
                state=RetrainState(),
                surrogate=facade,
            )
        self.assertEqual(outcome.status, "train_failed")
        assert outcome.train_result is not None
        train_result = outcome.train_result
        self.assertFalse(train_result.success)
        self.assertIn("lightgbm", train_result.error_message or "")

    def test_train_from_buffer_missing_dependency_does_not_raise(self) -> None:
        with mock.patch(
            "worldspace.surrogate.training_runtime.find_spec",
            return_value=None,
        ):
            result = train_from_buffer(
                buffer_path=Path("unused.jsonl"),
                checkpoint_path=Path("unused.pkl"),
            )
        self.assertFalse(result.success)
        self.assertIn("lightgbm", result.error_message or "")


class TestSurrogateFacadeReload(unittest.TestCase):
    def test_reload_clears_cache(self) -> None:
        model_a = _mock_model(predict_value=0.1)
        model_b = _mock_model(predict_value=0.9)
        facade = build_surrogate_facade(
            model_a, uncertainty_fallback=0.5, cache_capacity=8
        )
        spec = replace(_FIXED_SPEC)
        first = facade.predict(spec)
        facade.predict(spec)
        self.assertEqual(facade.cache_hits(), 1)

        with mock.patch(
            "worldspace.surrogate.surrogate.load_surrogate_checkpoint",
            return_value=model_b,
        ):
            facade.reload(Path("ignored.pkl"))

        second = facade.predict(spec)
        self.assertNotAlmostEqual(second.fitness, first.fitness, places=5)
        self.assertEqual(facade.cache_hits(), 0)
        facade.predict(spec)
        self.assertEqual(facade.cache_hits(), 1)


class TestRunSchedulerRetrainHook(unittest.TestCase):
    def test_retrain_called_on_iteration_boundary_not_mid_batch(self) -> None:
        config = replace(
            _MINI_CONFIG,
            iterations=3,
            retrain=RetrainConfig(
                enabled=True,
                every_iterations=1,
                min_new_buffer_rows=0,
            ),
        )
        train_result = TrainResult(
            success=True,
            sample_count=100,
            holdout_metrics={"r2_fitness": 0.9},
            quality_passed=True,
            checkpoint_path=Path("x.pkl"),
            summary_path=Path("x.summary.json"),
        )
        with mock.patch(
            "worldspace.surrogate.retrain.train_from_buffer",
            return_value=train_result,
        ) as train_mock:
            with mock.patch.object(SurrogateFacade, "reload") as reload_mock:
                facade = build_surrogate_facade(
                    _mock_model(),
                    uncertainty_fallback=0.5,
                )
                state = RetrainState()
                run_scheduler(
                    config,
                    GridArchive(5),
                    np.random.default_rng(0),
                    StubCandidateEmitter(),
                    grid_size=8,
                    steps=200,
                    surrogate=facade,
                    retrain_state=state,
                )
        self.assertEqual(train_mock.call_count, 3)
        self.assertEqual(reload_mock.call_count, 3)

    def test_train_failure_keeps_run_going_without_reload(self) -> None:
        config = replace(
            _MINI_CONFIG,
            iterations=2,
            retrain=RetrainConfig(
                enabled=True,
                every_iterations=1,
                min_new_buffer_rows=0,
            ),
        )
        failed = TrainResult(
            success=False,
            sample_count=0,
            holdout_metrics={},
            quality_passed=False,
            checkpoint_path=Path("x.pkl"),
            summary_path=Path("x.summary.json"),
            error_message="forced failure",
        )
        with mock.patch(
            "worldspace.surrogate.retrain.train_from_buffer",
            return_value=failed,
        ):
            with mock.patch.object(SurrogateFacade, "reload") as reload_mock:
                facade = build_surrogate_facade(
                    _mock_model(),
                    uncertainty_fallback=0.5,
                )
                counters = run_scheduler(
                    config,
                    GridArchive(5),
                    np.random.default_rng(0),
                    StubCandidateEmitter(),
                    grid_size=8,
                    steps=200,
                    surrogate=facade,
                    retrain_state=RetrainState(),
                )
        reload_mock.assert_not_called()
        self.assertEqual(counters.candidates_evaluated, 2 * config.batch_size)


def _mock_model(*, predict_value: float = 0.1):
    model = mock.MagicMock()
    components = {
        "stability": predict_value,
        "diversity": predict_value,
        "oscillation_score": predict_value,
        "topology_interface_index": predict_value,
        "topology_window_heterogeneity": predict_value,
        "final_density": predict_value,
        "early_extinction_prob": predict_value,
    }
    model.predict_components.return_value = components
    model.predict_uncertainty.return_value = 0.2
    return model


if __name__ == "__main__":
    unittest.main()

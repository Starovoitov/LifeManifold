"""Integration tests for surrogate scheduler/loop wiring (E5)."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from worldspace.illuminators.archive import GridArchive
from worldspace.illuminators.emitters.stub import StubCandidateEmitter
from worldspace.illuminators.loop import run_iteration
from worldspace.illuminators.scheduler import (
    RunCounters,
    SchedulerConfig,
    resolve_surrogate_stub,
)
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec
from worldspace.surrogate import get_surrogate
from worldspace.surrogate.buffer import SurrogateBuffer
from worldspace.surrogate.model import SurrogateModel
from worldspace.surrogate.types import SurrogateConfig, SurrogatePrediction


def _scheduler_config(**overrides: object) -> SchedulerConfig:
    base = SchedulerConfig(
        schema_version="1.2",
        iterations=1,
        batch_size=2,
        grid_resolution=5,
        early_extinction_step=200,
        min_steps=200,
        batch_emitters=("random", "random"),
        initial_random_candidates=0,
        llm_enabled=False,
        surrogate_enabled=False,
        surrogate_model_type="lightgbm",
        surrogate_checkpoint="artifacts/surrogate/checkpoints/latest.pkl",
        surrogate_buffer_path="artifacts/surrogate/buffer.jsonl",
        surrogate_stub_mean=0.5,
        surrogate_stub_uncertainty=1.0,
        genetic_mutation_scale=0.02,
    )
    return replace(base, **overrides)


def _sample_spec() -> WorldSpec:
    return WorldSpec(
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


@dataclass(frozen=True)
class _FixedSurrogate:
    fitness: float
    uncertainty: float

    def predict(self, world_spec: WorldSpec) -> SurrogatePrediction:
        _ = world_spec
        return SurrogatePrediction(
            components={"stability": 0.1, "diversity": 0.2},
            measures={"stability": 0.1, "diversity": 0.2},
            fitness=self.fitness,
            uncertainty=self.uncertainty,
        )


class TestResolveSurrogateStub(unittest.TestCase):
    def test_disabled_returns_yaml_stub_without_predict(self) -> None:
        config = _scheduler_config(surrogate_enabled=False, surrogate_stub_mean=0.42)
        surrogate = _FixedSurrogate(fitness=0.99, uncertainty=0.11)
        values = resolve_surrogate_stub(config, surrogate, _sample_spec())
        self.assertEqual(values, (0.42, 1.0))

    def test_enabled_calls_surrogate_predict(self) -> None:
        config = _scheduler_config(surrogate_enabled=True)
        surrogate = _FixedSurrogate(fitness=0.77, uncertainty=0.33)
        values = resolve_surrogate_stub(config, surrogate, _sample_spec())
        self.assertEqual(values, (0.77, 0.33))


class TestGetSurrogateCheckpoint(unittest.TestCase):
    def test_loads_pickle_surrogate_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pkl"
            model = SurrogateModel()
            model.set_component_defaults(0.45)
            with checkpoint.open("wb") as fh:
                pickle.dump(model, fh)
            config = SurrogateConfig(
                enabled=True,
                model_type="lightgbm",
                checkpoint=str(checkpoint),
                stub_mean=0.45,
                stub_uncertainty=0.85,
            )
            surrogate = get_surrogate(config)
            prediction = surrogate.predict(_sample_spec())
            self.assertAlmostEqual(prediction.fitness, 0.45)


class TestLoopSurrogateBuffer(unittest.TestCase):
    def test_run_iteration_appends_one_row_per_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            config = _scheduler_config(batch_size=2)
            archive = GridArchive(5)
            rng = np.random.default_rng(0)
            buffer = SurrogateBuffer(path=buffer_path, flush_every=32)
            run_iteration(
                config,
                archive,
                rng,
                RunCounters(),
                StubCandidateEmitter(),
                iteration_index=1,
                grid_size=8,
                steps=200,
                surrogate_buffer=buffer,
            )
            buffer.flush()
            lines = buffer_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()

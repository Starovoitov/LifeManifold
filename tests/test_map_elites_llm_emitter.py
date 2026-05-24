"""Tests for MAP-Elites LLM emitter parse, fallback, and surrogate stub."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace

import numpy as np

from worldspace.illuminators.archive import (
    ArchiveElite,
    GridArchive,
    new_elite_metadata,
)
from worldspace.illuminators.emitters.llm_emitter import LlmEmitter
from worldspace.illuminators.emitters.llm_prompts import system_prompt_version
from worldspace.illuminators.scheduler import SchedulerConfig, TargetBin
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec
from worldspace.specs.world_spec_from_llm import (
    extract_json_object_from_text,
    world_spec_from_llm_payload,
)
from worldspace.surrogate.types import SurrogatePrediction

_TARGET = TargetBin(bin=(1, 1), target_stability=0.5, target_diversity=0.6)
_BASE = WorldSpec(
    birth=[1, 3],
    survival=[2, 3],
    noise=0.02,
    resource_regen=0.05,
    predation=0.1,
    cell_types=["life", "food"],
    grid_size=8,
    steps=200,
    seed=0,
)
_VALID_RESPONSE = json.dumps(
    {
        "reasoning": "Tune rules toward the target niche.",
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


def _scheduler_config(**overrides: object) -> SchedulerConfig:
    base = SchedulerConfig(
        schema_version="1.2",
        iterations=1,
        batch_size=4,
        grid_resolution=10,
        early_extinction_step=200,
        min_steps=200,
        batch_emitters=("random", "genetic", "genetic", "llm"),
        initial_random_candidates=0,
        llm_enabled=True,
        surrogate_enabled=False,
        surrogate_model_type="lightgbm",
        surrogate_checkpoint="artifacts/surrogate/checkpoints/latest.pkl",
        surrogate_buffer_path="artifacts/surrogate/buffer.jsonl",
        surrogate_stub_mean=0.5,
        surrogate_stub_uncertainty=1.0,
        genetic_mutation_scale=0.02,
    )
    return replace(base, **overrides)


class TestWorldSpecFromLlm(unittest.TestCase):
    def test_extract_json_from_fenced_block(self) -> None:
        text = f"```json\n{_VALID_RESPONSE}\n```"
        parsed = extract_json_object_from_text(text)
        self.assertIsNotNone(parsed)
        spec = world_spec_from_llm_payload(parsed, grid_size=8, steps=200, base=_BASE)
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.birth, [2])
        self.assertEqual(spec.cell_types, CANONICAL_CELL_TYPES)


class TestLlmEmitter(unittest.TestCase):
    def test_valid_json_emitter_type_llm(self) -> None:
        calls: list[str] = []

        def mock_llm(**kwargs: object) -> str:
            calls.append(str(kwargs.get("prompt", "")))
            return _VALID_RESPONSE

        emitter = LlmEmitter(
            grid_resolution=10,
            surrogate_mean=0.5,
            surrogate_uncertainty=1.0,
            call_llm_text=mock_llm,
        )
        output = emitter.emit(
            target=_TARGET,
            archive=GridArchive(5),
            rng=np.random.default_rng(0),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(len(calls), 1)
        self.assertIn("0.500", calls[0])
        self.assertEqual(output.metadata.emitter_type, "llm")
        self.assertEqual(output.metadata.prompt_version, system_prompt_version())
        self.assertEqual(output.world_spec.seed, 0)

    def test_invalid_json_uses_fallback(self) -> None:
        emitter = LlmEmitter(
            grid_resolution=10,
            surrogate_mean=0.5,
            surrogate_uncertainty=1.0,
            fallback_scale=0.02,
            call_llm_text=lambda **_: "not json at all",
        )
        output = emitter.emit(
            target=_TARGET,
            archive=GridArchive(5),
            rng=np.random.default_rng(1),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(output.metadata.emitter_type, "llm_fallback")
        self.assertEqual(output.world_spec.seed, 0)

    def test_fallback_differs_from_parent(self) -> None:
        archive = GridArchive(5)
        parent = replace(_BASE, birth=[1], noise=0.02)
        archive.try_insert(
            ArchiveElite(
                bin=(1, 1),
                fitness=0.5,
                world_spec=replace(parent, seed=1),
                measures={"stability": 0.5, "diversity": 0.5},
                metadata=new_elite_metadata(
                    generated_by="random",
                    emitter_type="random",
                    elite_id="p1",
                    timestamp="2026-01-01T00:00:00+00:00",
                ),
            )
        )
        emitter = LlmEmitter(
            grid_resolution=10,
            surrogate_mean=0.5,
            surrogate_uncertainty=1.0,
            call_llm_text=lambda **_: "{bad",
        )
        output = emitter.emit(
            target=_TARGET,
            archive=archive,
            rng=np.random.default_rng(2),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(output.metadata.emitter_type, "llm_fallback")
        self.assertNotEqual(output.world_spec.birth, parent.birth)


class TestLlmEmitterCalibratedSurrogate(unittest.TestCase):
    def test_emitter_uses_predicted_surrogate_values(self) -> None:
        captured: list[str] = []

        class _PredictingSurrogate:
            def predict(self, world_spec: WorldSpec) -> SurrogatePrediction:
                _ = world_spec
                return SurrogatePrediction(
                    components={},
                    measures={"stability": 0.31, "diversity": 0.29},
                    fitness=0.31,
                    uncertainty=0.07,
                )

        def mock_llm(**kwargs: object) -> str:
            captured.append(str(kwargs.get("prompt", "")))
            return "not json"

        config = _scheduler_config(surrogate_enabled=True)
        emitter = LlmEmitter(
            grid_resolution=config.grid_resolution,
            scheduler=config,
            surrogate=_PredictingSurrogate(),
            call_llm_text=mock_llm,
        )
        emitter.emit(
            target=_TARGET,
            archive=GridArchive(5),
            rng=np.random.default_rng(0),
            grid_size=8,
            steps=200,
        )
        self.assertIn("0.310", captured[0])
        self.assertIn("0.070", captured[0])
        self.assertNotIn("0.500", captured[0])


class TestSurrogateStub(unittest.TestCase):
    def test_emitter_uses_static_surrogate_when_scheduler_unset(self) -> None:
        captured: list[str] = []

        def mock_llm(**kwargs: object) -> str:
            captured.append(str(kwargs.get("prompt", "")))
            return "not json"

        emitter = LlmEmitter(
            grid_resolution=10,
            surrogate_mean=0.42,
            surrogate_uncertainty=0.88,
            call_llm_text=mock_llm,
        )
        emitter.emit(
            target=_TARGET,
            archive=GridArchive(5),
            rng=np.random.default_rng(0),
            grid_size=8,
            steps=200,
        )
        self.assertIn("0.420", captured[0])
        self.assertIn("0.880", captured[0])


if __name__ == "__main__":
    unittest.main()

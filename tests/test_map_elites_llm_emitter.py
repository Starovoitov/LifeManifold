"""Tests for MAP-Elites LLM emitter parse, fallback, and surrogate stub."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from typing import Sequence

import numpy as np

from worldspace.illuminators.archive import (
    ArchiveElite,
    GridArchive,
    new_elite_metadata,
)
from worldspace.illuminators.emitters.llm_emitter import LlmEmitter
from worldspace.illuminators.emitters.llm_prompts import (
    components_user_prompt_path,
    emitter_prompt_version,
    parent_user_prompt_path,
    user_prompt_version,
)
from worldspace.illuminators.scheduler import SchedulerConfig, TargetCell
from worldspace.metrics import WorldMetrics
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec
from worldspace.specs.world_spec_from_llm import (
    extract_json_object_from_text,
    world_spec_from_llm_payload,
)
from worldspace.surrogate.model import TARGET_KEYS
from worldspace.surrogate.types import SurrogatePrediction

_TARGET = TargetCell(
    cell_id=6,
    target_stability=0.5,
    target_diversity=0.6,
    bin_ij=(1, 1),
)
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
        self.assertEqual(output.metadata.prompt_version, emitter_prompt_version())
        self.assertEqual(output.world_spec.seed, 0)

    def test_prepare_finalize_matches_emit(self) -> None:
        rng = np.random.default_rng(3)
        archive = GridArchive(5)

        def mock_llm(**kwargs: object) -> str:
            return _VALID_RESPONSE

        emitter = LlmEmitter(
            grid_resolution=10,
            surrogate_mean=0.5,
            surrogate_uncertainty=1.0,
            call_llm_text=mock_llm,
        )
        prepared = emitter.prepare_emit(
            target=_TARGET,
            archive=archive,
            rng=rng,
            grid_size=8,
            steps=200,
        )
        via_parts = emitter.finalize_emit(
            prepared,
            response=_VALID_RESPONSE,
            rng=rng,
        )
        via_emit = emitter.emit(
            target=_TARGET,
            archive=archive,
            rng=np.random.default_rng(3),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(
            via_parts.metadata.emitter_type, via_emit.metadata.emitter_type
        )
        self.assertEqual(via_parts.world_spec.birth, via_emit.world_spec.birth)
        self.assertEqual(via_parts.world_spec.survival, via_emit.world_spec.survival)

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

    def test_llm_config_value_error_uses_fallback(self) -> None:
        def raise_value_error(**_: object) -> str:
            raise ValueError("Unknown provider in llm config: 'missing'")

        emitter = LlmEmitter(
            grid_resolution=10,
            surrogate_mean=0.5,
            surrogate_uncertainty=1.0,
            call_llm_text=raise_value_error,
        )
        output = emitter.emit(
            target=_TARGET,
            archive=GridArchive(5),
            rng=np.random.default_rng(3),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(output.metadata.emitter_type, "llm_fallback")

    def test_llm_runtime_error_uses_fallback(self) -> None:
        def raise_runtime_error(**_: object) -> str:
            raise RuntimeError("LLM request failed: SSL handshake timeout")

        emitter = LlmEmitter(
            grid_resolution=10,
            surrogate_mean=0.5,
            surrogate_uncertainty=1.0,
            call_llm_text=raise_runtime_error,
        )
        output = emitter.emit(
            target=_TARGET,
            archive=GridArchive(5),
            rng=np.random.default_rng(4),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(output.metadata.emitter_type, "llm_fallback")

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
                    components={key: 0.31 for key in TARGET_KEYS},
                    measures={"stability": 0.31, "diversity": 0.29},
                    fitness=0.31,
                    uncertainty=0.07,
                )

            def predict_batch(
                self, world_specs: Sequence[WorldSpec]
            ) -> list[SurrogatePrediction]:
                return [self.predict(world_spec) for world_spec in world_specs]

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


class TestLlmEmitterComponentsPrompt(unittest.TestCase):
    def test_prepare_emit_uses_components_user_prompt(self) -> None:
        class _RichSurrogate:
            def predict(self, world_spec: WorldSpec) -> SurrogatePrediction:
                _ = world_spec
                return SurrogatePrediction(
                    components={
                        "stability": 0.41,
                        "diversity": 0.55,
                        "oscillation_score": 0.12,
                        "topology_interface_index": 0.33,
                        "topology_window_heterogeneity": 0.28,
                        "final_density": 0.62,
                        "early_extinction_prob": 0.38,
                    },
                    measures={"stability": 0.41, "diversity": 0.55},
                    fitness=0.487,
                    uncertainty=0.71,
                )

            def predict_batch(
                self, world_specs: Sequence[WorldSpec]
            ) -> list[SurrogatePrediction]:
                return [self.predict(world_spec) for world_spec in world_specs]

        config = _scheduler_config(
            surrogate_enabled=True,
            llm_user_prompt_path="prompts/map_elites_llm_emitter_user_components.txt",
        )
        emitter = LlmEmitter(
            grid_resolution=config.grid_resolution,
            scheduler=config,
            surrogate=_RichSurrogate(),
        )
        prepared = emitter.prepare_emit(
            target=_TARGET,
            archive=GridArchive(5),
            rng=np.random.default_rng(0),
            grid_size=8,
            steps=200,
        )
        self.assertIn("early_extinction_prob", prepared.user_prompt)
        self.assertIn("0.380", prepared.user_prompt)
        self.assertIn("0.710", prepared.user_prompt)
        self.assertIsNotNone(prepared.surrogate_prediction)
        assert prepared.surrogate_prediction is not None
        self.assertAlmostEqual(
            prepared.surrogate_prediction.components["early_extinction_prob"],
            0.38,
        )

    def test_components_prompt_version_differs_from_default(self) -> None:
        class _RichSurrogate:
            def predict(self, world_spec: WorldSpec) -> SurrogatePrediction:
                _ = world_spec
                return SurrogatePrediction(
                    components={key: 0.5 for key in TARGET_KEYS},
                    measures={"stability": 0.5, "diversity": 0.5},
                    fitness=0.5,
                    uncertainty=1.0,
                )

            def predict_batch(
                self, world_specs: Sequence[WorldSpec]
            ) -> list[SurrogatePrediction]:
                return [self.predict(world_spec) for world_spec in world_specs]

        config = _scheduler_config(
            surrogate_enabled=True,
            llm_user_prompt_path="prompts/map_elites_llm_emitter_user_components.txt",
        )
        emitter = LlmEmitter(
            grid_resolution=config.grid_resolution,
            scheduler=config,
            surrogate=_RichSurrogate(),
        )
        prepared = emitter.prepare_emit(
            target=_TARGET,
            archive=GridArchive(5),
            rng=np.random.default_rng(0),
            grid_size=8,
            steps=200,
        )
        self.assertNotEqual(
            prepared.prompt_version.split(":")[-1],
            user_prompt_version(),
        )
        self.assertEqual(
            prepared.prompt_version.split(":")[-1],
            user_prompt_version(components_user_prompt_path()),
        )


class TestLlmEmitterParentPrompt(unittest.TestCase):
    def test_prepare_emit_uses_parent_metrics_hint_block(self) -> None:
        class _RichSurrogate:
            def predict(self, world_spec: WorldSpec) -> SurrogatePrediction:
                _ = world_spec
                return SurrogatePrediction(
                    components={
                        "stability": 0.41,
                        "diversity": 0.55,
                        "oscillation_score": 0.12,
                        "topology_interface_index": 0.33,
                        "topology_window_heterogeneity": 0.28,
                        "final_density": 0.62,
                        "early_extinction_prob": 0.38,
                    },
                    measures={"stability": 0.41, "diversity": 0.55},
                    fitness=0.487,
                    uncertainty=0.71,
                )

            def predict_batch(
                self, world_specs: Sequence[WorldSpec]
            ) -> list[SurrogatePrediction]:
                return [self.predict(world_spec) for world_spec in world_specs]

        archive = GridArchive(10)
        parent_bin = archive.bin_from_cell_id(_TARGET.cell_id)
        archive.try_insert(
            ArchiveElite(
                bin=parent_bin,
                fitness=0.782,
                world_spec=replace(_BASE, seed=1),
                measures={"stability": 0.642, "diversity": 0.875},
                metrics=WorldMetrics(
                    entropy=0.5,
                    stability=0.642,
                    average_lifespan=0.4,
                    density_mean=0.62,
                    oscillation_score=0.12,
                    diversity=0.875,
                    mo_eoc_indicator=0.1,
                    topology_interface_index=0.33,
                    topology_window_heterogeneity=0.28,
                    compressibility_score=0.2,
                    ecology_state_entropy_norm=0.3,
                    ecology_resource_adjacency=0.4,
                ),
                metadata=new_elite_metadata(
                    generated_by="random",
                    emitter_type="random",
                    elite_id="parent-cell",
                    timestamp="2026-01-01T00:00:00+00:00",
                ),
            )
        )
        config = _scheduler_config(
            surrogate_enabled=True,
            llm_user_prompt_path="prompts/map_elites_llm_emitter_user_parent_hints.txt",
        )
        emitter = LlmEmitter(
            grid_resolution=config.grid_resolution,
            scheduler=config,
            surrogate=_RichSurrogate(),
        )
        prepared = emitter.prepare_emit(
            target=_TARGET,
            archive=archive,
            rng=np.random.default_rng(0),
            grid_size=8,
            steps=200,
        )
        self.assertIn("Parent cell (observed from simulation)", prepared.user_prompt)
        self.assertIn("fitness: 0.782", prepared.user_prompt)
        self.assertIn("composed fitness: 0.487", prepared.user_prompt)
        self.assertIn("0.710", prepared.user_prompt)

    def test_parent_prompt_version_differs_from_default(self) -> None:
        class _RichSurrogate:
            def predict(self, world_spec: WorldSpec) -> SurrogatePrediction:
                _ = world_spec
                return SurrogatePrediction(
                    components={key: 0.5 for key in TARGET_KEYS},
                    measures={"stability": 0.5, "diversity": 0.5},
                    fitness=0.5,
                    uncertainty=1.0,
                )

            def predict_batch(
                self, world_specs: Sequence[WorldSpec]
            ) -> list[SurrogatePrediction]:
                return [self.predict(world_spec) for world_spec in world_specs]

        config = _scheduler_config(
            surrogate_enabled=True,
            llm_user_prompt_path="prompts/map_elites_llm_emitter_user_parent_hints.txt",
        )
        emitter = LlmEmitter(
            grid_resolution=config.grid_resolution,
            scheduler=config,
            surrogate=_RichSurrogate(),
        )
        prepared = emitter.prepare_emit(
            target=_TARGET,
            archive=GridArchive(10),
            rng=np.random.default_rng(0),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(
            prepared.prompt_version.split(":")[-1],
            user_prompt_version(parent_user_prompt_path()),
        )


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

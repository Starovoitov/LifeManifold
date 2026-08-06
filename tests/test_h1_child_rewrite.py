"""Unit tests for H1 child-rewrite trigger and emit path."""

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
from worldspace.illuminators.emitters.llm_emitter import (
    LlmEmitter,
    should_rewrite_child,
)
from worldspace.illuminators.scheduler import (
    ChildRewriteConfig,
    SchedulerConfig,
    TargetCell,
)
from worldspace.specs.spec import WorldSpec
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
_REWRITE_RESPONSE = json.dumps(
    {
        "world_spec": {
            "birth": [2, 3],
            "survival": [2, 3],
            "noise": 0.05,
            "resource_regen": 0.07,
            "predation": 0.11,
            "cell_types": ["life", "food"],
            "neighborhood": "moore",
        },
    }
)


class _LowChildSurrogate:
    def predict(self, world_spec: WorldSpec) -> SurrogatePrediction:
        _ = world_spec
        components = {key: 0.30 for key in TARGET_KEYS}
        return SurrogatePrediction(
            components=components,
            measures={"stability": 0.30, "diversity": 0.30},
            fitness=0.30,
            uncertainty=0.16,
        )


def _scheduler(**overrides: object) -> SchedulerConfig:
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
        surrogate_enabled=True,
        surrogate_model_type="mlp",
        surrogate_checkpoint="artifacts/surrogate/checkpoints/nightly_v3_mc_d005.pkl",
        surrogate_buffer_path="artifacts/surrogate/buffer.jsonl",
        surrogate_stub_mean=0.5,
        surrogate_stub_uncertainty=1.0,
        genetic_mutation_scale=0.02,
        llm_child_rewrite=ChildRewriteConfig(
            enabled=True,
            trigger="below_parent_true",
            keep_draft_on_rewrite_fail=True,
            user_prompt_path="prompts/map_elites_llm_emitter_user_rewrite.txt",
        ),
    )
    return replace(base, **overrides)


class TestShouldRewriteChild(unittest.TestCase):
    def test_below_parent_true(self) -> None:
        cfg = ChildRewriteConfig(enabled=True, trigger="below_parent_true")
        self.assertTrue(
            should_rewrite_child(cfg, child_pred_fitness=0.3, parent_true_fitness=0.5)
        )
        self.assertFalse(
            should_rewrite_child(cfg, child_pred_fitness=0.6, parent_true_fitness=0.5)
        )
        self.assertTrue(
            should_rewrite_child(
                cfg, child_pred_fitness=0.4, parent_true_fitness=float("nan")
            )
        )

    def test_disabled(self) -> None:
        cfg = ChildRewriteConfig(enabled=False, trigger="always")
        self.assertFalse(
            should_rewrite_child(cfg, child_pred_fitness=0.1, parent_true_fitness=0.9)
        )


class TestChildRewriteEmit(unittest.TestCase):
    def test_rewrite_fires_and_marks_emitter_type(self) -> None:
        prompts: list[str] = []

        def mock_llm(**kwargs: object) -> str:
            prompts.append(str(kwargs.get("prompt", "")))
            if len(prompts) == 1:
                return _VALID_RESPONSE
            return _REWRITE_RESPONSE

        archive = GridArchive(resolution=10)
        archive.try_insert(
            ArchiveElite(
                bin=(0, 6),
                fitness=0.55,
                world_spec=replace(_BASE, seed=1),
                measures={"stability": 0.5, "diversity": 0.6},
                metadata=new_elite_metadata(
                    generated_by="seed",
                    emitter_type="random",
                    elite_id="parent0",
                ),
            )
        )

        emitter = LlmEmitter(
            scheduler=_scheduler(),
            surrogate=_LowChildSurrogate(),
            call_llm_text=mock_llm,
        )
        target = TargetCell(
            cell_id=6,
            target_stability=0.5,
            target_diversity=0.6,
            bin_ij=(0, 6),
        )
        output = emitter.emit(
            target=target,
            archive=archive,
            rng=np.random.default_rng(0),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(output.metadata.emitter_type, "llm_rewrite")
        self.assertEqual(len(prompts), 2)
        self.assertIn("Draft WorldSpec", prompts[1])
        self.assertEqual(output.world_spec.birth, [2, 3])


if __name__ == "__main__":
    unittest.main()

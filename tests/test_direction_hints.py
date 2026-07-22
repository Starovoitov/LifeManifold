"""Tests for surrogate direction-of-improvement hint helpers."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from worldspace.illuminators.evaluation import apply_canonical_seed
from worldspace.illuminators.emitters.llm_emitter import build_user_prompt
from worldspace.illuminators.emitters.llm_prompts import (
    DIRECTION_USER_PROMPT_PATH,
    DIRECTION_USER_PROMPT_FIELD_NAMES,
    direction_user_prompt_path,
    load_user_prompt_template,
    render_user_prompt,
    resolve_direction_prompt_fields,
    surrogate_prompt_fields,
    user_prompt_version,
)
from worldspace.illuminators.archive import GridArchive
from worldspace.illuminators.scheduler import TargetCell
from worldspace.specs.spec import WorldSpec
from worldspace.surrogate.direction_hints import (
    DIRECTION_HINT_EMPTY,
    compute_composed_fitness_gradient,
    direction_prompt_fields,
    format_direction_hint_block,
    suggestion_for_feature,
)
from worldspace.surrogate.feature_extractor import extract
from worldspace.surrogate.types import SurrogatePrediction

_BASE_SPEC = WorldSpec(
    birth=[1, 3],
    survival=[2],
    noise=0.05,
    resource_regen=0.1,
    predation=0.0,
    cell_types=["life", "food"],
    grid_size=8,
    steps=200,
    seed=0,
)

_TARGET = TargetCell(
    cell_id=6,
    target_stability=0.42,
    target_diversity=0.55,
    bin_ij=(1, 1),
)


def _canonical_spec() -> WorldSpec:
    spec = replace(_BASE_SPEC, seed=0)
    apply_canonical_seed(spec)
    return spec


_COMPONENT_PREDICTION = SurrogatePrediction(
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


class _LinearFitnessModel:
    """Fitness = sum(features); used to validate FD gradients."""

    def predict_components(self, features: np.ndarray) -> dict[str, float]:
        _ = features
        return {
            "stability": 0.5,
            "diversity": 0.5,
            "oscillation_score": 0.1,
            "topology_interface_index": 0.2,
            "topology_window_heterogeneity": 0.2,
            "final_density": 0.5,
            "early_extinction_prob": 0.1,
        }

    def predict_fitness(self, features: np.ndarray) -> float:
        vector = np.asarray(features, dtype=float).reshape(-1)
        return float(np.sum(vector))


class TestDirectionHints(unittest.TestCase):
    def test_linear_model_gradient_positive_on_enabled_birth(self) -> None:
        model = _LinearFitnessModel()
        features = extract(_canonical_spec())
        gradient, names = compute_composed_fitness_gradient(model, features)
        birth_1_index = names.index("birth_1")
        birth_3_index = names.index("birth_3")
        self.assertGreater(gradient[birth_1_index], 0.0)
        self.assertGreater(gradient[birth_3_index], 0.0)

    def test_suggestion_for_birth_rule(self) -> None:
        text = suggestion_for_feature("birth_3", 0.08)
        self.assertIn("birth rule index 3", text)
        self.assertIn("enabling", text.lower())

    def test_format_direction_hint_block_non_flat(self) -> None:
        gradient = np.zeros(21, dtype=float)
        gradient[0] = 0.12
        block = format_direction_hint_block(
            gradient, tuple(f"birth_{i}" for i in range(9))
        )
        self.assertIn("Surrogate local sensitivity", block)
        self.assertIn("birth rule index 0", block)

    def test_direction_prompt_fields(self) -> None:
        fields = direction_prompt_fields(_canonical_spec(), _LinearFitnessModel())
        self.assertIn("direction_hint_block", fields)
        self.assertIn("Surrogate local sensitivity", fields["direction_hint_block"])

    def test_resolve_direction_prompt_fields_stub_model(self) -> None:
        template = load_user_prompt_template(direction_user_prompt_path())
        fields = resolve_direction_prompt_fields(
            template,
            parent_world_spec=_BASE_SPEC,
            surrogate_model=None,
        )
        self.assertEqual(fields["direction_hint_block"], DIRECTION_HINT_EMPTY)

    def test_resolve_direction_prompt_fields_skips_default_template(self) -> None:
        template = load_user_prompt_template()
        self.assertEqual(resolve_direction_prompt_fields(template), {})

    def test_direction_template_renders(self) -> None:
        template = load_user_prompt_template(direction_user_prompt_path())
        for field_name in DIRECTION_USER_PROMPT_FIELD_NAMES:
            self.assertIn(f"{{{field_name}}}", template)
        fields = direction_prompt_fields(_canonical_spec(), _LinearFitnessModel())
        rendered = render_user_prompt(
            template,
            target_stability=_TARGET.target_stability,
            target_diversity=_TARGET.target_diversity,
            **surrogate_prompt_fields(_COMPONENT_PREDICTION),
            **fields,
            current_elite_json="null",
            few_shot_examples="(none)",
            constraints="{}",
        )
        self.assertNotIn("{direction_hint_block}", rendered)
        self.assertIn("Surrogate local sensitivity", rendered)
        self.assertIn("0.487", rendered)

    def test_build_user_prompt_includes_direction_block(self) -> None:
        template = load_user_prompt_template(direction_user_prompt_path())
        prompt = build_user_prompt(
            target=_TARGET,
            archive=GridArchive(10),
            rng=np.random.default_rng(0),
            prediction=_COMPONENT_PREDICTION,
            user_prompt_template=template,
            direction_parent_spec=_canonical_spec(),
            direction_surrogate_model=_LinearFitnessModel(),
        )
        self.assertIn("Surrogate local sensitivity", prompt)

    def test_direction_prompt_version_differs_from_default(self) -> None:
        direction_hash = user_prompt_version(direction_user_prompt_path())
        self.assertNotEqual(direction_hash, user_prompt_version())
        self.assertEqual(direction_user_prompt_path(), DIRECTION_USER_PROMPT_PATH)


if __name__ == "__main__":
    unittest.main()

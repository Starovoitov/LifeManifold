"""Tests for MAP-Elites LLM system and user prompts."""

from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from worldspace.illuminators.archive import (
    ArchiveElite,
    GridArchive,
    new_elite_metadata,
)
from worldspace.illuminators.emitters.llm_emitter import (
    build_user_prompt,
    format_few_shot_block,
    moore_neighbor_elites,
)
from worldspace.illuminators.emitters.llm_prompts import (
    DEFAULT_SYSTEM_PROMPT_PATH,
    load_system_prompt_template,
    render_system_prompt,
    system_prompt_version,
)
from worldspace.illuminators.scheduler import TargetBin
from worldspace.specs.spec import WorldSpec
from worldspace.specs.world_param_bounds import NOISE_MAX, NOISE_MIN
from worldspace.specs.world_spec_constraints import (
    WORLD_SPEC_CONSTRAINTS,
    format_world_spec_constraints,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_BASE_SPEC = WorldSpec(
    birth=[1],
    survival=[2, 3],
    noise=0.02,
    resource_regen=0.05,
    predation=0.1,
    cell_types=["life", "food"],
    grid_size=8,
    steps=200,
    seed=0,
)


def _elite(
    bin_coord: tuple[int, int],
    fitness: float,
    *,
    elite_id: str,
) -> ArchiveElite:
    return ArchiveElite(
        bin=bin_coord,
        fitness=fitness,
        world_spec=replace(_BASE_SPEC, seed=1),
        measures={"stability": 0.5, "diversity": 0.6},
        metadata=new_elite_metadata(
            generated_by="random",
            emitter_type="random",
            elite_id=elite_id,
            timestamp="2026-01-01T00:00:00+00:00",
        ),
    )


class TestSystemPrompt(unittest.TestCase):
    def test_render_substitutes_n_and_bin_width(self) -> None:
        rendered = render_system_prompt(50)
        self.assertIn("50×50", rendered)
        self.assertIn("0.02", rendered)

    def test_system_prompt_version_is_sha256_prefix(self) -> None:
        expected = hashlib.sha256(DEFAULT_SYSTEM_PROMPT_PATH.read_bytes()).hexdigest()[
            :8
        ]
        self.assertEqual(system_prompt_version(), expected)

    def test_system_prompt_contains_fitness_formula(self) -> None:
        text = load_system_prompt_template()
        self.assertIn("stability", text)
        self.assertIn("diversity", text)
        self.assertIn("0.45*diversity", text)


class TestUserPrompt(unittest.TestCase):
    def test_user_prompt_template_keeps_surrogate_placeholders(self) -> None:
        from worldspace.illuminators.emitters.llm_prompts import USER_PROMPT_TEMPLATE

        self.assertIn("{surrogate_mean:", USER_PROMPT_TEMPLATE)
        self.assertIn("{surrogate_uncertainty:", USER_PROMPT_TEMPLATE)

    def test_build_user_prompt_includes_targets_and_surrogate(self) -> None:
        archive = GridArchive(5)
        target = TargetBin(bin=(2, 2), target_stability=0.42, target_diversity=0.57)
        prompt = build_user_prompt(
            target=target,
            archive=archive,
            surrogate_mean=0.5,
            surrogate_uncertainty=1.0,
            rng=np.random.default_rng(0),
        )
        self.assertIn("0.42", prompt)
        self.assertIn("0.57", prompt)
        self.assertIn("0.500", prompt)
        self.assertIn("1.000", prompt)

    def test_moore_neighbors_only_occupied(self) -> None:
        archive = GridArchive(5)
        archive.try_insert(_elite((2, 2), 0.5, elite_id="center"))
        archive.try_insert(_elite((1, 1), 0.3, elite_id="n1"))
        archive.try_insert(_elite((3, 3), 0.4, elite_id="n2"))
        neighbors = moore_neighbor_elites(
            archive, (2, 2), rng=np.random.default_rng(0), max_count=4
        )
        bins = {elite.bin for elite in neighbors}
        self.assertNotIn((2, 2), bins)
        self.assertTrue(
            bins.issubset(
                {(1, 1), (1, 2), (2, 1), (2, 3), (3, 2), (3, 3), (1, 3), (3, 1)}
            )
        )

    def test_few_shot_empty_neighbors(self) -> None:
        self.assertIn("no occupied", format_few_shot_block([]))

    def test_few_shot_deterministic_subset(self) -> None:
        archive = GridArchive(6)
        center = (3, 3)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                archive.try_insert(
                    _elite((center[0] + di, center[1] + dj), 0.1, elite_id=f"{di}-{dj}")
                )
        a = moore_neighbor_elites(
            archive, center, rng=np.random.default_rng(7), max_count=3
        )
        b = moore_neighbor_elites(
            archive, center, rng=np.random.default_rng(7), max_count=3
        )
        self.assertEqual([e.bin for e in a], [e.bin for e in b])
        self.assertEqual(len(a), 3)

    def test_fixture_format_roundtrip(self) -> None:
        raw = json.loads(
            (_FIXTURES / "llm_few_shot_examples.json").read_text(encoding="utf-8")
        )
        text = json.dumps(raw, ensure_ascii=True, indent=2)
        self.assertIn("reasoning", text)
        self.assertIn("fitness", text)

    def test_constraints_match_world_param_bounds(self) -> None:
        text = format_world_spec_constraints()
        self.assertIn(str(NOISE_MIN), text)
        self.assertIn(str(NOISE_MAX), text)
        self.assertEqual(WORLD_SPEC_CONSTRAINTS["noise"], f"[{NOISE_MIN},{NOISE_MAX}]")


if __name__ == "__main__":
    unittest.main()

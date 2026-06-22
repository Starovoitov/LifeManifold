"""Unit tests for dashboard LLM prompt tester helpers."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from worldspace.illuminators.archive import (
    ArchiveElite,
    GridArchive,
    new_elite_metadata,
)
from worldspace.illuminators.emitters.llm_emitter import build_user_prompt
from worldspace.illuminators.scheduler import TargetBin, TargetCell
from worldspace.specs.spec import WorldSpec

_REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

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


def _elite(bin_coord: tuple[int, int], fitness: float) -> ArchiveElite:
    return ArchiveElite(
        bin=bin_coord,
        fitness=fitness,
        world_spec=replace(_BASE_SPEC, seed=1),
        measures={"stability": 0.4, "diversity": 0.6},
        metadata=new_elite_metadata(
            generated_by="random",
            emitter_type="random",
            elite_id="e1",
            timestamp="2026-01-01T00:00:00+00:00",
        ),
    )


class TestDashboardLlmPromptTester(unittest.TestCase):
    def test_load_user_prompt_from_repo(self) -> None:
        from dashboard.components.llm_prompt_tester import (
            load_user_prompt_from_config,
            resolve_prompts_dir,
        )

        text = load_user_prompt_from_config()
        self.assertIn("{surrogate_mean", text)
        self.assertTrue(
            (resolve_prompts_dir() / "map_elites_llm_emitter_user.txt").is_file()
        )

    def test_list_format_placeholders(self) -> None:
        from dashboard.components.llm_prompt_tester import list_format_placeholders

        names = list_format_placeholders(
            "a {surrogate_mean:.3f} b {target_stability:.2f} c {surrogate_mean:.3f}"
        )
        self.assertEqual(names, ["surrogate_mean", "target_stability"])

    def test_render_user_prompt_changes_with_surrogate(self) -> None:
        from dashboard.components.llm_prompt_tester import (
            render_user_prompt_preview,
            user_prompt_format_kwargs,
        )
        from dashboard.components.llm_prompt_tester import (
            load_user_prompt_from_config,
        )

        archive = GridArchive(5)
        archive.try_insert(_elite((2, 2), 0.5))
        target = TargetBin(bin=(2, 2), target_stability=0.42, target_diversity=0.57)
        rng = np.random.default_rng(0)
        template = load_user_prompt_from_config()
        low, low_err = render_user_prompt_preview(
            template,
            user_prompt_format_kwargs(archive, target, 0.1, 0.9, rng=rng),
        )
        high, high_err = render_user_prompt_preview(
            template,
            user_prompt_format_kwargs(archive, target, 0.9, 0.1, rng=rng),
        )
        self.assertIsNone(low_err)
        self.assertIsNone(high_err)
        self.assertIn("0.100", low)
        self.assertIn("0.900", high)
        self.assertNotEqual(low, high)

    def test_render_user_prompt_preview_unknown_placeholder(self) -> None:
        from dashboard.components.llm_prompt_tester import (
            render_user_prompt_preview,
            user_prompt_format_kwargs,
        )

        archive = GridArchive(3)
        archive.try_insert(_elite((1, 1), 0.5))
        target = TargetBin(bin=(1, 1), target_stability=0.5, target_diversity=0.5)
        kwargs = user_prompt_format_kwargs(
            archive, target, 0.5, 0.5, rng=np.random.default_rng(0)
        )
        text, err = render_user_prompt_preview("Value: {unknown_field}", kwargs)
        self.assertEqual(text, "")
        self.assertIsNotNone(err)
        self.assertIn("unknown_field", err or "")

    def test_reused_rng_breaks_few_shot_parity_with_fresh_rng(self) -> None:
        from dashboard.components.llm_prompt_tester import (
            build_user_prompt_like_emitter,
            load_user_prompt_from_config,
            render_user_prompt_preview,
            user_prompt_format_kwargs,
        )

        archive = GridArchive(5)
        center = (2, 2)
        for ni in range(5):
            for nj in range(5):
                if (ni, nj) != center:
                    archive.try_insert(_elite((ni, nj), float(ni + nj)))
        target = TargetBin(bin=center, target_stability=0.5, target_diversity=0.5)
        template = load_user_prompt_from_config()
        seed = 42

        shared = np.random.default_rng(seed)
        kwargs_shared = user_prompt_format_kwargs(archive, target, 0.5, 0.5, rng=shared)
        reference_after_shared = build_user_prompt_like_emitter(
            archive, target, 0.5, 0.5, rng=shared
        )
        rendered_shared, _ = render_user_prompt_preview(template, kwargs_shared)

        kwargs_fresh = user_prompt_format_kwargs(
            archive, target, 0.5, 0.5, rng=np.random.default_rng(seed)
        )
        reference_fresh = build_user_prompt_like_emitter(
            archive, target, 0.5, 0.5, rng=np.random.default_rng(seed)
        )
        rendered_fresh, _ = render_user_prompt_preview(template, kwargs_fresh)

        self.assertEqual(rendered_fresh, reference_fresh)
        self.assertNotEqual(rendered_shared, reference_after_shared)

    def test_build_user_prompt_like_emitter_matches_worldspace(self) -> None:
        from dashboard.components.llm_prompt_tester import (
            build_user_prompt_like_emitter,
        )

        archive = GridArchive(5)
        target = TargetCell(
            cell_id=6,
            target_stability=0.3,
            target_diversity=0.7,
            bin_ij=(1, 1),
        )
        target_bin = TargetBin.from_target_cell(target)
        rng = np.random.default_rng(3)
        expected = build_user_prompt(
            target=target,
            archive=archive,
            surrogate_mean=0.55,
            surrogate_uncertainty=0.2,
            rng=rng,
        )
        actual = build_user_prompt_like_emitter(
            archive,
            target_bin,
            0.55,
            0.2,
            rng=rng,
        )
        self.assertEqual(actual, expected)

    def test_render_system_prompt_preview_cvt_requires_n_centroids(self) -> None:
        from dashboard.components.llm_prompt_tester import render_system_prompt_preview

        with self.assertRaises(ValueError):
            render_system_prompt_preview(10, archive_type="cvt")

    def test_render_system_prompt_preview_cvt_uses_explicit_n_centroids(self) -> None:
        from dashboard.components.llm_prompt_tester import render_system_prompt_preview

        text = render_system_prompt_preview(
            10,
            archive_type="cvt",
            n_centroids=25,
        )
        self.assertIn("25", text)
        self.assertIn("Voronoi", text)
        self.assertNotIn("10×10", text)

    def test_render_system_prompt_preview_cvt_reads_defaults_from_config(self) -> None:
        from dashboard.components.llm_prompt_tester import render_system_prompt_preview

        text = render_system_prompt_preview(
            10,
            {"defaults": {"n_centroids": 9}},
            archive_type="cvt",
        )
        self.assertIn("9", text)


if __name__ == "__main__":
    unittest.main()

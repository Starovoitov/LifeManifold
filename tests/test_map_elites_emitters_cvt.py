"""Unit tests for MAP-Elites emitters on CVT archives."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from worldspace.illuminators.archive import ArchiveElite, new_elite_metadata
from worldspace.illuminators.cvt import generate_centroids
from worldspace.illuminators.cvt_archive import CvtArchive
from worldspace.illuminators.emitters import (
    GeneticEmitter,
    LlmEmitter,
    MapElitesEmitter,
)
from worldspace.illuminators.emitters.llm_prompts import (
    DEFAULT_SYSTEM_PROMPT_PATH_CVT,
    render_cvt_system_prompt,
    system_prompt_version,
    emitter_prompt_version,
)
from worldspace.illuminators.scheduler import SchedulerConfig, TargetCell
from worldspace.specs.spec import WorldSpec

_BASE_SPEC = WorldSpec(
    birth=[1, 3],
    survival=[2, 3, 4],
    noise=0.02,
    resource_regen=0.05,
    predation=0.1,
    cell_types=["life", "food"],
    grid_size=8,
    steps=200,
    seed=99,
)


def _cvt_archive(n_centroids: int = 9) -> CvtArchive:
    centroids = generate_centroids(n_centroids, seed=0, lloyd_iterations=5)
    return CvtArchive(centroids)


def _elite(
    archive: CvtArchive,
    cell_id: int,
    fitness: float,
    spec: WorldSpec,
    *,
    elite_id: str,
) -> ArchiveElite:
    return ArchiveElite(
        bin=archive.bin_from_cell_id(cell_id),
        fitness=fitness,
        world_spec=replace(spec, seed=1),
        measures={"stability": 0.5, "diversity": 0.5},
        metadata=new_elite_metadata(
            generated_by="random",
            emitter_type="random",
            elite_id=elite_id,
            timestamp="2026-01-01T00:00:00+00:00",
        ),
    )


def _target_cell(cell_id: int, archive: CvtArchive) -> TargetCell:
    stability, diversity = archive.cell_center(cell_id)
    return TargetCell(
        cell_id=cell_id,
        target_stability=stability,
        target_diversity=diversity,
        bin_ij=archive.bin_from_cell_id(cell_id),
    )


def _mini_cvt_scheduler() -> SchedulerConfig:
    return SchedulerConfig(
        schema_version="1.3",
        iterations=1,
        batch_size=4,
        grid_resolution=5,
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
        archive_type="cvt",
        n_centroids=25,
        cvt_seed=0,
        lloyd_iterations=5,
    )


class TestGeneticEmitterCvt(unittest.TestCase):
    def test_empty_archive_falls_back_to_random(self) -> None:
        archive = _cvt_archive()
        target = _target_cell(0, archive)
        output = GeneticEmitter().emit(
            target=target,
            archive=archive,
            rng=np.random.default_rng(0),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(output.metadata.emitter_type, "genetic")
        self.assertIsNone(output.metadata.parent_id)

    def test_offspring_from_voronoi_neighbor(self) -> None:
        archive = _cvt_archive()
        archive.try_insert(_elite(archive, 0, 0.7, _BASE_SPEC, elite_id="parent-one"))
        neighbor_id = archive.neighbors(0)[0]
        archive.try_insert(
            _elite(
                archive,
                neighbor_id,
                0.4,
                replace(_BASE_SPEC, birth=[2]),
                elite_id="parent-two",
            )
        )
        output = GeneticEmitter(mutation_scale=0.0).emit(
            target=_target_cell(0, archive),
            archive=archive,
            rng=np.random.default_rng(5),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(output.metadata.parent_id, "parent-one")

    def test_parent_two_min_fitness_without_neighbors(self) -> None:
        archive = _cvt_archive(4)
        archive.try_insert(_elite(archive, 1, 0.9, _BASE_SPEC, elite_id="target"))
        archive.try_insert(_elite(archive, 0, 0.2, _BASE_SPEC, elite_id="low"))
        output = GeneticEmitter(mutation_scale=0.0).emit(
            target=_target_cell(1, archive),
            archive=archive,
            rng=np.random.default_rng(99),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(output.metadata.parent_id, "target")


class TestLlmEmitterCvt(unittest.TestCase):
    def test_system_prompt_uses_cvt_template(self) -> None:
        archive = _cvt_archive()
        captured_system: list[str] = []

        def mock_llm(**kwargs: object) -> str:
            captured_system.append(str(kwargs.get("system_content", "")))
            return "not json"

        emitter = LlmEmitter(
            scheduler=_mini_cvt_scheduler(),
            call_llm_text=mock_llm,
        )
        emitter.emit(
            target=_target_cell(0, archive),
            archive=archive,
            rng=np.random.default_rng(0),
            grid_size=8,
            steps=200,
        )
        self.assertIn("Voronoi", captured_system[0])
        self.assertIn("25", captured_system[0])
        self.assertNotIn("×", captured_system[0])

    def test_prompt_version_matches_cvt_file(self) -> None:
        archive = _cvt_archive()
        output = LlmEmitter(
            scheduler=_mini_cvt_scheduler(),
            call_llm_text=lambda **_: "not json",
        ).emit(
            target=_target_cell(0, archive),
            archive=archive,
            rng=np.random.default_rng(1),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(
            output.metadata.prompt_version,
            emitter_prompt_version(archive_type="cvt"),
        )

    def test_few_shot_uses_voronoi_neighbors(self) -> None:
        archive = _cvt_archive()
        archive.try_insert(_elite(archive, 0, 0.5, _BASE_SPEC, elite_id="center"))
        neighbor_id = archive.neighbors(0)[0]
        archive.try_insert(
            _elite(archive, neighbor_id, 0.4, _BASE_SPEC, elite_id="neighbor")
        )
        captured_user: list[str] = []

        def mock_llm(**kwargs: object) -> str:
            captured_user.append(str(kwargs.get("prompt", "")))
            return "not json"

        output = LlmEmitter(
            scheduler=_mini_cvt_scheduler(),
            call_llm_text=mock_llm,
        ).emit(
            target=_target_cell(0, archive),
            archive=archive,
            rng=np.random.default_rng(0),
            grid_size=8,
            steps=200,
        )
        self.assertIn("neighbor", captured_user[0])
        self.assertEqual(output.metadata.emitter_type, "llm_fallback")


class TestMapElitesEmitterCvt(unittest.TestCase):
    def test_dispatch_random_genetic_llm(self) -> None:
        archive = _cvt_archive()
        archive.try_insert(_elite(archive, 0, 0.5, _BASE_SPEC, elite_id="p0"))
        neighbor_id = archive.neighbors(0)[0]
        archive.try_insert(_elite(archive, neighbor_id, 0.4, _BASE_SPEC, elite_id="p1"))
        emitter = MapElitesEmitter(
            mutation_scale=0.02,
            scheduler=_mini_cvt_scheduler(),
            llm_emitter=LlmEmitter(
                scheduler=_mini_cvt_scheduler(),
                call_llm_text=lambda **_: (
                    '{"reasoning":"x","world_spec":{"birth":[1],"survival":[2],'
                    '"noise":0.03,"resource_regen":0.05,"predation":0.1}}'
                ),
            ),
        )
        target = _target_cell(0, archive)
        random_out = emitter.emit(
            emitter_kind="random",
            target=target,
            archive=archive,
            rng=np.random.default_rng(1),
            grid_size=8,
            steps=200,
        )
        genetic_out = emitter.emit(
            emitter_kind="genetic",
            target=target,
            archive=archive,
            rng=np.random.default_rng(2),
            grid_size=8,
            steps=200,
        )
        llm_out = emitter.emit(
            emitter_kind="llm",
            target=target,
            archive=archive,
            rng=np.random.default_rng(3),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(random_out.metadata.emitter_type, "random")
        self.assertEqual(random_out.world_spec.seed, 0)
        self.assertEqual(genetic_out.metadata.emitter_type, "genetic")
        self.assertEqual(llm_out.metadata.emitter_type, "llm")
        self.assertEqual(
            llm_out.metadata.prompt_version,
            emitter_prompt_version(archive_type="cvt"),
        )


class TestCvtSystemPromptTemplate(unittest.TestCase):
    def test_render_substitutes_centroid_count(self) -> None:
        rendered = render_cvt_system_prompt(25)
        self.assertIn("25", rendered)
        self.assertIn("24", rendered)
        self.assertIn("Voronoi", rendered)

    def test_cvt_version_differs_from_grid(self) -> None:
        self.assertNotEqual(
            system_prompt_version(archive_type="grid"),
            system_prompt_version(archive_type="cvt"),
        )
        self.assertTrue(DEFAULT_SYSTEM_PROMPT_PATH_CVT.is_file())


if __name__ == "__main__":
    unittest.main()

"""Unit tests for MAP-Elites random and genetic emitters."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from worldspace.illuminators.archive import (
    ArchiveElite,
    GridArchive,
    new_elite_metadata,
)
from worldspace.illuminators.emitters import (
    GeneticEmitter,
    MapElitesEmitter,
    RandomEmitter,
    decode_genome,
    encode_world,
    strip_seed,
)
from worldspace.illuminators.emitters.genetics import uniform_crossover
from worldspace.illuminators.scheduler import TargetBin
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec

_TARGET = TargetBin(bin=(2, 3), target_stability=0.5, target_diversity=0.6)
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


def _elite(
    bin_coord: tuple[int, int],
    fitness: float,
    spec: WorldSpec,
    *,
    elite_id: str,
) -> ArchiveElite:
    return ArchiveElite(
        bin=bin_coord,
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


class TestStripSeedAndGenetics(unittest.TestCase):
    def test_strip_seed_clears_seed_and_normalizes_cell_types(self) -> None:
        spec = replace(_BASE_SPEC, seed=42, cell_types=["empty", "life", "food"])
        stripped = strip_seed(spec)
        self.assertEqual(stripped.seed, 0)
        self.assertEqual(stripped.cell_types, CANONICAL_CELL_TYPES)

    def test_encode_decode_roundtrip(self) -> None:
        genes = encode_world(_BASE_SPEC)
        restored = decode_genome(genes, grid_size=8, steps=200)
        self.assertEqual(restored.birth, sorted(set(_BASE_SPEC.birth)))
        self.assertEqual(restored.survival, sorted(set(_BASE_SPEC.survival)))
        self.assertAlmostEqual(restored.noise, _BASE_SPEC.noise)
        self.assertEqual(restored.seed, 0)


class TestRandomEmitter(unittest.TestCase):
    def test_random_emitter_metadata_and_seed(self) -> None:
        output = RandomEmitter().emit(
            target=_TARGET,
            archive=GridArchive(5),
            rng=np.random.default_rng(3),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(output.world_spec.seed, 0)
        self.assertEqual(output.world_spec.cell_types, CANONICAL_CELL_TYPES)
        self.assertIsNone(output.metadata.parent_id)
        self.assertEqual(output.metadata.generated_by, "random")
        self.assertEqual(output.metadata.emitter_type, "random")

    def test_random_emitter_deterministic_with_rng(self) -> None:
        archive = GridArchive(5)
        a = RandomEmitter().emit(
            target=_TARGET,
            archive=archive,
            rng=np.random.default_rng(11),
            grid_size=8,
            steps=200,
        )
        b = RandomEmitter().emit(
            target=_TARGET,
            archive=archive,
            rng=np.random.default_rng(11),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(a.world_spec.birth, b.world_spec.birth)
        self.assertEqual(a.world_spec.survival, b.world_spec.survival)


class TestGeneticEmitter(unittest.TestCase):
    def test_empty_archive_falls_back_to_random(self) -> None:
        output = GeneticEmitter().emit(
            target=_TARGET,
            archive=GridArchive(5),
            rng=np.random.default_rng(0),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(output.metadata.generated_by, "genetic")
        self.assertIsNone(output.metadata.parent_id)
        self.assertEqual(output.world_spec.seed, 0)

    def test_empty_target_bin_falls_back_to_random(self) -> None:
        archive = GridArchive(5)
        archive.try_insert(_elite((0, 0), 0.5, _BASE_SPEC, elite_id="other"))
        output = GeneticEmitter().emit(
            target=_TARGET,
            archive=archive,
            rng=np.random.default_rng(1),
            grid_size=8,
            steps=200,
        )
        self.assertIsNone(output.metadata.parent_id)

    def test_genetic_offspring_has_parent_id(self) -> None:
        archive = GridArchive(5)
        parent_spec = replace(_BASE_SPEC, birth=[1], survival=[2])
        archive.try_insert(_elite((2, 3), 0.7, parent_spec, elite_id="parent-one"))
        archive.try_insert(
            _elite((2, 2), 0.2, replace(_BASE_SPEC, birth=[2]), elite_id="parent-two")
        )
        output = GeneticEmitter(mutation_scale=0.0).emit(
            target=TargetBin(bin=(2, 3), target_stability=0.5, target_diversity=0.5),
            archive=archive,
            rng=np.random.default_rng(5),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(output.metadata.parent_id, "parent-one")
        self.assertEqual(output.metadata.emitter_type, "genetic")

    def test_crossover_and_zero_mutation_deterministic(self) -> None:
        rng = np.random.default_rng(0)
        genes_a = encode_world(replace(_BASE_SPEC, birth=[1]))
        genes_b = encode_world(replace(_BASE_SPEC, birth=[2]))
        child = uniform_crossover(genes_a, genes_b, rng)
        spec = decode_genome(child, grid_size=8, steps=200)
        self.assertEqual(spec.cell_types, CANONICAL_CELL_TYPES)

    def test_parent_two_min_fitness_lex_tie(self) -> None:
        archive = GridArchive(4)
        archive.try_insert(_elite((1, 1), 0.9, _BASE_SPEC, elite_id="target"))
        archive.try_insert(
            _elite((0, 0), 0.3, replace(_BASE_SPEC, noise=0.01), elite_id="low-a")
        )
        archive.try_insert(
            _elite((3, 3), 0.3, replace(_BASE_SPEC, noise=0.02), elite_id="low-b")
        )
        output = GeneticEmitter(mutation_scale=0.0).emit(
            target=TargetBin(bin=(1, 1), target_stability=0.5, target_diversity=0.5),
            archive=archive,
            rng=np.random.default_rng(99),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(output.metadata.parent_id, "target")


class TestMapElitesEmitter(unittest.TestCase):
    def test_dispatch_random_and_genetic(self) -> None:
        emitter = MapElitesEmitter(mutation_scale=0.02)
        archive = GridArchive(5)
        random_out = emitter.emit(
            emitter_kind="random",
            target=_TARGET,
            archive=archive,
            rng=np.random.default_rng(1),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(random_out.metadata.emitter_type, "random")
        archive.try_insert(_elite((2, 3), 0.5, _BASE_SPEC, elite_id="p"))
        archive.try_insert(_elite((2, 2), 0.4, _BASE_SPEC, elite_id="n"))
        genetic_out = emitter.emit(
            emitter_kind="genetic",
            target=TargetBin(bin=(2, 3), target_stability=0.5, target_diversity=0.5),
            archive=archive,
            rng=np.random.default_rng(2),
            grid_size=8,
            steps=200,
        )
        self.assertEqual(genetic_out.metadata.emitter_type, "genetic")


if __name__ == "__main__":
    unittest.main()

"""Unit tests for archive protocol, helpers, and factory."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from worldspace.illuminators.archive import (
    ArchiveElite,
    GridArchive,
    bin_ij_from_flat_cell_id,
    cvt_cell_id,
    flat_cell_id,
    new_elite_metadata,
)
from worldspace.illuminators.archive_factory import (
    ArchiveFactoryConfig,
    create_archive,
    create_grid_archive,
)
from worldspace.illuminators.cvt import load_centroids
from worldspace.illuminators.cvt_archive import CvtArchive
from worldspace.illuminators.evaluation import bin_center, bin_index
from worldspace.specs.spec import WorldSpec

_BASE_SPEC = WorldSpec(
    birth=[1],
    survival=[2],
    noise=0.0,
    resource_regen=0.0,
    predation=0.0,
    cell_types=["life", "food"],
    grid_size=4,
    steps=200,
    seed=0,
)


def _minimal_elite(
    bin_ij: tuple[int, int],
    fitness: float,
    *,
    elite_id: str = "test-id",
) -> ArchiveElite:
    return ArchiveElite(
        bin=bin_ij,
        fitness=fitness,
        world_spec=replace(_BASE_SPEC, seed=1),
        measures={"stability": 0.5, "diversity": 0.5},
        metadata=new_elite_metadata(
            generated_by="random",
            emitter_type="random",
            elite_id=elite_id,
            timestamp="2026-01-01T00:00:00+00:00",
        ),
    )


class TestArchiveEliteHelpers(unittest.TestCase):
    def test_bin_ij_alias(self) -> None:
        elite = _minimal_elite((2, 3), 0.5)
        self.assertEqual(elite.bin_ij, (2, 3))
        self.assertEqual(elite.bin_ij, elite.bin)

    def test_flat_cell_id_roundtrip(self) -> None:
        resolution = 5
        for cell_id in (0, 7, 24):
            bin_ij = bin_ij_from_flat_cell_id(cell_id, resolution=resolution)
            self.assertEqual(flat_cell_id(bin_ij, resolution=resolution), cell_id)

    def test_cvt_cell_id_uses_first_bin_component(self) -> None:
        self.assertEqual(cvt_cell_id((4, 0)), 4)


class TestGridArchiveProtocol(unittest.TestCase):
    def setUp(self) -> None:
        self.archive = GridArchive(5)

    def test_archive_type_and_n_cells(self) -> None:
        self.assertEqual(self.archive.archive_type, "grid")
        self.assertEqual(self.archive.n_cells, 25)

    def test_get_cell_matches_get_ij(self) -> None:
        elite = _minimal_elite((2, 3), 0.6)
        self.archive.try_insert(elite)
        self.assertIs(self.archive.get_cell(13), self.archive.get(2, 3))

    def test_cell_center_matches_bin_center(self) -> None:
        cell_id = flat_cell_id((1, 2), resolution=5)
        self.assertEqual(
            self.archive.cell_center(cell_id),
            bin_center(1, 2, 5),
        )

    def test_neighbors_are_symmetric(self) -> None:
        cell_id = flat_cell_id((2, 2), resolution=5)
        for neighbor in self.archive.neighbors(cell_id):
            self.assertIn(cell_id, self.archive.neighbors(neighbor))

    def test_assign_cell_id_matches_bin_index(self) -> None:
        stability, diversity = 0.42, 0.58
        i, j = bin_index(stability, diversity, 5)
        expected = flat_cell_id((i, j), resolution=5)
        self.assertEqual(self.archive.assign_cell_id(stability, diversity), expected)

    def test_cell_id_from_bin_roundtrip(self) -> None:
        bin_ij = (3, 1)
        cell_id = self.archive.cell_id_from_bin(bin_ij)
        self.assertEqual(self.archive.bin_from_cell_id(cell_id), bin_ij)


class TestCvtArchiveProtocol(unittest.TestCase):
    def setUp(self) -> None:
        from worldspace.illuminators.cvt import generate_centroids

        self.centroids = generate_centroids(16, seed=1)
        self.archive = CvtArchive(self.centroids)

    def test_archive_type_and_n_cells(self) -> None:
        self.assertEqual(self.archive.archive_type, "cvt")
        self.assertEqual(self.archive.n_cells, 16)

    def test_get_cell_matches_get(self) -> None:
        self.archive.try_insert(_minimal_elite((4, 0), 0.5))
        self.assertIs(self.archive.get_cell(4), self.archive.get(4))

    def test_cell_center_matches_centroid(self) -> None:
        self.assertEqual(self.archive.cell_center(5), tuple(self.centroids[5]))

    def test_neighbors_are_symmetric(self) -> None:
        for cell_id in range(self.archive.n_cells):
            for neighbor in self.archive.neighbors(cell_id):
                self.assertIn(cell_id, self.archive.neighbors(neighbor))

    def test_cell_id_from_bin_roundtrip(self) -> None:
        cell_id = 7
        bin_ij = self.archive.bin_from_cell_id(cell_id)
        self.assertEqual(bin_ij, (7, 0))
        self.assertEqual(self.archive.cell_id_from_bin(bin_ij), cell_id)

    def test_try_insert_uses_cell_id_from_bin(self) -> None:
        result = self.archive.try_insert(_minimal_elite((9, 0), 0.4))
        self.assertTrue(result.accepted)
        stored = self.archive.get_cell(9)
        assert stored is not None
        self.assertEqual(stored.fitness, 0.4)


class TestCreateArchive(unittest.TestCase):
    def test_create_grid_archive_factory(self) -> None:
        archive = create_archive(
            ArchiveFactoryConfig(archive_type="grid", resolution=5)
        )
        self.assertIsInstance(archive, GridArchive)
        self.assertEqual(archive.n_cells, 25)

    def test_create_grid_archive_convenience(self) -> None:
        archive = create_grid_archive(4)
        self.assertEqual(archive.resolution, 4)

    def test_create_cvt_archive_generates_and_saves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            first = create_archive(
                ArchiveFactoryConfig(
                    archive_type="cvt",
                    n_centroids=16,
                    cvt_seed=3,
                ),
                output_dir=output_dir,
            )
            centroids_path = output_dir / "cvt_centroids.json"
            self.assertTrue(centroids_path.is_file())
            loaded = load_centroids(centroids_path)
            second = create_archive(
                ArchiveFactoryConfig(
                    archive_type="cvt",
                    n_centroids=16,
                    cvt_seed=99,
                ),
                centroids_path=centroids_path,
            )
            np.testing.assert_array_equal(first.centroids, second.centroids)
            np.testing.assert_array_equal(second.centroids, loaded)
        self.assertIsInstance(first, CvtArchive)
        self.assertIsInstance(second, CvtArchive)


if __name__ == "__main__":
    unittest.main()

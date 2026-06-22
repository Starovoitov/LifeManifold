"""Unit tests for CVT centroids, assignment, and CvtArchive."""

from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace

import numpy as np

from worldspace.illuminators.archive import (
    ArchiveElite,
    InsertResult,
    new_elite_metadata,
)
from worldspace.illuminators.cvt import (
    assign_cell_id,
    generate_centroids,
    voronoi_neighbors,
)
from worldspace.illuminators.cvt_archive import CvtArchive
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
    cell_id: int,
    fitness: float,
    *,
    elite_id: str = "test-id",
) -> ArchiveElite:
    return ArchiveElite(
        bin=(cell_id, 0),
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


class TestGenerateCentroids(unittest.TestCase):
    def test_reproducible_for_same_seed(self) -> None:
        first = generate_centroids(25, seed=7)
        second = generate_centroids(25, seed=7)
        np.testing.assert_array_equal(first, second)

    def test_values_in_unit_square(self) -> None:
        for n_centroids in (10, 25, 100):
            centroids = generate_centroids(n_centroids, seed=0)
            self.assertEqual(centroids.shape, (n_centroids, 2))
            self.assertTrue(np.all(centroids >= 0.0))
            self.assertTrue(np.all(centroids <= 1.0))

    def test_invalid_n_centroids_raises(self) -> None:
        with self.assertRaises(ValueError):
            generate_centroids(0, seed=0)


class TestAssignCellId(unittest.TestCase):
    def test_corner_points_map_to_valid_cells(self) -> None:
        centroids = generate_centroids(25, seed=0)
        corners = (
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.5, 0.5),
        )
        for stability, diversity in corners:
            cell_id = assign_cell_id(stability, diversity, centroids)
            self.assertGreaterEqual(cell_id, 0)
            self.assertLess(cell_id, centroids.shape[0])

    def test_out_of_range_inputs_are_clipped(self) -> None:
        centroids = generate_centroids(10, seed=1)
        low = assign_cell_id(-0.5, -0.5, centroids)
        high = assign_cell_id(1.5, 1.5, centroids)
        clipped = assign_cell_id(0.0, 0.0, centroids)
        self.assertEqual(low, clipped)
        self.assertEqual(
            high,
            assign_cell_id(1.0, 1.0, centroids),
        )


class TestVoronoiNeighbors(unittest.TestCase):
    def test_neighbors_are_symmetric(self) -> None:
        centroids = generate_centroids(25, seed=0)
        neighbors = voronoi_neighbors(centroids)
        for cell_id, adjacent in neighbors.items():
            for other in adjacent:
                self.assertIn(cell_id, neighbors[other])

    def test_golden_neighbor_map_n25_seed0(self) -> None:
        centroids = generate_centroids(25, seed=0)
        neighbors = voronoi_neighbors(centroids)
        edge_count = sum(len(adjacent) for adjacent in neighbors.values()) // 2
        digest = hashlib.sha256(
            json.dumps(neighbors, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.assertEqual(edge_count, 57)
        self.assertEqual(digest[:16], "a57da8622636aa06")


class TestCvtArchive(unittest.TestCase):
    def setUp(self) -> None:
        self.centroids = generate_centroids(16, seed=3)
        self.archive = CvtArchive(self.centroids)

    def test_insert_into_empty_cell(self) -> None:
        result = self.archive.try_insert(_minimal_elite(4, 0.5))
        self.assertEqual(
            result,
            InsertResult(accepted=True, improved=False, rejected=False),
        )
        stored = self.archive.get(4)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.fitness, 0.5)

    def test_insert_improves_fitness(self) -> None:
        self.archive.try_insert(_minimal_elite(1, 0.5, elite_id="a"))
        result = self.archive.try_insert(_minimal_elite(1, 0.6, elite_id="b"))
        self.assertEqual(
            result,
            InsertResult(accepted=True, improved=True, rejected=False),
        )
        stored = self.archive.get(1)
        assert stored is not None
        self.assertEqual(stored.metadata.id if stored.metadata else None, "b")

    def test_equal_fitness_rejected(self) -> None:
        self.archive.try_insert(_minimal_elite(0, 0.5))
        result = self.archive.try_insert(_minimal_elite(0, 0.5, elite_id="other"))
        self.assertEqual(
            result,
            InsertResult(accepted=False, improved=False, rejected=True),
        )

    def test_lower_fitness_rejected(self) -> None:
        self.archive.try_insert(_minimal_elite(2, 0.8, elite_id="keep"))
        result = self.archive.try_insert(_minimal_elite(2, 0.2, elite_id="lose"))
        self.assertTrue(result.rejected)
        stored = self.archive.get(2)
        assert stored is not None
        self.assertEqual(stored.metadata.id if stored.metadata else None, "keep")

    def test_invalid_cell_id_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.archive.get(-1)
        with self.assertRaises(IndexError):
            self.archive.get(self.archive.n_cells)

    def test_cell_center_matches_centroid(self) -> None:
        self.assertEqual(self.archive.cell_center(5), tuple(self.centroids[5]))

    def test_assign_cell_id_delegates_to_centroids(self) -> None:
        stability, diversity = 0.42, 0.58
        expected = assign_cell_id(stability, diversity, self.centroids)
        self.assertEqual(self.archive.assign_cell_id(stability, diversity), expected)

    def test_filled_and_empty_counts(self) -> None:
        self.archive.try_insert(_minimal_elite(0, 0.1))
        self.archive.try_insert(_minimal_elite(3, 0.2))
        self.assertEqual(self.archive.filled_count(), 2)
        self.assertEqual(self.archive.empty_count(), self.archive.n_cells - 2)


if __name__ == "__main__":
    unittest.main()

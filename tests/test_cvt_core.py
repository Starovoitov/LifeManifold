"""Unit tests for CVT centroids, assignment, and CvtArchive."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from worldspace.illuminators.archive import (
    ArchiveElite,
    InsertResult,
    new_elite_metadata,
)
from worldspace.illuminators.cvt import (
    assign_cell_id,
    centroids_path_for_output,
    generate_centroids,
    load_centroids,
    save_centroids,
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

    def test_different_seeds_produce_different_centroids(self) -> None:
        first = generate_centroids(25, seed=0)
        second = generate_centroids(25, seed=1)
        self.assertFalse(np.array_equal(first, second))

    def test_invalid_lloyd_iterations_raises(self) -> None:
        with self.assertRaises(ValueError):
            generate_centroids(10, seed=0, lloyd_iterations=0)


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

    def test_every_grid_point_maps_to_valid_owner(self) -> None:
        centroids = generate_centroids(25, seed=2)
        n_centroids = centroids.shape[0]
        axis = np.linspace(0.0, 1.0, 50, dtype=np.float64)
        for stability in axis:
            for diversity in axis:
                cell_id = assign_cell_id(float(stability), float(diversity), centroids)
                self.assertGreaterEqual(cell_id, 0)
                self.assertLess(cell_id, n_centroids)


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

    def test_invalid_centroid_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            voronoi_neighbors(np.zeros(5, dtype=np.float64))
        with self.assertRaises(ValueError):
            voronoi_neighbors(np.zeros((5, 3), dtype=np.float64))


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

    def test_neighbors_match_voronoi_helper(self) -> None:
        expected = voronoi_neighbors(self.centroids)
        for cell_id in range(self.archive.n_cells):
            self.assertEqual(self.archive.neighbors(cell_id), expected[cell_id])

    def test_invalid_centroids_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            CvtArchive(np.zeros(5, dtype=np.float64))


class TestCentroidsIo(unittest.TestCase):
    def test_centroids_path_for_output(self) -> None:
        path = centroids_path_for_output("/tmp/run")
        self.assertEqual(path.name, "cvt_centroids.json")
        self.assertEqual(path.parent, Path("/tmp/run"))

    def test_save_load_roundtrip_bit_identical(self) -> None:
        centroids = generate_centroids(25, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cvt_centroids.json"
            save_centroids(path, centroids)
            loaded = load_centroids(path)
        np.testing.assert_array_equal(loaded, centroids)

    def test_load_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_centroids("/nonexistent/cvt_centroids.json")

    def test_load_invalid_n_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps({"n": 10, "centroids": [[0.0, 0.0]]}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_centroids(path)

    def test_load_invalid_row_shape_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps({"n": 1, "centroids": [[0.0, 0.0, 0.0]]}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_centroids(path)

    def test_cvt_archive_from_loaded_centroids(self) -> None:
        centroids = generate_centroids(16, seed=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cvt_centroids.json"
            save_centroids(path, centroids)
            archive = CvtArchive(load_centroids(path))
        result = archive.try_insert(_minimal_elite(2, 0.55))
        self.assertTrue(result.accepted)
        stored = archive.get(2)
        assert stored is not None
        self.assertEqual(stored.fitness, 0.55)


class TestResumeAssignStability(unittest.TestCase):
    def test_assign_stable_after_save_load(self) -> None:
        centroids = generate_centroids(25, seed=0)
        probes = ((0.3, 0.7), (0.9, 0.1), (0.5, 0.5))
        before = {
            probe: assign_cell_id(probe[0], probe[1], centroids) for probe in probes
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cvt_centroids.json"
            save_centroids(path, centroids)
            loaded = load_centroids(path)
        after = {probe: assign_cell_id(probe[0], probe[1], loaded) for probe in probes}
        self.assertEqual(before, after)

    def test_neighbors_stable_after_save_load(self) -> None:
        centroids = generate_centroids(25, seed=0)
        before = voronoi_neighbors(centroids)
        digest_before = hashlib.sha256(
            json.dumps(before, sort_keys=True).encode("utf-8")
        ).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cvt_centroids.json"
            save_centroids(path, centroids)
            loaded = load_centroids(path)
        after = voronoi_neighbors(loaded)
        digest_after = hashlib.sha256(
            json.dumps(after, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.assertEqual(digest_before, digest_after)


if __name__ == "__main__":
    unittest.main()

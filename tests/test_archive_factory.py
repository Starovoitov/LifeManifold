"""Unit tests for archive factory and archive-aware assignment helper."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from worldspace.illuminators.archive import GridArchive
from worldspace.illuminators.archive_factory import (
    ArchiveFactoryConfig,
    create_archive,
    create_empty_archive,
)
from worldspace.illuminators.cvt import generate_centroids, load_centroids
from worldspace.illuminators.cvt_archive import CvtArchive
from worldspace.illuminators.evaluation import (
    assign_cell_for_archive,
    bin_index,
    evaluate_candidate,
)
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec

_BASE_SPEC = WorldSpec(
    birth=[1],
    survival=[2, 3],
    noise=0.02,
    resource_regen=0.05,
    predation=0.1,
    cell_types=CANONICAL_CELL_TYPES.copy(),
    grid_size=4,
    steps=200,
    seed=0,
)


class TestArchiveFactory(unittest.TestCase):
    def test_create_archive_grid(self) -> None:
        archive = create_archive(
            ArchiveFactoryConfig(archive_type="grid", resolution=5)
        )
        self.assertIsInstance(archive, GridArchive)
        self.assertEqual(archive.n_cells, 25)

    def test_create_archive_cvt_generates_and_reuses_saved_centroids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            first = create_archive(
                ArchiveFactoryConfig(archive_type="cvt", n_centroids=16, cvt_seed=3),
                output_dir=output_dir,
            )
            centroids_path = output_dir / "cvt_centroids.json"
            self.assertTrue(centroids_path.is_file())
            saved = load_centroids(centroids_path)
            second = create_archive(
                ArchiveFactoryConfig(archive_type="cvt", n_centroids=16, cvt_seed=99),
                centroids_path=centroids_path,
            )
            np.testing.assert_array_equal(first.centroids, second.centroids)
            np.testing.assert_array_equal(second.centroids, saved)
        self.assertIsInstance(first, CvtArchive)
        self.assertIsInstance(second, CvtArchive)

    def test_create_empty_archive_cvt_requires_existing_centroids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            centroids_path = Path(tmp) / "cvt_centroids.json"
            config = ArchiveFactoryConfig(archive_type="cvt", n_centroids=4)
            with self.assertRaises(ValueError):
                create_empty_archive(config, centroids_path=None)
            with self.assertRaises(FileNotFoundError):
                create_empty_archive(config, centroids_path=centroids_path)

            create_archive(config, output_dir=Path(tmp))
            empty = create_empty_archive(config, centroids_path=centroids_path)
        self.assertIsInstance(empty, CvtArchive)
        self.assertEqual(empty.n_cells, 4)


class TestAssignCellForArchive(unittest.TestCase):
    def test_assign_cell_for_archive_matches_grid_bin_index(self) -> None:
        archive = GridArchive(10)
        measures = {"stability": 0.33, "diversity": 0.77}
        i, j = bin_index(
            measures["stability"], measures["diversity"], archive.resolution
        )
        expected = archive.cell_id_from_bin((i, j))
        self.assertEqual(assign_cell_for_archive(measures, archive), expected)

    def test_assign_cell_for_archive_uses_cvt_nearest_centroid(self) -> None:
        centroids = np.array([[0.1, 0.1], [0.8, 0.8], [0.25, 0.75]], dtype=np.float64)
        archive = CvtArchive(centroids)
        measures = {"stability": 0.26, "diversity": 0.74}
        self.assertEqual(assign_cell_for_archive(measures, archive), 2)


class TestEvaluateCandidateForArchive(unittest.TestCase):
    def test_evaluate_candidate_uses_grid_archive_bin(self) -> None:
        archive = GridArchive(10)
        result = evaluate_candidate(_BASE_SPEC, archive=archive)
        i, j = bin_index(
            result.measures["stability"],
            result.measures["diversity"],
            archive.resolution,
        )
        self.assertEqual(result.bin, (i, j))

    def test_evaluate_candidate_uses_cvt_archive_bin(self) -> None:
        archive = CvtArchive(generate_centroids(9, seed=0, lloyd_iterations=5))
        result = evaluate_candidate(_BASE_SPEC, archive=archive)
        cell_id = assign_cell_for_archive(result.measures, archive)
        self.assertEqual(result.bin, (cell_id, 0))


if __name__ == "__main__":
    unittest.main()

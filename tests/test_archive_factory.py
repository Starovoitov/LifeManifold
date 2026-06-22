"""Unit tests for archive factory and archive-aware assignment helper."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from worldspace.illuminators.archive import GridArchive
from worldspace.illuminators.archive_factory import ArchiveFactoryConfig, create_archive
from worldspace.illuminators.cvt import load_centroids
from worldspace.illuminators.cvt_archive import CvtArchive
from worldspace.illuminators.evaluation import assign_cell_for_archive, bin_index


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


if __name__ == "__main__":
    unittest.main()

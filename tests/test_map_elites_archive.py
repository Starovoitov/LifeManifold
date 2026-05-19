"""Unit tests for MAP-Elites grid archive."""

from __future__ import annotations

import unittest

from worldspace.illuminators.archive import (
    DEFAULT_GRID_RESOLUTION,
    ArchiveElite,
    GridArchive,
    InsertResult,
)


class TestGridArchiveStructure(unittest.TestCase):
    def test_grid_archive_size_50(self) -> None:
        archive = GridArchive(DEFAULT_GRID_RESOLUTION)
        self.assertEqual(archive.resolution, 50)
        self.assertEqual(archive.filled_count(), 0)
        self.assertEqual(archive.empty_count(), 2500)

    def test_bc_range_fixed(self) -> None:
        archive = GridArchive(10)
        self.assertEqual(archive.bc_min, 0.0)
        self.assertEqual(archive.bc_max, 1.0)

    def test_invalid_bin_raises(self) -> None:
        archive = GridArchive(5)
        with self.assertRaises(IndexError):
            archive.get(-1, 0)
        with self.assertRaises(IndexError):
            archive.is_empty(5, 0)

    def test_resolution_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            GridArchive(0)


class TestGridArchiveInsert(unittest.TestCase):
    def test_insert_empty_accepts_zero_fitness(self) -> None:
        archive = GridArchive(5)
        elite = ArchiveElite(bin=(2, 3), fitness=0.0)
        result = archive.try_insert(elite)
        self.assertEqual(result, InsertResult(accepted=True, improved=False, rejected=False))
        stored = archive.get(2, 3)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.fitness, 0.0)

    def test_insert_improves_strict(self) -> None:
        archive = GridArchive(5)
        archive.try_insert(ArchiveElite(bin=(1, 1), fitness=0.5))
        improved = archive.try_insert(ArchiveElite(bin=(1, 1), fitness=0.6))
        self.assertEqual(improved, InsertResult(accepted=True, improved=True, rejected=False))
        stored = archive.get(1, 1)
        assert stored is not None
        self.assertEqual(stored.fitness, 0.6)

    def test_insert_equal_fitness_rejected(self) -> None:
        archive = GridArchive(5)
        archive.try_insert(ArchiveElite(bin=(0, 0), fitness=0.5))
        result = archive.try_insert(ArchiveElite(bin=(0, 0), fitness=0.5))
        self.assertEqual(result, InsertResult(accepted=False, improved=False, rejected=True))

    def test_insert_lower_fitness_rejected(self) -> None:
        archive = GridArchive(5)
        archive.try_insert(ArchiveElite(bin=(4, 4), fitness=0.8))
        result = archive.try_insert(ArchiveElite(bin=(4, 4), fitness=0.2))
        self.assertTrue(result.rejected)
        stored = archive.get(4, 4)
        assert stored is not None
        self.assertEqual(stored.fitness, 0.8)

    def test_insert_reject_does_not_mutate(self) -> None:
        archive = GridArchive(5)
        original = ArchiveElite(bin=(2, 2), fitness=0.7)
        archive.try_insert(original)
        archive.try_insert(ArchiveElite(bin=(2, 2), fitness=0.1))
        stored = archive.get(2, 2)
        assert stored is not None
        self.assertEqual(stored.fitness, 0.7)
        self.assertEqual(stored.bin, (2, 2))

    def test_filled_and_empty_counts(self) -> None:
        archive = GridArchive(4)
        self.assertEqual(archive.empty_count(), 16)
        archive.try_insert(ArchiveElite(bin=(0, 0), fitness=0.1))
        archive.try_insert(ArchiveElite(bin=(3, 3), fitness=0.2))
        self.assertEqual(archive.filled_count(), 2)
        self.assertEqual(archive.empty_count(), 14)


if __name__ == "__main__":
    unittest.main()

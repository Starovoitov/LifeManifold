"""Tests for bounded archive grid neighbor helpers."""

from __future__ import annotations

import unittest

from worldspace.illuminators.grid_neighbors import (
    cardinal_neighbors_bounded,
    moore_neighbors_bounded,
)


class TestGridNeighbors(unittest.TestCase):
    def test_cardinal_center(self) -> None:
        self.assertEqual(
            cardinal_neighbors_bounded(2, 2, 5),
            ((1, 2), (3, 2), (2, 1), (2, 3)),
        )

    def test_cardinal_corner(self) -> None:
        self.assertEqual(cardinal_neighbors_bounded(0, 0, 5), ((1, 0), (0, 1)))

    def test_moore_center(self) -> None:
        neighbors = moore_neighbors_bounded(2, 2, 5)
        self.assertEqual(len(neighbors), 8)
        self.assertNotIn((2, 2), neighbors)

    def test_moore_corner(self) -> None:
        self.assertEqual(
            moore_neighbors_bounded(0, 0, 5),
            ((0, 1), (1, 0), (1, 1)),
        )


if __name__ == "__main__":
    unittest.main()

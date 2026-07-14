"""Unit tests for paired Vargha–Delaney A₁₂ in Q1 statistics."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.analyze_q1_statistics import vargha_delaney_a12_paired


class VarghaDelaneyA12Tests(unittest.TestCase):
    def test_all_wins_greater(self) -> None:
        delta = np.array([1.0, 2.0, 0.5])
        self.assertAlmostEqual(
            vargha_delaney_a12_paired(delta, direction="greater"), 1.0
        )

    def test_all_losses_greater(self) -> None:
        delta = np.array([-1.0, -2.0])
        self.assertAlmostEqual(
            vargha_delaney_a12_paired(delta, direction="greater"), 0.0
        )

    def test_half_wins(self) -> None:
        delta = np.array([1.0, -1.0])
        self.assertAlmostEqual(
            vargha_delaney_a12_paired(delta, direction="greater"), 0.5
        )

    def test_ties_count_half(self) -> None:
        delta = np.array([0.0, 1.0])
        self.assertAlmostEqual(
            vargha_delaney_a12_paired(delta, direction="greater"), 0.75
        )

    def test_less_direction(self) -> None:
        delta = np.array([-0.1, 0.2, -0.3])
        self.assertAlmostEqual(
            vargha_delaney_a12_paired(delta, direction="less"), 2 / 3
        )

    def test_empty_returns_nan(self) -> None:
        self.assertTrue(np.isnan(vargha_delaney_a12_paired(np.array([]))))


if __name__ == "__main__":
    unittest.main()

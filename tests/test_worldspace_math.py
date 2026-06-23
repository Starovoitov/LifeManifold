"""Unit tests for worldspace.math helpers."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from worldspace import math as ws_math
from worldspace.metrics import METRICS_VECTOR_DIM


class TestKmeansLloydOnMemmap(unittest.TestCase):
    def test_assigns_valid_cluster_ids_on_memmap(self) -> None:
        n = 64
        k = 4
        fd_metrics, metrics_path = tempfile.mkstemp(suffix=".metrics")
        fd_labels, labels_path = tempfile.mkstemp(suffix=".labels")
        os.close(fd_metrics)
        os.close(fd_labels)
        mm: np.memmap | None = None
        labels: np.memmap | None = None
        try:
            mm = np.memmap(
                metrics_path,
                dtype=np.float32,
                mode="w+",
                shape=(n, METRICS_VECTOR_DIM),
            )
            labels = np.memmap(labels_path, dtype=np.int32, mode="w+", shape=(n,))
            rng = np.random.default_rng(0)
            mm[:] = rng.standard_normal((n, METRICS_VECTOR_DIM)).astype(np.float32)
            labels[:] = -1

            ws_math.kmeans_lloyd_on_memmap(mm, labels, n, k)

            assigned = np.asarray(labels[:n], dtype=np.int32)
            self.assertEqual(assigned.shape, (n,))
            self.assertTrue(np.all(assigned >= 0))
            self.assertTrue(np.all(assigned < k))
            self.assertGreater(len(set(assigned.tolist())), 1)
        finally:
            del mm
            del labels
            os.unlink(metrics_path)
            os.unlink(labels_path)

    def test_reproducible_for_same_data(self) -> None:
        n = 32
        k = 3
        rng = np.random.default_rng(7)
        data = rng.standard_normal((n, METRICS_VECTOR_DIM)).astype(np.float32)

        def run_once() -> np.ndarray:
            fd_metrics, metrics_path = tempfile.mkstemp(suffix=".metrics")
            fd_labels, labels_path = tempfile.mkstemp(suffix=".labels")
            os.close(fd_metrics)
            os.close(fd_labels)
            mm: np.memmap | None = None
            labels: np.memmap | None = None
            try:
                mm = np.memmap(
                    metrics_path,
                    dtype=np.float32,
                    mode="w+",
                    shape=(n, METRICS_VECTOR_DIM),
                )
                labels = np.memmap(labels_path, dtype=np.int32, mode="w+", shape=(n,))
                mm[:] = data
                ws_math.kmeans_lloyd_on_memmap(mm, labels, n, k)
                return np.asarray(labels[:n], dtype=np.int32).copy()
            finally:
                del mm
                del labels
                os.unlink(metrics_path)
                os.unlink(labels_path)

        first = run_once()
        second = run_once()
        np.testing.assert_array_equal(first, second)

    def test_no_op_when_n_zero(self) -> None:
        fd_metrics, metrics_path = tempfile.mkstemp(suffix=".metrics")
        fd_labels, labels_path = tempfile.mkstemp(suffix=".labels")
        os.close(fd_metrics)
        os.close(fd_labels)
        mm: np.memmap | None = None
        labels: np.memmap | None = None
        try:
            mm = np.memmap(
                metrics_path,
                dtype=np.float32,
                mode="w+",
                shape=(1, METRICS_VECTOR_DIM),
            )
            labels = np.memmap(labels_path, dtype=np.int32, mode="w+", shape=(1,))
            labels[0] = 99
            ws_math.kmeans_lloyd_on_memmap(mm, labels, 0, 2)
            self.assertEqual(int(labels[0]), 99)
        finally:
            del mm
            del labels
            os.unlink(metrics_path)
            os.unlink(labels_path)


def _neighbor_count_roll_reference(grid: np.ndarray) -> np.ndarray:
    """Legacy double-np.roll Moore count (reference for stencil equivalence tests)."""
    total = np.zeros_like(grid, dtype=np.int16)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            total += np.roll(np.roll(grid, dx, axis=0), dy, axis=1)
    return total


def _topology_interface_index_roll_reference(life: np.ndarray) -> float:
    if life.size == 0:
        return 0.0
    g = life.astype(np.float32)
    diff_sum = np.zeros_like(g, dtype=np.float32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nb = np.roll(np.roll(g, dx, axis=0), dy, axis=1)
            diff_sum += (nb != g).astype(np.float32)
    return float(np.clip(diff_sum.mean() / 8.0, 0.0, 1.0))


def _topology_interface_strength_map_roll_reference(life: np.ndarray) -> np.ndarray:
    if life.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    g = life.astype(np.float32)
    diff_sum = np.zeros_like(g, dtype=np.float32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nb = np.roll(np.roll(g, dx, axis=0), dy, axis=1)
            diff_sum += (nb != g).astype(np.float32)
    return np.clip(diff_sum / 8.0, 0.0, 1.0).astype(np.float64)


class TestMooreStencil(unittest.TestCase):
    def test_neighbor_count_small_grid_manual(self) -> None:
        grid = np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 1],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(
            ws_math.neighbor_count(grid),
            _neighbor_count_roll_reference(grid),
        )

    def test_neighbor_count_matches_roll_reference(self) -> None:
        rng = np.random.default_rng(0)
        for n in (4, 8, 32):
            with self.subTest(n=n):
                grid = rng.integers(0, 2, size=(n, n), dtype=np.uint8)
                np.testing.assert_array_equal(
                    ws_math.neighbor_count(grid),
                    _neighbor_count_roll_reference(grid),
                )

    def test_topology_interface_index_matches_roll_reference(self) -> None:
        rng = np.random.default_rng(1)
        for n in (4, 16, 32):
            with self.subTest(n=n):
                life = rng.integers(0, 2, size=(n, n), dtype=np.uint8)
                self.assertAlmostEqual(
                    ws_math.topology_interface_index(life),
                    _topology_interface_index_roll_reference(life),
                )

    def test_topology_interface_strength_map_matches_roll_reference(self) -> None:
        rng = np.random.default_rng(2)
        for n in (4, 16, 32):
            with self.subTest(n=n):
                life = rng.integers(0, 2, size=(n, n), dtype=np.uint8)
                np.testing.assert_allclose(
                    ws_math.topology_interface_strength_map(life),
                    _topology_interface_strength_map_roll_reference(life),
                )

    def test_empty_grid_topology_helpers(self) -> None:
        empty = np.zeros((0, 0), dtype=np.uint8)
        self.assertEqual(ws_math.topology_interface_index(empty), 0.0)
        np.testing.assert_array_equal(
            ws_math.topology_interface_strength_map(empty),
            np.zeros((0, 0), dtype=np.float64),
        )


if __name__ == "__main__":
    unittest.main()

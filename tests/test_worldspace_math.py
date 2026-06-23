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


if __name__ == "__main__":
    unittest.main()

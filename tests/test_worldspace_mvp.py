import json
import os
import tempfile
import unittest

import numpy as np

from src.worldspace import math as ws_math
from src.worldspace.generators import RandomWorldGenerator
from src.worldspace.metrics import METRIC_INDEX_AVERAGE_LIFESPAN, METRICS_VECTOR_DIM
from src.worldspace.pipeline import stream_world_space_to_jsonl


class TestWorldSpaceMVP(unittest.TestCase):
    def test_stream_jsonl_line_count_and_metrics_bounds(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            generator = RandomWorldGenerator(grid_size=12, steps=20)
            stream_world_space_to_jsonl(
                generator, 5, path, k_clusters=2, echo_stdout=False
            )
            with open(path, encoding="utf-8") as f:
                lines = [ln for ln in f if ln.strip()]
            self.assertEqual(len(lines), 5)
            row = json.loads(lines[0])
            self.assertIn("world", row)
            self.assertIn("metrics", row)
            self.assertIn("embedding_2d", row)
            self.assertEqual(len(row["embedding_2d"]), 2)
            self.assertIn("cluster_id", row)
            st = row["metrics"]["stability"]
            self.assertGreaterEqual(st, 0.0)
            self.assertLessEqual(st, 1.0)
            self.assertIn("interestingness", row["metrics"])
        finally:
            os.unlink(path)

    def test_lifespan_orthogonal_embedding_axes(self):
        rng = np.random.default_rng(0)
        n, d = 50, METRICS_VECTOR_DIM
        x = rng.standard_normal((n, d))
        x[:, METRIC_INDEX_AVERAGE_LIFESPAN] *= 40.0
        sum_x = x.sum(axis=0)
        sum_xx = sum(np.outer(x[i], x[i]) for i in range(n))
        mean, basis = ws_math.lifespan_orthogonal_mean_and_basis_2d(sum_x, sum_xx, n)
        self.assertEqual(basis.shape, (d, 2))
        u, w = basis[:, 0], basis[:, 1]
        self.assertAlmostEqual(float(np.linalg.norm(u)), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(w)), 1.0)
        self.assertAlmostEqual(float(u @ w), 0.0)
        self.assertAlmostEqual(float(w[METRIC_INDEX_AVERAGE_LIFESPAN]), 0.0)
        vec = x[3]
        emb = ws_math.project_pca_2d(vec, mean, basis)
        self.assertAlmostEqual(emb[0], vec[METRIC_INDEX_AVERAGE_LIFESPAN] - mean[METRIC_INDEX_AVERAGE_LIFESPAN])


if __name__ == "__main__":
    unittest.main()

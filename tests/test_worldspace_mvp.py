import json
import os
import tempfile
import unittest

import numpy as np

from src.worldspace.generators import RandomWorldGenerator
from src.worldspace.metrics import METRIC_KEYS, METRICS_VECTOR_DIM
from src.worldspace.pipeline import (
    _fit_dominant_metric_orthogonal_pca,
    _project_dominant_metric_orthogonal,
    stream_world_space_to_jsonl,
)


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
            self.assertIn("embedding_axes", row)
            self.assertIn("x_metric", row["embedding_axes"])
            self.assertIn("cluster_id", row)
            st = row["metrics"]["stability"]
            self.assertGreaterEqual(st, 0.0)
            self.assertLessEqual(st, 1.0)
            self.assertIn("interestingness", row["metrics"])
        finally:
            os.unlink(path)

    def test_dominant_metric_embedding_axes(self):
        rng = np.random.default_rng(0)
        n, d = 50, METRICS_VECTOR_DIM
        x = rng.standard_normal((n, d))
        x[:, 4] *= 30.0
        j = 4
        mean_exp = x.mean(axis=0)
        mean, j_fit, axis_name, pca = _fit_dominant_metric_orthogonal_pca(x)
        self.assertEqual(j_fit, j)
        self.assertEqual(axis_name, METRIC_KEYS[j])
        np.testing.assert_allclose(mean, mean_exp)
        self.assertIsNotNone(pca)
        np.testing.assert_allclose(pca.mean_, np.delete(mean_exp, j), rtol=0, atol=1e-9)
        vec = x[3]
        emb = _project_dominant_metric_orthogonal(vec, mean, j, pca)
        self.assertAlmostEqual(emb[0], vec[j] - mean[j])
        self.assertFalse(np.isnan(emb[1]))


if __name__ == "__main__":
    unittest.main()

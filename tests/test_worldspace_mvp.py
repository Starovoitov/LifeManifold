import json
import os
import tempfile
import unittest

import numpy as np

from src.worldspace.generators import GeneticWorldGenerator, RandomWorldGenerator
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

    def test_genetic_generator_bounds_and_count(self):
        generator = GeneticWorldGenerator(
            grid_size=10,
            steps=10,
            population_size=6,
            elite_count=2,
            mutation_scale=0.03,
            seed=7,
        )
        worlds = generator.generate(4)
        self.assertEqual(len(worlds), 4)
        for w in worlds:
            self.assertGreaterEqual(w.noise, 0.0)
            self.assertLessEqual(w.noise, 0.2)
            self.assertGreaterEqual(w.resource_regen, 0.0)
            self.assertLessEqual(w.resource_regen, 0.5)
            self.assertGreaterEqual(w.predation, 0.0)
            self.assertLessEqual(w.predation, 1.0)
            self.assertGreaterEqual(len(w.birth), 1)
            self.assertGreaterEqual(len(w.survival), 1)
            self.assertTrue(all(0 <= v <= 8 for v in w.birth))
            self.assertTrue(all(0 <= v <= 8 for v in w.survival))

    def test_genetic_generator_preserves_diversity(self):
        generator = GeneticWorldGenerator(
            grid_size=10,
            steps=10,
            population_size=8,
            elite_count=3,
            mutation_scale=0.02,
            seed=0,
        )
        worlds = generator.generate(20)
        signatures = {
            (
                tuple(w.birth),
                tuple(w.survival),
                round(w.noise, 4),
                round(w.resource_regen, 4),
                round(w.predation, 4),
            )
            for w in worlds
        }
        self.assertGreaterEqual(len(signatures), 10)


if __name__ == "__main__":
    unittest.main()

import unittest

from src.worldspace.generators import RandomWorldGenerator
from src.worldspace.pipeline import explore_world_space


class TestWorldSpaceMVP(unittest.TestCase):
    def test_worldspace_pipeline_returns_points(self):
        generator = RandomWorldGenerator(grid_size=20, steps=30)
        points = explore_world_space(generator, n_worlds=5, k_clusters=2)
        self.assertEqual(len(points), 5)
        for point in points:
            self.assertEqual(len(point.embedding_2d), 2)
            self.assertGreaterEqual(point.metrics.stability, 0.0)
            self.assertLessEqual(point.metrics.stability, 1.0)


if __name__ == "__main__":
    unittest.main()

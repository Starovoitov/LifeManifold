import unittest


class TestNeuralWorldGenerator(unittest.TestCase):
    def test_generate_from_bundled_spec(self):
        from src.worldspace.neural_world import NeuralWorldGenerator

        gen = NeuralWorldGenerator()
        worlds = gen.generate(4)
        self.assertEqual(len(worlds), 4)
        for i, w in enumerate(worlds):
            self.assertEqual(w.seed, i)
            self.assertTrue(1 <= len(w.birth) <= 4)
            self.assertTrue(2 <= len(w.survival) <= 5)
            self.assertGreaterEqual(w.noise, 0.0)
            self.assertLessEqual(w.noise, 0.08 + 1e-6)
            self.assertGreaterEqual(w.resource_regen, 0.0)
            self.assertLessEqual(w.resource_regen, 0.2 + 1e-6)
            self.assertGreaterEqual(w.predation, 0.0)
            self.assertLessEqual(w.predation, 0.5 + 1e-6)

    def test_lazy_export_from_generators(self):
        """``NeuralWorldGenerator`` resolves via PEP 562 without importing torch eagerly."""
        from src.worldspace import generators as g

        self.assertTrue(callable(getattr(g, "NeuralWorldGenerator")))


if __name__ == "__main__":
    unittest.main()

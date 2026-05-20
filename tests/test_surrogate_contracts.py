from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from worldspace.surrogate import StubSurrogate, SurrogateFacade, get_surrogate
from worldspace.surrogate.types import SurrogateConfig, SurrogatePrediction


class SurrogateContractsTests(unittest.TestCase):
    def test_get_surrogate_returns_stub_when_disabled(self) -> None:
        config = SurrogateConfig(
            enabled=False,
            model_type="lightgbm",
            checkpoint="artifacts/surrogate/checkpoints/latest.pkl",
            stub_mean=0.45,
            stub_uncertainty=0.85,
        )
        surrogate = get_surrogate(config)
        self.assertIsInstance(surrogate, StubSurrogate)

    def test_get_surrogate_returns_stub_without_checkpoint(self) -> None:
        config = SurrogateConfig(
            enabled=True,
            model_type="lightgbm",
            checkpoint="artifacts/surrogate/checkpoints/missing.pkl",
            stub_mean=0.45,
            stub_uncertainty=0.85,
        )
        surrogate = get_surrogate(config)
        self.assertIsInstance(surrogate, StubSurrogate)

    def test_get_surrogate_returns_facade_with_existing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pkl"
            checkpoint.write_text("placeholder", encoding="utf-8")
            config = SurrogateConfig(
                enabled=True,
                model_type="lightgbm",
                checkpoint=str(checkpoint),
                stub_mean=0.45,
                stub_uncertainty=0.85,
            )
            surrogate = get_surrogate(config)
            self.assertIsInstance(surrogate, SurrogateFacade)

    def test_predict_contract_is_stable_for_stub_and_facade(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pkl"
            checkpoint.write_text("placeholder", encoding="utf-8")
            config = SurrogateConfig(
                enabled=True,
                model_type="lightgbm",
                checkpoint=str(checkpoint),
                stub_mean=0.45,
                stub_uncertainty=0.85,
            )
            facade = get_surrogate(config)
            prediction = facade.predict(world_spec={})
            self.assertIsInstance(prediction, SurrogatePrediction)
            self.assertEqual(set(prediction.measures.keys()), {"stability", "diversity"})
            self.assertEqual(set(prediction.components.keys()), {"stability", "diversity"})
            self.assertAlmostEqual(prediction.fitness, 0.45)
            self.assertAlmostEqual(prediction.uncertainty, 0.85)


if __name__ == "__main__":
    unittest.main()

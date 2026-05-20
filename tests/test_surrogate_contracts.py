from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from worldspace.illuminators.evaluation import apply_canonical_seed, canonical_seed
from worldspace.specs.spec import WorldSpec
from worldspace.surrogate.feature_extractor import extract as extract_features
from worldspace.surrogate import StubSurrogate, SurrogateFacade, get_surrogate
from worldspace.surrogate.types import SurrogateConfig, SurrogatePrediction


class SurrogateContractsTests(unittest.TestCase):
    def _sample_spec(self) -> WorldSpec:
        return WorldSpec(
            birth=[3],
            survival=[2, 3],
            noise=0.1,
            resource_regen=0.2,
            predation=0.05,
            cell_types=["life", "food"],
            grid_size=30,
            steps=220,
            seed=0,
        )

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
            prediction = facade.predict(world_spec=self._sample_spec())
            self.assertIsInstance(prediction, SurrogatePrediction)
            self.assertEqual(
                set(prediction.measures.keys()), {"stability", "diversity"}
            )
            self.assertEqual(
                set(prediction.components.keys()), {"stability", "diversity"}
            )
            self.assertAlmostEqual(prediction.fitness, 0.45)
            self.assertAlmostEqual(prediction.uncertainty, 0.85)

    def test_feature_extractor_requires_canonical_seed(self) -> None:
        spec = self._sample_spec()
        with self.assertRaises(ValueError):
            extract_features(spec)

    def test_feature_extractor_is_deterministic_for_canonicalized_spec(self) -> None:
        spec = self._sample_spec()
        apply_canonical_seed(spec)
        first = extract_features(spec)
        second = extract_features(spec)
        self.assertEqual(first.tolist(), second.tolist())

    def test_apply_canonical_seed_is_idempotent(self) -> None:
        spec = self._sample_spec()
        first = apply_canonical_seed(spec)
        second = apply_canonical_seed(spec)
        self.assertEqual(first, second)
        self.assertEqual(spec.seed, first)
        self.assertEqual(spec.seed, canonical_seed(spec))


if __name__ == "__main__":
    unittest.main()

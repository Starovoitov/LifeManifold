from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from worldspace.illuminators.evaluation import apply_canonical_seed, canonical_seed
from worldspace.specs.spec import WorldSpec
from worldspace.surrogate.feature_extractor import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    extract as extract_features,
)
from worldspace.surrogate.genome_features import FEATURE_DIM
from worldspace.surrogate.model import TARGET_KEYS, SurrogateModel
from worldspace.surrogate import StubSurrogate, SurrogateFacade, get_surrogate
from worldspace.surrogate.types import SurrogateConfig, SurrogatePrediction
from worldspace.surrogate.utils import compute_fitness_from_prediction


def _write_model_checkpoint(path: Path, *, mean: float = 0.45) -> None:
    model = SurrogateModel()
    model.set_component_defaults(mean)
    with path.open("wb") as fh:
        pickle.dump(model, fh)


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
            _write_model_checkpoint(checkpoint)
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
            _write_model_checkpoint(checkpoint)
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
            self.assertEqual(set(prediction.components.keys()), set(TARGET_KEYS))
            self.assertAlmostEqual(prediction.fitness, 0.45)
            self.assertAlmostEqual(prediction.uncertainty, 0.85)

    def test_stub_and_facade_use_consistent_component_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pkl"
            _write_model_checkpoint(checkpoint)
            enabled_config = SurrogateConfig(
                enabled=True,
                model_type="lightgbm",
                checkpoint=str(checkpoint),
                stub_mean=0.45,
                stub_uncertainty=0.85,
            )
            disabled_config = SurrogateConfig(
                enabled=False,
                model_type="lightgbm",
                checkpoint=str(checkpoint),
                stub_mean=0.45,
                stub_uncertainty=0.85,
            )
            facade_prediction = get_surrogate(enabled_config).predict(
                self._sample_spec()
            )
            stub_prediction = get_surrogate(disabled_config).predict(
                self._sample_spec()
            )
            self.assertEqual(
                facade_prediction.components["early_extinction_prob"],
                stub_prediction.components["early_extinction_prob"],
            )

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

    def test_feature_extractor_v2_shape_and_schema(self) -> None:
        spec = self._sample_spec()
        apply_canonical_seed(spec)
        vector = extract_features(spec)
        self.assertEqual(FEATURE_SCHEMA_VERSION, "2.0")
        self.assertEqual(len(FEATURE_NAMES), FEATURE_DIM)
        self.assertEqual(vector.shape, (FEATURE_DIM,))

    def test_feature_extractor_v2_breaks_v1_density_aliasing(self) -> None:
        left = WorldSpec(
            birth=[2, 6],
            survival=[1, 7],
            noise=0.1,
            resource_regen=0.2,
            predation=0.05,
            cell_types=["life", "food"],
            grid_size=30,
            steps=220,
            seed=0,
        )
        right = WorldSpec(
            birth=[1, 7],
            survival=[2, 6],
            noise=0.1,
            resource_regen=0.2,
            predation=0.05,
            cell_types=["life", "food"],
            grid_size=30,
            steps=220,
            seed=0,
        )
        apply_canonical_seed(left)
        apply_canonical_seed(right)
        left_vector = extract_features(left)
        right_vector = extract_features(right)
        self.assertNotEqual(left_vector.tolist(), right_vector.tolist())

    def test_apply_canonical_seed_is_idempotent(self) -> None:
        spec = self._sample_spec()
        first = apply_canonical_seed(spec)
        second = apply_canonical_seed(spec)
        self.assertEqual(first, second)
        self.assertEqual(spec.seed, first)
        self.assertEqual(spec.seed, canonical_seed(spec))

    def test_compute_fitness_from_prediction_uses_strategy_a_components(self) -> None:
        prediction = SurrogatePrediction(
            components={
                "stability": 0.3,
                "diversity": 0.4,
                "oscillation_score": 0.5,
                "topology_interface_index": 0.7,
                "topology_window_heterogeneity": 0.1,
                "final_density": 0.6,
                "early_extinction_prob": 0.0,
            },
            measures={"stability": 0.3, "diversity": 0.4},
            fitness=0.0,
            uncertainty=0.1,
        )
        fitness = compute_fitness_from_prediction(prediction)
        self.assertAlmostEqual(fitness, 0.47)

    def test_surrogate_facade_uses_cache_for_same_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pkl"
            _write_model_checkpoint(checkpoint)
            config = SurrogateConfig(
                enabled=True,
                model_type="lightgbm",
                checkpoint=str(checkpoint),
                stub_mean=0.45,
                stub_uncertainty=0.85,
            )
            facade = get_surrogate(config)
            self.assertIsInstance(facade, SurrogateFacade)
            spec = self._sample_spec()
            first = facade.predict(spec)
            second = facade.predict(spec)
            self.assertEqual(first, second)
            self.assertEqual(facade.cache_hits(), 1)


if __name__ == "__main__":
    unittest.main()

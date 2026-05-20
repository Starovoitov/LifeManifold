"""Determinism tests for surrogate predict (E6.1)."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from worldspace.illuminators.evaluation import apply_canonical_seed
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec
from worldspace.surrogate import StubSurrogate, get_surrogate
from worldspace.surrogate.model import SurrogateModel
from worldspace.surrogate.surrogate import SurrogateFacade, build_surrogate_facade
from worldspace.surrogate.types import SurrogateConfig, SurrogatePrediction


def _sample_spec() -> WorldSpec:
    return WorldSpec(
        birth=[1, 3],
        survival=[2, 3],
        noise=0.02,
        resource_regen=0.05,
        predation=0.1,
        cell_types=CANONICAL_CELL_TYPES.copy(),
        grid_size=8,
        steps=200,
        seed=0,
    )


def _prediction_tuple(pred: SurrogatePrediction) -> tuple:
    return (
        pred.fitness,
        pred.uncertainty,
        tuple(sorted(pred.components.items())),
        tuple(sorted(pred.measures.items())),
    )


class TestSurrogatePredictDeterminism(unittest.TestCase):
    def test_stub_predict_twice_identical(self) -> None:
        stub = StubSurrogate(mean=0.42, uncertainty=0.88)
        spec = _sample_spec()
        first = stub.predict(spec)
        second = stub.predict(spec)
        self.assertEqual(_prediction_tuple(first), _prediction_tuple(second))

    def test_facade_predict_twice_identical(self) -> None:
        model = SurrogateModel()
        model.set_component_defaults(0.45)
        facade = build_surrogate_facade(model, uncertainty_fallback=0.85)
        spec = _sample_spec()
        first = facade.predict(spec)
        second = facade.predict(spec)
        self.assertEqual(_prediction_tuple(first), _prediction_tuple(second))

    def test_facade_cache_does_not_change_value(self) -> None:
        model = SurrogateModel()
        model.set_component_defaults(0.45)
        facade = build_surrogate_facade(model, uncertainty_fallback=0.85)
        spec = _sample_spec()
        first = facade.predict(spec)
        second = facade.predict(spec)
        third = facade.predict(spec)
        self.assertEqual(_prediction_tuple(first), _prediction_tuple(third))
        self.assertEqual(facade.cache_hits(), 2)
        self.assertEqual(_prediction_tuple(second), _prediction_tuple(first))

    def test_get_surrogate_checkpoint_predict_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pkl"
            model = SurrogateModel()
            model.set_component_defaults(0.45)
            with checkpoint.open("wb") as fh:
                pickle.dump(model, fh)
            config = SurrogateConfig(
                enabled=True,
                model_type="lightgbm",
                checkpoint=str(checkpoint),
                stub_mean=0.45,
                stub_uncertainty=0.85,
            )
            surrogate = get_surrogate(config)
            self.assertIsInstance(surrogate, SurrogateFacade)
            spec = _sample_spec()
            first = surrogate.predict(spec)
            second = surrogate.predict(spec)
            self.assertEqual(_prediction_tuple(first), _prediction_tuple(second))

    def test_enabled_stub_path_matches_disabled_yaml_stub(self) -> None:
        """Stub path must be stable when checkpoint is missing."""
        config = SurrogateConfig(
            enabled=True,
            model_type="lightgbm",
            checkpoint="artifacts/surrogate/checkpoints/does_not_exist.pkl",
            stub_mean=0.33,
            stub_uncertainty=0.77,
        )
        spec = _sample_spec()
        apply_canonical_seed(spec)
        first = get_surrogate(config).predict(spec)
        second = get_surrogate(config).predict(spec)
        self.assertEqual(_prediction_tuple(first), _prediction_tuple(second))


if __name__ == "__main__":
    unittest.main()

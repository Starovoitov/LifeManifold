"""Unit tests for surrogate ``predict_batch``."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from worldspace.illuminators.evaluation import apply_canonical_seed
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec
from worldspace.surrogate.feature_extractor import extract, extract_batch
from worldspace.surrogate.model import SurrogateModel
from worldspace.surrogate.surrogate import StubSurrogate, build_surrogate_facade
from worldspace.surrogate.types import SurrogatePrediction


def _sample_spec(*, seed: int = 0) -> WorldSpec:
    spec = WorldSpec(
        birth=[1, 3],
        survival=[2, 3],
        noise=0.02,
        resource_regen=0.05,
        predation=0.1,
        cell_types=CANONICAL_CELL_TYPES.copy(),
        grid_size=8,
        steps=200,
        seed=seed,
    )
    apply_canonical_seed(spec)
    return spec


def _prediction_tuple(pred: SurrogatePrediction) -> tuple:
    return (
        pred.fitness,
        pred.uncertainty,
        tuple(sorted(pred.components.items())),
        tuple(sorted(pred.measures.items())),
    )


class TestSurrogateBatch(unittest.TestCase):
    def test_extract_batch_matches_sequential(self) -> None:
        specs = [_sample_spec(seed=index) for index in range(4)]
        batch = extract_batch(specs)
        sequential = np.stack([extract(spec) for spec in specs], axis=0)
        np.testing.assert_allclose(batch, sequential)

    def test_extract_batch_empty(self) -> None:
        batch = extract_batch([])
        self.assertEqual(batch.shape, (0, 24))

    def test_stub_predict_batch_empty(self) -> None:
        stub = StubSurrogate(mean=0.42, uncertainty=0.88)
        self.assertEqual(stub.predict_batch([]), [])

    def test_stub_predict_batch_matches_predict(self) -> None:
        stub = StubSurrogate(mean=0.42, uncertainty=0.88)
        specs = [_sample_spec(seed=index) for index in range(3)]
        batch = stub.predict_batch(specs)
        self.assertEqual(len(batch), 3)
        for spec, prediction in zip(specs, batch):
            self.assertEqual(
                _prediction_tuple(prediction),
                _prediction_tuple(stub.predict(spec)),
            )

    def test_facade_predict_batch_matches_predict(self) -> None:
        model = SurrogateModel()
        model.set_component_defaults(0.45)
        facade = build_surrogate_facade(model, uncertainty_fallback=0.85)
        specs = [_sample_spec(seed=index) for index in range(4)]
        batch = facade.predict_batch(specs)
        self.assertEqual(len(batch), 4)
        for spec, prediction in zip(specs, batch):
            self.assertEqual(
                _prediction_tuple(prediction),
                _prediction_tuple(facade.predict(spec)),
            )

    def test_model_predict_components_batch_matches_sequential(self) -> None:
        model = SurrogateModel()
        model.set_component_defaults(0.45)
        specs = [_sample_spec(seed=index) for index in range(3)]
        matrix = extract_batch(specs)
        batch = model.predict_components_batch(matrix)
        sequential = [model.predict_components(matrix[row]) for row in range(3)]
        self.assertEqual(batch, sequential)

    def test_facade_predict_batch_uses_cache(self) -> None:
        model = SurrogateModel()
        model.set_component_defaults(0.45)
        facade = build_surrogate_facade(model, uncertainty_fallback=0.85)
        specs = [_sample_spec(seed=index) for index in range(2)]
        first = facade.predict_batch(specs)
        second = facade.predict_batch(list(reversed(specs)))
        self.assertEqual(facade.cache_hits(), 2)
        self.assertEqual(_prediction_tuple(first[0]), _prediction_tuple(second[1]))
        self.assertEqual(_prediction_tuple(first[1]), _prediction_tuple(second[0]))

    def test_facade_predict_batch_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pkl"
            model = SurrogateModel()
            model.set_component_defaults(0.45)
            with checkpoint.open("wb") as fh:
                pickle.dump(model, fh)
            facade = build_surrogate_facade(model, uncertainty_fallback=0.85)
            specs = [_sample_spec(seed=index) for index in (0, 1, 2)]
            batch = facade.predict_batch(specs)
            singles = [facade.predict(spec) for spec in specs]
            self.assertEqual(
                [_prediction_tuple(pred) for pred in batch],
                [_prediction_tuple(pred) for pred in singles],
            )


if __name__ == "__main__":
    unittest.main()

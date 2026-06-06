"""Checkpoint feature-dimension guard for v2 extractor vs legacy v1 pickles."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec
from worldspace.surrogate import StubSurrogate, SurrogateFacade, get_surrogate
from worldspace.surrogate.checkpoint_io import load_surrogate_checkpoint
from worldspace.surrogate.model import (
    EXPECTED_FEATURE_DIM,
    TARGET_KEYS,
    SurrogateModel,
    checkpoint_feature_dim,
    checkpoint_matches_extractor,
)
from worldspace.surrogate.surrogate import build_surrogate_facade
from worldspace.surrogate.synthetic_buffer import write_synthetic_buffer
from worldspace.surrogate.training import holdout_split, load_buffer
from worldspace.surrogate.types import SurrogateConfig


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


def _write_legacy_v1_checkpoint(path: Path) -> None:
    """Pickle a LightGBM checkpoint trained on 8 features (legacy nightly.pkl shape)."""
    import lightgbm as lgb

    model = SurrogateModel()
    rng = np.random.default_rng(0)
    features = rng.random((64, 8))
    estimator = lgb.LGBMRegressor(n_estimators=4, random_state=0, verbosity=-1)
    estimator.fit(features, rng.random(64))
    model._component_means = {key: 0.45 for key in TARGET_KEYS}
    model._uses_lightgbm = True
    model._ensemble = {key: [estimator] for key in TARGET_KEYS}
    with path.open("wb") as fh:
        pickle.dump(model, fh)


def _write_inconsistent_dim_checkpoint(path: Path) -> None:
    """Pickle a checkpoint whose ensemble members disagree on input width."""
    import lightgbm as lgb

    model = SurrogateModel()
    rng = np.random.default_rng(1)
    est_8 = lgb.LGBMRegressor(n_estimators=2, random_state=0, verbosity=-1)
    est_8.fit(rng.random((32, 8)), rng.random(32))
    est_21 = lgb.LGBMRegressor(n_estimators=2, random_state=1, verbosity=-1)
    est_21.fit(rng.random((32, 21)), rng.random(32))
    model._component_means = {key: 0.45 for key in TARGET_KEYS}
    model._uses_lightgbm = True
    model._ensemble = {
        key: [est_8 if key == "stability" else est_21] for key in TARGET_KEYS
    }
    with path.open("wb") as fh:
        pickle.dump(model, fh)


def _write_v2_checkpoint(path: Path) -> None:
    """Train and pickle a v2 checkpoint on synthetic buffer rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        buffer_path = Path(tmpdir) / "buffer.jsonl"
        write_synthetic_buffer(buffer_path, n_samples=200, seed=17)
        features, targets = load_buffer(buffer_path)
        x_train, y_train, _, _ = holdout_split(features, targets)
        model = SurrogateModel(model_type="lightgbm", random_state=42, ensemble_size=4)
        model.fit(x_train, y_train)
        if not model._uses_lightgbm:
            raise unittest.SkipTest("lightgbm backend unavailable")
        with path.open("wb") as fh:
            pickle.dump(model, fh)


@unittest.skipIf(find_spec("lightgbm") is None, "lightgbm not installed")
class TestSurrogateCheckpointGuard(unittest.TestCase):
    def test_legacy_v1_dim_mismatch_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "nightly.pkl"
            _write_legacy_v1_checkpoint(checkpoint)
            model = load_surrogate_checkpoint(checkpoint)
            self.assertEqual(checkpoint_feature_dim(model), 8)
            self.assertNotEqual(8, EXPECTED_FEATURE_DIM)
            self.assertFalse(checkpoint_matches_extractor(model))

    def test_get_surrogate_stubs_on_v1_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "nightly.pkl"
            _write_legacy_v1_checkpoint(checkpoint)
            config = SurrogateConfig(
                enabled=True,
                model_type="lightgbm",
                checkpoint=str(checkpoint),
                stub_mean=0.45,
                stub_uncertainty=0.85,
            )
            surrogate = get_surrogate(config)
            self.assertIsInstance(surrogate, StubSurrogate)
            prediction = surrogate.predict(_sample_spec())
            self.assertAlmostEqual(prediction.fitness, 0.45)

    def test_get_surrogate_stubs_on_inconsistent_checkpoint_dims(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "corrupt.pkl"
            _write_inconsistent_dim_checkpoint(checkpoint)
            model = load_surrogate_checkpoint(checkpoint)
            with self.assertRaises(ValueError):
                checkpoint_matches_extractor(model)
            config = SurrogateConfig(
                enabled=True,
                model_type="lightgbm",
                checkpoint=str(checkpoint),
                stub_mean=0.45,
                stub_uncertainty=0.85,
            )
            surrogate = get_surrogate(config)
            self.assertIsInstance(surrogate, StubSurrogate)

    def test_get_surrogate_loads_v2_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "nightly_v2.pkl"
            _write_v2_checkpoint(checkpoint)
            model = load_surrogate_checkpoint(checkpoint)
            self.assertEqual(checkpoint_feature_dim(model), EXPECTED_FEATURE_DIM)
            self.assertTrue(checkpoint_matches_extractor(model))
            config = SurrogateConfig(
                enabled=True,
                model_type="lightgbm",
                checkpoint=str(checkpoint),
                stub_mean=0.45,
                stub_uncertainty=0.85,
            )
            surrogate = get_surrogate(config)
            self.assertIsInstance(surrogate, SurrogateFacade)

    def test_facade_reload_raises_on_v1_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            v2_path = root / "nightly_v2.pkl"
            v1_path = root / "nightly.pkl"
            _write_v2_checkpoint(v2_path)
            _write_legacy_v1_checkpoint(v1_path)
            model = load_surrogate_checkpoint(v2_path)
            facade = build_surrogate_facade(model, uncertainty_fallback=0.85)
            with self.assertRaises(ValueError):
                facade.reload(v1_path)


if __name__ == "__main__":
    unittest.main()

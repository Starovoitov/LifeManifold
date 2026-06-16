"""Tests for train-time emitter one-hot feature augmentation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from worldspace.surrogate.emitter_features import (
    EMITTER_ONEHOT_DIM,
    augment_features_with_emitter,
    emitter_onehot_vector,
    training_feature_dim,
)
from worldspace.surrogate.feature_extractor import (
    FEATURE_SCHEMA_VERSION,
    feature_dim_for_schema,
)
from worldspace.surrogate.synthetic_buffer import write_synthetic_buffer
from worldspace.surrogate.training import load_buffer


class TestSurrogateEmitterFeatures(unittest.TestCase):
    def test_emitter_onehot_vector_known_labels(self) -> None:
        self.assertEqual(list(emitter_onehot_vector("random")), [1.0, 0.0, 0.0])
        self.assertEqual(list(emitter_onehot_vector("genetic")), [0.0, 1.0, 0.0])
        self.assertEqual(list(emitter_onehot_vector("llm")), [0.0, 0.0, 1.0])
        self.assertEqual(list(emitter_onehot_vector("unknown")), [0.0, 0.0, 0.0])

    def test_load_buffer_emitter_onehot_increases_feature_dim(self) -> None:
        base_dim = feature_dim_for_schema(FEATURE_SCHEMA_VERSION)
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            write_synthetic_buffer(buffer_path, n_samples=12, seed=4)
            features, _ = load_buffer(buffer_path, emitter_onehot=True)
            self.assertEqual(
                features.shape[1],
                training_feature_dim(base_dim=base_dim, emitter_onehot=True),
            )
            self.assertEqual(features.shape[1], base_dim + EMITTER_ONEHOT_DIM)

    def test_augment_features_with_emitter_matches_row_count(self) -> None:
        matrix = np.ones((3, 5), dtype=float)
        labels = np.asarray(["random", "genetic", "llm"], dtype=object)
        augmented = augment_features_with_emitter(matrix, labels)
        self.assertEqual(augmented.shape, (3, 8))
        self.assertEqual(float(augmented[0, 5]), 1.0)
        self.assertEqual(float(augmented[1, 6]), 1.0)
        self.assertEqual(float(augmented[2, 7]), 1.0)


if __name__ == "__main__":
    unittest.main()

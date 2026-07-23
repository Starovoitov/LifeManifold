"""Tests for maze surrogate buffers, checkpoints, and predictions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from worldspace.mazes.evaluation import evaluate_maze
from worldspace.mazes.features import FEATURE_NAMES, extract_features
from worldspace.mazes.genetics import random_maze
from worldspace.mazes.surrogate import (
    MazeSurrogate,
    buffer_row,
    load_buffer,
    save_checkpoint,
    train_checkpoint,
)


class TestMazeSurrogate(unittest.TestCase):
    def test_buffer_round_trip_and_checkpoint_prediction(self) -> None:
        rng = np.random.default_rng(4)
        specs = [random_maze(rng) for _ in range(80)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            buffer = root / "buffer.jsonl"
            with buffer.open("w", encoding="utf-8") as handle:
                for index, spec in enumerate(specs):
                    evaluation = evaluate_maze(spec)
                    handle.write(
                        json.dumps(
                            buffer_row(
                                spec,
                                evaluation,
                                design_seed=20000 + index,
                            )
                        )
                        + "\n"
                    )
            features, targets = load_buffer(buffer)
            self.assertEqual(features.shape, (80, len(FEATURE_NAMES)))
            checkpoint, report = train_checkpoint(
                features,
                targets,
                ensemble_size=2,
                max_iter=80,
            )
            path = root / "checkpoint.pkl"
            save_checkpoint(checkpoint, path)
            prediction = MazeSurrogate.load(path).predict(specs[0])
            self.assertGreaterEqual(prediction.fitness, 0.0)
            self.assertLessEqual(prediction.fitness, 1.0)
            self.assertGreaterEqual(prediction.uncertainty, 0.0)
            self.assertEqual(report["rows"], 80)

    def test_feature_schema_is_target_free_and_fixed(self) -> None:
        spec = random_maze(np.random.default_rng(9))
        self.assertEqual(extract_features(spec).shape, (len(FEATURE_NAMES),))


if __name__ == "__main__":
    unittest.main()

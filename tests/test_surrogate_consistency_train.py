"""Tests for optional consistency refinement during training."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec
from worldspace.surrogate.buffer import buffer_record, world_spec_dict_for_buffer
from worldspace.surrogate.feature_extractor import FEATURE_NAMES
from worldspace.surrogate.model import (
    TARGET_KEYS,
    SurrogateModel,
    consistency_mae_on_rows,
)
from worldspace.surrogate.training_runtime import train_from_buffer


def _sample_world_spec_dict() -> dict:
    spec = WorldSpec(
        birth=[1],
        survival=[2],
        noise=0.1,
        resource_regen=0.2,
        predation=0.05,
        cell_types=list(CANONICAL_CELL_TYPES),
        grid_size=30,
        steps=220,
        seed=0,
    )
    return world_spec_dict_for_buffer(spec)


def _write_synthetic_buffer(path: Path, rows: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    world_spec = _sample_world_spec_dict()
    with path.open("w", encoding="utf-8") as handle:
        for index in range(rows):
            features = np.linspace(0.1, 0.9, len(FEATURE_NAMES)) + index * 0.001
            targets = {key: float(0.2 + (index % 7) * 0.05) for key in TARGET_KEYS}
            record = buffer_record(
                features=features,
                targets=targets,
                emitter_type="random",
                world_spec=world_spec,
            )
            handle.write(__import__("json").dumps(record) + "\n")


class TestConsistencyTraining(unittest.TestCase):
    def test_consistency_weight_zero_skips_refinement(self) -> None:
        model = SurrogateModel(model_type="lightgbm", ensemble_size=2)
        features = np.tile(0.5, (8, len(FEATURE_NAMES))).reshape(8, -1)
        targets = {key: np.full(8, 0.3, dtype=float) for key in TARGET_KEYS}
        model.fit(features, targets)
        before = model.apply_consistency_refinement(features, targets, weight=0.0)
        after = consistency_mae_on_rows(model, features, targets)
        self.assertAlmostEqual(before, after)

    def test_micro_train_with_consistency_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            checkpoint_path = Path(tmpdir) / "model.pkl"
            summary_path = Path(tmpdir) / "model.summary.json"
            _write_synthetic_buffer(buffer_path, rows=120)
            result = train_from_buffer(
                buffer_path=buffer_path,
                checkpoint_path=checkpoint_path,
                summary_path=summary_path,
                micro=True,
                consistency_weight=0.1,
                require_quality_gate=False,
            )
            self.assertTrue(result.success, result.error_message)
            self.assertIsNotNone(result.consistency_mae_before)


if __name__ == "__main__":
    unittest.main()

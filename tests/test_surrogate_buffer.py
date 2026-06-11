from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from worldspace.illuminators.evaluation import apply_canonical_seed
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec
from worldspace.surrogate.buffer import (
    SurrogateBuffer,
    append_eval_to_buffer,
    buffer_record,
    world_spec_dict_for_buffer,
)
from worldspace.surrogate.feature_extractor import FEATURE_SCHEMA_VERSION, extract
from worldspace.surrogate.genome_features import FEATURE_DIM
from worldspace.surrogate.model import TARGET_KEYS


def _sample_targets() -> dict[str, float]:
    return {
        "stability": 0.1,
        "diversity": 0.2,
        "oscillation_score": 0.3,
        "topology_interface_index": 0.4,
        "topology_window_heterogeneity": 0.5,
        "final_density": 0.6,
        "early_extinction_prob": 0.0,
    }


def _sample_world_spec() -> WorldSpec:
    spec = WorldSpec(
        birth=[3],
        survival=[2, 3],
        noise=0.1,
        resource_regen=0.2,
        predation=0.05,
        cell_types=list(CANONICAL_CELL_TYPES),
        grid_size=30,
        steps=220,
        seed=0,
    )
    apply_canonical_seed(spec)
    return spec


def _sample_world_spec_dict() -> dict:
    return world_spec_dict_for_buffer(_sample_world_spec())


def _sample_features() -> np.ndarray:
    return extract(_sample_world_spec())


class SurrogateBufferTests(unittest.TestCase):
    def test_buffer_record_requires_all_targets(self) -> None:
        with self.assertRaises(ValueError):
            buffer_record(
                features=_sample_features(),
                targets={"stability": 0.1},
                emitter_type="random",
                world_spec=_sample_world_spec_dict(),
            )

    def test_buffer_record_requires_world_spec(self) -> None:
        with self.assertRaises(ValueError):
            buffer_record(
                features=_sample_features(),
                targets=_sample_targets(),
                emitter_type="random",
                world_spec={},
            )

    def test_buffer_record_roundtrip_world_spec_and_features(self) -> None:
        spec = _sample_world_spec()
        features = extract(spec)
        record = buffer_record(
            features=features,
            targets=_sample_targets(),
            emitter_type="random",
            world_spec=world_spec_dict_for_buffer(spec),
        )
        self.assertEqual(record["feature_schema_version"], FEATURE_SCHEMA_VERSION)
        self.assertEqual(len(record["features"]), FEATURE_DIM)
        restored = WorldSpec.from_json_dict(record["world_spec"])
        apply_canonical_seed(restored)
        np.testing.assert_allclose(extract(restored), features)

    def test_surrogate_buffer_flushes_batched_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer.jsonl"
            buf = SurrogateBuffer(path=path, flush_every=2)
            sample_targets = _sample_targets()
            world_spec = _sample_world_spec_dict()
            buf.append(
                features=_sample_features(),
                targets=sample_targets,
                emitter_type="random",
                world_spec=world_spec,
            )
            self.assertFalse(path.exists())
            buf.append(
                features=_sample_features(),
                targets=sample_targets,
                emitter_type="genetic",
                world_spec=world_spec,
            )
            self.assertTrue(path.exists())
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            row = json.loads(lines[0])
            self.assertEqual(row["emitter_type"], "random")
            self.assertEqual(len(row["features"]), FEATURE_DIM)
            self.assertIn("world_spec", row)

    def test_append_eval_to_buffer_writes_world_spec(self) -> None:
        spec = _sample_world_spec()
        result = Mock()
        result.world_spec = spec
        result.measures = {"stability": 0.4, "diversity": 0.5}
        result.metrics = Mock(
            density_mean=0.6,
            oscillation_score=0.2,
            topology_interface_index=0.3,
            topology_window_heterogeneity=0.1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer.jsonl"
            buf = SurrogateBuffer(path=path, flush_every=1)
            append_eval_to_buffer(buf, result, emitter_type="genetic")
            row = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(row["feature_schema_version"], FEATURE_SCHEMA_VERSION)
            self.assertIn("world_spec", row)
            self.assertEqual(len(row["features"]), FEATURE_DIM)
            for key in TARGET_KEYS:
                self.assertIn(key, row["targets"])
            self.assertEqual(row["metadata"]["source"], "live_eval")


if __name__ == "__main__":
    unittest.main()

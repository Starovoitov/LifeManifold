from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from worldspace.surrogate.buffer import SurrogateBuffer, buffer_record


class SurrogateBufferTests(unittest.TestCase):
    def test_buffer_record_requires_all_targets(self) -> None:
        with self.assertRaises(ValueError):
            buffer_record(
                features=np.array([1.0, 2.0]),
                targets={"stability": 0.1},
                emitter_type="random",
            )

    def test_surrogate_buffer_flushes_batched_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer.jsonl"
            buf = SurrogateBuffer(path=path, flush_every=2)
            sample_targets = {
                "stability": 0.1,
                "diversity": 0.2,
                "oscillation_score": 0.3,
                "topology_interface_index": 0.4,
                "topology_window_heterogeneity": 0.5,
                "final_density": 0.6,
                "early_extinction_prob": 0.0,
            }
            buf.append(
                features=np.array([1.0, 2.0]),
                targets=sample_targets,
                emitter_type="random",
            )
            self.assertFalse(path.exists())
            buf.append(
                features=np.array([3.0, 4.0]),
                targets=sample_targets,
                emitter_type="genetic",
            )
            self.assertTrue(path.exists())
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            row = json.loads(lines[0])
            self.assertEqual(row["emitter_type"], "random")
            self.assertEqual(len(row["features"]), 2)


if __name__ == "__main__":
    unittest.main()

"""Tests for surrogate buffer filter CLI and core helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldspace.illuminators.evaluation import apply_canonical_seed
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec
from worldspace.surrogate.buffer import buffer_record
from worldspace.surrogate.buffer_filter import filter_buffer_path
from worldspace.surrogate.feature_extractor import extract
from worldspace.surrogate.synthetic_buffer import write_synthetic_buffer
from worldspace.surrogate.training import load_buffer


def _record_with_source(
    *,
    emitter_type: str,
    source: str,
    spec: WorldSpec,
) -> dict[str, object]:
    apply_canonical_seed(spec)
    return buffer_record(
        features=extract(spec),
        targets={
            "stability": 0.5,
            "diversity": 0.4,
            "oscillation_score": 0.3,
            "topology_interface_index": 0.2,
            "topology_window_heterogeneity": 0.25,
            "final_density": 0.35,
            "early_extinction_prob": 0.1,
            "fitness": 0.42,
        },
        emitter_type=emitter_type,
        world_spec=spec.to_json_dict(),
        metadata={"source": source},
    )


class TestFilterSurrogateBuffer(unittest.TestCase):
    def test_dedupe_keeps_first_row_per_world_spec(self) -> None:
        spec = WorldSpec(
            birth=[1, 2],
            survival=[2, 3],
            noise=0.02,
            resource_regen=0.05,
            predation=0.1,
            cell_types=list(CANONICAL_CELL_TYPES),
            grid_size=20,
            steps=100,
            seed=0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "buffer.jsonl"
            rows = [
                _record_with_source(
                    emitter_type="genetic",
                    source="live_eval",
                    spec=spec,
                ),
                _record_with_source(
                    emitter_type="random",
                    source="archive_backfill",
                    spec=spec,
                ),
            ]
            source.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                encoding="utf-8",
            )
            output = root / "filtered.jsonl"
            stats = filter_buffer_path(
                source,
                output,
                dedupe=True,
                live_only=False,
                drop_backfill=False,
            )
            self.assertEqual(stats["rows_read"], 2)
            self.assertEqual(stats["rows_written"], 1)
            self.assertEqual(stats["duplicates_dropped"], 1)
            features, targets = load_buffer(output)
            self.assertEqual(features.shape[0], 1)
            self.assertEqual(len(targets["stability"]), 1)

    def test_live_only_and_drop_backfill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "buffer.jsonl"
            write_synthetic_buffer(source, n_samples=4, seed=1)
            rows: list[dict[str, object]] = []
            for index, line in enumerate(
                source.read_text(encoding="utf-8").splitlines()
            ):
                row = json.loads(line)
                row["metadata"] = {
                    "source": "live_eval" if index % 2 == 0 else "archive_backfill"
                }
                rows.append(row)
            source.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                encoding="utf-8",
            )
            output = root / "live.jsonl"
            stats = filter_buffer_path(
                source,
                output,
                dedupe=False,
                live_only=True,
                drop_backfill=False,
            )
            self.assertEqual(stats["rows_written"], 2)
            features, _ = load_buffer(output)
            self.assertEqual(features.shape[0], 2)


if __name__ == "__main__":
    unittest.main()

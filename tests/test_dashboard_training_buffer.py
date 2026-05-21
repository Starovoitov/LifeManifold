"""Unit tests for dashboard surrogate training buffer loader."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from worldspace.surrogate.buffer import buffer_record
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


def _write_buffer(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestDashboardTrainingBuffer(unittest.TestCase):
    def test_flatten_buffer_record_has_all_target_columns(self) -> None:
        from dashboard.components.training_buffer_loader import flatten_buffer_record

        record = buffer_record(
            features=np.array([1.0, 2.0, 3.0]),
            targets=_sample_targets(),
            emitter_type="random",
            feature_schema_version="1.0",
        )
        row = flatten_buffer_record(record)
        for key in TARGET_KEYS:
            self.assertIn(f"target_{key}", row)

    def test_load_buffer_counts_emitter_and_schema(self) -> None:
        from dashboard.components.training_buffer_loader import (
            buffer_summary_counts,
            read_buffer_jsonl,
        )
        from dashboard.utils.config import load_config

        rows = [
            buffer_record(
                features=np.array([1.0, 2.0]),
                targets=_sample_targets(),
                emitter_type="random",
            ),
            buffer_record(
                features=np.array([3.0, 4.0]),
                targets=_sample_targets(),
                emitter_type="genetic",
            ),
            buffer_record(
                features=np.array([5.0, 6.0]),
                targets=_sample_targets(),
                emitter_type="random",
                feature_schema_version="1.0",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer.jsonl"
            _write_buffer(path, rows)
            frame, raw, line_count, invalid = read_buffer_jsonl(path, load_config())
        self.assertEqual(len(frame), 3)
        self.assertEqual(len(raw), 3)
        self.assertEqual(line_count, 3)
        self.assertEqual(invalid, 0)
        counts = buffer_summary_counts(frame)
        emitter_counts = counts["emitter_type"]
        schema_counts = counts["feature_schema_version"]
        self.assertEqual(int(emitter_counts.loc["random"]), 2)
        self.assertEqual(int(emitter_counts.loc["genetic"]), 1)
        self.assertEqual(int(schema_counts.loc["1.0"]), 3)

    def test_skip_malformed_lines(self) -> None:
        from dashboard.components.training_buffer_loader import read_buffer_jsonl
        from dashboard.utils.config import load_config

        valid = buffer_record(
            features=np.array([1.0]),
            targets=_sample_targets(),
            emitter_type="llm",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer.jsonl"
            path.write_text(
                json.dumps(valid)
                + "\n{broken\n"
                + json.dumps({"no": "targets"})
                + "\n",
                encoding="utf-8",
            )
            frame, raw, line_count, invalid = read_buffer_jsonl(path, load_config())
        self.assertEqual(len(frame), 1)
        self.assertEqual(line_count, 3)
        self.assertEqual(invalid, 2)

    def test_slice_for_display_respects_max_rows(self) -> None:
        from dashboard.components.training_buffer_loader import slice_for_display

        frame = pd.DataFrame({"emitter_type": [f"e{i}" for i in range(10)]})
        page = slice_for_display(frame, page=0, page_size=1000, max_rows=4)
        self.assertEqual(len(page), 4)
        page_two = slice_for_display(frame, page=1, page_size=4, max_rows=500)
        self.assertEqual(len(page_two), 4)
        self.assertEqual(page_two.iloc[0]["emitter_type"], "e4")

    def test_export_subset_roundtrip(self) -> None:
        from dashboard.components.training_buffer_loader import (
            export_subset_jsonl,
            read_buffer_jsonl,
            try_flatten_buffer_record,
        )
        from dashboard.utils.config import load_config

        rows = [
            buffer_record(
                features=np.array([1.0, 2.0]),
                targets=_sample_targets(),
                emitter_type="random",
            ),
            buffer_record(
                features=np.array([3.0, 4.0]),
                targets=_sample_targets(),
                emitter_type="genetic",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer.jsonl"
            _write_buffer(path, rows)
            exported = export_subset_jsonl(rows)
            out_path = Path(tmpdir) / "export.jsonl"
            out_path.write_text(exported, encoding="utf-8")
            frame, raw, _, _ = read_buffer_jsonl(out_path, load_config())
        self.assertEqual(len(frame), 2)
        self.assertEqual(len(raw), 2)
        for record in raw:
            self.assertIsNotNone(try_flatten_buffer_record(record))

    def test_apply_buffer_filters_subset(self) -> None:
        from dashboard.components.training_buffer_loader import (
            BufferBundle,
            apply_buffer_filters,
            read_buffer_jsonl,
        )
        from dashboard.utils.config import load_config

        rows = [
            buffer_record(
                features=np.array([1.0]),
                targets=_sample_targets(),
                emitter_type="random",
            ),
            buffer_record(
                features=np.array([2.0]),
                targets=_sample_targets(),
                emitter_type="genetic",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer.jsonl"
            _write_buffer(path, rows)
            frame, raw, line_count, invalid = read_buffer_jsonl(path, load_config())
        bundle = BufferBundle(
            records=frame,
            raw_records=raw,
            line_count_raw=line_count,
            invalid_line_count=invalid,
            large_buffer_mode=False,
            source_path=str(path),
        )
        filtered, filtered_raw = apply_buffer_filters(
            bundle,
            emitter_types=["genetic"],
            schema_versions=["1.0"],
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(len(filtered_raw), 1)
        self.assertEqual(filtered_raw[0]["emitter_type"], "genetic")

    def test_empty_multiselect_yields_no_rows(self) -> None:
        from dashboard.components.training_buffer_loader import (
            BufferBundle,
            apply_buffer_filters,
            read_buffer_jsonl,
        )
        from dashboard.utils.config import load_config

        rows = [
            buffer_record(
                features=np.array([1.0]),
                targets=_sample_targets(),
                emitter_type="random",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer.jsonl"
            _write_buffer(path, rows)
            frame, raw, line_count, invalid = read_buffer_jsonl(path, load_config())
        bundle = BufferBundle(
            records=frame,
            raw_records=raw,
            line_count_raw=line_count,
            invalid_line_count=invalid,
            large_buffer_mode=False,
            source_path=str(path),
        )
        filtered, filtered_raw = apply_buffer_filters(
            bundle,
            emitter_types=[],
            schema_versions=["1.0"],
        )
        self.assertTrue(filtered.empty)
        self.assertEqual(filtered_raw, [])

    def test_effective_table_page_size_clamps_invalid_config(self) -> None:
        from dashboard.components.training_buffer_view import effective_table_page_size

        self.assertEqual(effective_table_page_size(0), 1)
        self.assertEqual(effective_table_page_size(-10), 1)
        self.assertEqual(effective_table_page_size(500), 500)
        self.assertEqual(effective_table_page_size(1000), 500)

    def test_slice_uses_effective_page_size_not_raw_config(self) -> None:
        from dashboard.components.training_buffer_loader import slice_for_display
        from dashboard.components.training_buffer_view import effective_table_page_size

        frame = pd.DataFrame({"emitter_type": [f"e{i}" for i in range(8)]})
        page_size = effective_table_page_size(1000)
        page = slice_for_display(frame, page=0, page_size=page_size, max_rows=page_size)
        self.assertEqual(len(page), 8)
        self.assertEqual(page_size, 500)

    def test_resolve_surrogate_buffer_path(self) -> None:
        from dashboard.utils.config import load_config, resolve_surrogate_buffer_path

        path = resolve_surrogate_buffer_path(load_config())
        self.assertTrue(path.name.endswith(".jsonl"))


if __name__ == "__main__":
    unittest.main()

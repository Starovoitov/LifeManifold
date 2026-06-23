"""Unit tests for acquisition run comparison script."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_acquisition_runs import (
    _infer_grid_resolution_from_bin_cell,
    _meta_from_nightly_summary,
    _summarize_run,
    resolve_run_archive_meta,
)


class TestResolveRunArchiveMeta(unittest.TestCase):
    def test_nightly_summary_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "nightly_run_summary.json").write_text(
                json.dumps(
                    {
                        "archive_type": "grid",
                        "n_cells": 100,
                        "grid_resolution": 10,
                    }
                ),
                encoding="utf-8",
            )
            meta = resolve_run_archive_meta(root, grid_resolution=5)
        self.assertEqual(meta["archive_type"], "grid")
        self.assertEqual(meta["n_cells"], 100)

    def test_nightly_summary_cvt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "nightly_run_summary.json").write_text(
                json.dumps(
                    {
                        "archive_type": "cvt",
                        "n_cells": 25,
                        "grid_resolution": 5,
                    }
                ),
                encoding="utf-8",
            )
            meta = resolve_run_archive_meta(root, grid_resolution=10)
        self.assertEqual(meta["archive_type"], "cvt")
        self.assertEqual(meta["n_cells"], 25)

    def test_nightly_summary_invalid_grid_resolution_with_n_cells_returns_none(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "nightly_run_summary.json").write_text(
                json.dumps(
                    {
                        "archive_type": "grid",
                        "n_cells": 100,
                        "grid_resolution": "invalid",
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNone(_meta_from_nightly_summary(root))
            meta = resolve_run_archive_meta(root, grid_resolution=10)
        self.assertEqual(meta["archive_type"], "grid")
        self.assertEqual(meta["n_cells"], 100)

    def test_legacy_fallback_uses_grid_resolution_squared(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = resolve_run_archive_meta(Path(tmpdir), grid_resolution=10)
        self.assertEqual(meta["archive_type"], "grid")
        self.assertEqual(meta["n_cells"], 100)

    def test_n_cells_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = resolve_run_archive_meta(
                Path(tmpdir),
                grid_resolution=10,
                n_cells_override=9,
            )
        self.assertEqual(meta["n_cells"], 9)

    def test_jsonl_cvt_with_centroids_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "map_elites_archive.jsonl").write_text(
                json.dumps(
                    {
                        "schema_version": "1.3",
                        "archive_type": "cvt",
                        "cell_id": 0,
                        "fitness": 0.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "cvt_centroids.json").write_text(
                json.dumps({"n": 9, "centroids": [[0.1, 0.2]] * 9}),
                encoding="utf-8",
            )
            meta = resolve_run_archive_meta(root, grid_resolution=10)
        self.assertEqual(meta["archive_type"], "cvt")
        self.assertEqual(meta["n_cells"], 9)

    def test_jsonl_skips_invalid_json_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            valid = json.dumps(
                {
                    "schema_version": "1.3",
                    "archive_type": "cvt",
                    "cell_id": 0,
                    "fitness": 0.5,
                }
            )
            (root / "map_elites_archive.jsonl").write_text(
                "not valid json\n" + valid + "\n",
                encoding="utf-8",
            )
            (root / "cvt_centroids.json").write_text(
                json.dumps({"n": 4, "centroids": [[0.1, 0.2]] * 4}),
                encoding="utf-8",
            )
            meta = resolve_run_archive_meta(root, grid_resolution=10)
        self.assertEqual(meta["archive_type"], "cvt")
        self.assertEqual(meta["n_cells"], 4)

    def test_jsonl_skips_non_dict_json_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            valid = json.dumps(
                {
                    "schema_version": "1.3",
                    "archive_type": "cvt",
                    "cell_id": 1,
                    "fitness": 0.6,
                }
            )
            (root / "map_elites_archive.jsonl").write_text(
                "[1, 2, 3]\n" + valid + "\n",
                encoding="utf-8",
            )
            (root / "cvt_centroids.json").write_text(
                json.dumps({"n": 6, "centroids": [[0.2, 0.3]] * 6}),
                encoding="utf-8",
            )
            meta = resolve_run_archive_meta(root, grid_resolution=10)
        self.assertEqual(meta["archive_type"], "cvt")
        self.assertEqual(meta["n_cells"], 6)

    def test_grid_jsonl_rejects_mismatched_cell_id_and_bin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "map_elites_archive.jsonl").write_text(
                json.dumps(
                    {
                        "schema_version": "1.3",
                        "archive_type": "grid",
                        "cell_id": 23,
                        "bin": [5, 0],
                        "fitness": 0.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            meta = resolve_run_archive_meta(root, grid_resolution=10)
        self.assertEqual(meta["archive_type"], "grid")
        self.assertEqual(meta["n_cells"], 100)

    def test_grid_jsonl_infers_resolution_when_bin_matches_cell_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "map_elites_archive.jsonl").write_text(
                json.dumps(
                    {
                        "schema_version": "1.3",
                        "archive_type": "grid",
                        "cell_id": 12,
                        "bin": [1, 2],
                        "fitness": 0.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            meta = resolve_run_archive_meta(root, grid_resolution=5)
        self.assertEqual(meta["grid_resolution"], 10)
        self.assertEqual(meta["n_cells"], 100)


class TestInferGridResolution(unittest.TestCase):
    def test_mismatched_bin_returns_none(self) -> None:
        self.assertIsNone(_infer_grid_resolution_from_bin_cell(5, 0, 23))

    def test_consistent_bin_returns_resolution(self) -> None:
        self.assertEqual(_infer_grid_resolution_from_bin_cell(1, 2, 12), 10)

    def test_i_zero_is_ambiguous_returns_none(self) -> None:
        self.assertIsNone(_infer_grid_resolution_from_bin_cell(0, 5, 5))


class TestSummarizeRun(unittest.TestCase):
    def test_filled_pct_uses_n_cells_from_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "nightly_run_summary.json").write_text(
                json.dumps(
                    {
                        "archive_type": "cvt",
                        "n_cells": 25,
                        "grid_resolution": 5,
                    }
                ),
                encoding="utf-8",
            )
            archive_lines = "\n".join(json.dumps({"fitness": 0.5}) for _ in range(5))
            (root / "map_elites_archive.jsonl").write_text(
                archive_lines + "\n",
                encoding="utf-8",
            )
            summary = _summarize_run(root, grid_resolution=10)
        self.assertEqual(summary["archive_type"], "cvt")
        self.assertEqual(summary["n_cells"], 25)
        self.assertAlmostEqual(summary["filled_cells_pct"], 20.0)

    def test_grid_legacy_matches_resolution_squared(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "map_elites_archive.jsonl").write_text(
                json.dumps({"fitness": 0.4}) + "\n",
                encoding="utf-8",
            )
            summary = _summarize_run(root, grid_resolution=10)
        self.assertEqual(summary["n_cells"], 100)
        self.assertAlmostEqual(summary["filled_cells_pct"], 1.0)


if __name__ == "__main__":
    unittest.main()

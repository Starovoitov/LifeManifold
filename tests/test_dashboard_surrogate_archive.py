"""Unit tests for SurrogateArchive dashboard loader and acquisition metrics."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "surrogate_archive_smoke.jsonl"
)


def _sample_record(**overrides: object) -> dict:
    base = {
        "schema_version": "1.0",
        "run_id": "run-a",
        "iteration": 0,
        "candidate_id": 0,
        "emitter_type": "random",
        "target_bin": [2, 3],
        "world_spec_hash": "abc",
        "prediction": {
            "fitness": 0.2,
            "uncertainty": 0.3,
            "components": {},
            "measures": {},
        },
        "decision": "skip",
        "decision_reason": "below_fitness_threshold",
        "acquisition_mode": "filter",
        "eval_outcome": None,
    }
    base.update(overrides)
    return base


class TestSurrogateArchiveLoader(unittest.TestCase):
    def test_flatten_archive_record_target_bin_columns(self) -> None:
        from dashboard.components.surrogate_archive_loader import flatten_archive_record

        row = flatten_archive_record(_sample_record())
        self.assertEqual(row["target_bin_i"], 2)
        self.assertEqual(row["target_bin_j"], 3)
        self.assertEqual(row["target_bin_label"], "2,3")
        self.assertFalse(row["has_eval"])

    def test_load_smoke_fixture(self) -> None:
        from dashboard.components.surrogate_archive_loader import (
            load_surrogate_archive,
            read_surrogate_archive_jsonl,
        )
        from dashboard.utils.config import load_config

        frame = load_surrogate_archive(_FIXTURE)
        self.assertEqual(len(frame), 6)
        self.assertNotIn("target_bin", frame.columns)

        _, _, line_count, invalid = read_surrogate_archive_jsonl(
            _FIXTURE,
            load_config(),
        )
        self.assertEqual(line_count, 6)
        self.assertEqual(invalid, 0)

    def test_invalid_line_counted(self) -> None:
        from dashboard.components.surrogate_archive_loader import (
            read_surrogate_archive_jsonl,
        )
        from dashboard.utils.config import load_config

        path = _FIXTURE.parent / "_tmp_bad_archive.jsonl"
        path.write_text(
            json.dumps(_sample_record()) + "\n{broken\n",
            encoding="utf-8",
        )
        try:
            frame, _, line_count, invalid = read_surrogate_archive_jsonl(
                path,
                load_config(),
            )
            self.assertEqual(len(frame), 1)
            self.assertEqual(line_count, 2)
            self.assertEqual(invalid, 1)
        finally:
            path.unlink(missing_ok=True)

    def test_resolve_surrogate_archive_path(self) -> None:
        from dashboard.utils.config import load_config, resolve_surrogate_archive_path

        path = resolve_surrogate_archive_path(load_config())
        self.assertTrue(path.name.endswith(".jsonl"))
        self.assertIn("surrogate_archive", str(path))

    def test_resolve_surrogate_archive_path_prefers_co_located(self) -> None:
        import tempfile

        from dashboard.utils.config import (
            load_config,
            resolve_surrogate_archive_path,
            surrogate_archive_path_for_map_elites_archive,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "acq_shadow"
            run_dir.mkdir()
            map_archive = run_dir / "map_elites_archive.jsonl"
            map_archive.write_text("{}\n", encoding="utf-8")
            log_path = run_dir / "surrogate_archive.jsonl"
            log_path.write_text(
                json.dumps(_sample_record()) + "\n",
                encoding="utf-8",
            )
            resolved = resolve_surrogate_archive_path(
                load_config(),
                archive_path=map_archive,
            )
            self.assertEqual(
                resolved,
                surrogate_archive_path_for_map_elites_archive(map_archive),
            )
            self.assertTrue(resolved.is_file())

    def test_resolve_co_located_without_configured_path(self) -> None:
        import tempfile

        from dashboard.utils.config import (
            resolve_surrogate_archive_path,
            surrogate_archive_path_for_map_elites_archive,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "acq_shadow"
            run_dir.mkdir()
            map_archive = run_dir / "map_elites_archive.jsonl"
            map_archive.write_text("{}\n", encoding="utf-8")
            log_path = run_dir / "surrogate_archive.jsonl"
            log_path.write_text(
                json.dumps(_sample_record()) + "\n",
                encoding="utf-8",
            )
            cfg = {"paths": {}}
            resolved = resolve_surrogate_archive_path(cfg, archive_path=map_archive)
            self.assertEqual(
                resolved,
                surrogate_archive_path_for_map_elites_archive(map_archive),
            )
            self.assertTrue(resolved.is_file())

    def test_resolve_raises_when_config_and_archive_missing(self) -> None:
        from dashboard.utils.config import resolve_surrogate_archive_path

        with self.assertRaises(KeyError):
            resolve_surrogate_archive_path({"paths": {}})


class TestAcquisitionMetrics(unittest.TestCase):
    def test_acquisition_kpis_smoke_fixture(self) -> None:
        from dashboard.components.surrogate_acquisition_view import acquisition_kpis
        from dashboard.components.surrogate_archive_loader import load_surrogate_archive

        frame = load_surrogate_archive(_FIXTURE)
        kpis = acquisition_kpis(frame)
        self.assertEqual(kpis["total"], 6)
        self.assertEqual(kpis["skip_count"], 3)
        self.assertAlmostEqual(kpis["skip_rate_pct"], 50.0)
        self.assertEqual(kpis["shadow_would_skip"], 1)
        self.assertEqual(kpis["filter_actual_skip"], 2)

    def test_skips_by_iteration_cumulative(self) -> None:
        from dashboard.components.surrogate_acquisition_view import skips_by_iteration
        from dashboard.components.surrogate_archive_loader import load_surrogate_archive

        stats = skips_by_iteration(load_surrogate_archive(_FIXTURE))
        self.assertEqual(list(stats["iteration"]), [0, 1, 2])
        self.assertEqual(list(stats["skips"]), [1, 1, 1])
        self.assertEqual(list(stats["cumulative_skips"]), [1, 2, 3])

    def test_apply_filters_empty_multiselect(self) -> None:
        from dashboard.components.surrogate_archive_loader import (
            ArchiveLogBundle,
            apply_archive_log_filters,
            load_surrogate_archive,
        )

        frame = load_surrogate_archive(_FIXTURE)
        bundle = ArchiveLogBundle(
            records=frame,
            raw_records=[],
            line_count_raw=6,
            invalid_line_count=0,
            source_path=str(_FIXTURE),
        )
        filtered = apply_archive_log_filters(
            bundle,
            decisions=[],
            acquisition_modes=["filter"],
            emitter_types=["random"],
            iteration_range=None,
        )
        self.assertTrue(filtered.empty)

    def test_charts_build_without_error(self) -> None:
        from dashboard.components.surrogate_acquisition_view import (
            plot_cumulative_skips,
            plot_skips_per_iteration,
            skips_by_iteration,
        )
        from dashboard.components.surrogate_archive_loader import load_surrogate_archive

        stats = skips_by_iteration(load_surrogate_archive(_FIXTURE))
        bar = plot_skips_per_iteration(stats)
        line = plot_cumulative_skips(stats)
        self.assertEqual(bar.data[0].x.dtype, stats["iteration"].dtype)
        self.assertEqual(len(line.data[0].x), 3)


if __name__ == "__main__":
    unittest.main()

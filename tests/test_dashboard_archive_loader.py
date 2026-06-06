"""Unit tests for dashboard archive loading and pivots."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_ARCHIVE = (
    _REPO_ROOT / "artifacts" / "map_elites_smoke" / "map_elites_archive.jsonl"
)


class TestDashboardArchiveLoader(unittest.TestCase):
    def test_load_skips_malformed_jsonl_lines(self) -> None:
        from dashboard.components.archive_loader import load_archive_bundle
        from dashboard.utils.config import load_config

        record = json.loads(_SMOKE_ARCHIVE.read_text(encoding="utf-8").splitlines()[0])
        bad = {"schema_version": "1.2", "bin": [0, 0]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed.jsonl"
            path.write_text(
                json.dumps(record) + "\n" + json.dumps(bad) + "\n",
                encoding="utf-8",
            )
            cfg = load_config()
            bundle = load_archive_bundle(path, path.stat().st_mtime, cfg)
            self.assertGreaterEqual(len(bundle.collapsed), 1)

    def test_smoke_bundle_pivot_shape(self) -> None:
        from dashboard.components.archive_loader import load_archive_bundle
        from dashboard.utils.config import load_config

        cfg = load_config()
        bundle = load_archive_bundle(
            _SMOKE_ARCHIVE, _SMOKE_ARCHIVE.stat().st_mtime, cfg
        )
        self.assertEqual(bundle.pivots["fitness"].shape, (50, 50))
        self.assertGreater(len(bundle.collapsed), 0)
        self.assertFalse(bundle.large_archive_mode)

    def test_large_archive_mode_with_low_threshold(self) -> None:
        from dashboard.components.archive_loader import load_archive_bundle
        from dashboard.utils.config import load_config

        cfg = load_config()
        performance = dict(cfg.get("performance") or {})
        performance["large_archive_line_threshold"] = 0
        cfg = {**cfg, "performance": performance}
        bundle = load_archive_bundle(_SMOKE_ARCHIVE, 0.0, cfg)
        self.assertTrue(bundle.large_archive_mode)

    def test_collapse_keeps_best_fitness_per_bin(self) -> None:
        from dashboard.components.archive_loader import collapse_dataframe
        from dashboard.utils.data_processing import flatten_archive_record

        rows = []
        for line in _SMOKE_ARCHIVE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(flatten_archive_record(json.loads(line)))
        frame = __import__("pandas").DataFrame(rows)
        collapsed = collapse_dataframe(frame)
        self.assertEqual(
            len(collapsed),
            collapsed.groupby(["bin_x", "bin_y"]).ngroups,
        )

    def test_mixed_prompt_version_jsonl_loads_with_polars(self) -> None:
        from worldspace.illuminators.archive import (
            ArchiveElite,
            elite_to_archive_record,
            new_elite_metadata,
            normalize_archive_record_metadata,
        )
        from worldspace.specs.spec import WorldSpec

        from dashboard.components.archive_loader import load_archive_bundle
        from dashboard.utils.config import load_config

        spec = WorldSpec(
            birth=[1],
            survival=[2],
            noise=0.0,
            resource_regen=0.0,
            predation=0.0,
            cell_types=["life", "food"],
            grid_size=50,
            steps=200,
            seed=1,
        )
        record = elite_to_archive_record(
            ArchiveElite(
                bin=(0, 0),
                fitness=0.5,
                world_spec=spec,
                measures={"stability": 0.5, "diversity": 0.5},
                metadata=new_elite_metadata(
                    generated_by="random",
                    emitter_type="random",
                    elite_id="mixed-test",
                    timestamp="2026-01-01T00:00:00+00:00",
                ),
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed_prompt_version.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for index in range(120):
                    copy = json.loads(json.dumps(record))
                    copy["bin"] = [index % 50, (index // 50) % 50]
                    copy["metadata"]["prompt_version"] = (
                        "" if index % 3 else "llm-prompt-hash"
                    )
                    handle.write(
                        json.dumps(normalize_archive_record_metadata(copy)) + "\n"
                    )
            cfg = load_config()
            bundle = load_archive_bundle(path, path.stat().st_mtime, cfg)
            self.assertEqual(bundle.line_count_raw, 120)
            self.assertGreater(len(bundle.collapsed), 0)

    def test_null_then_string_prompt_version_falls_back_from_polars(self) -> None:
        from worldspace.illuminators.archive import (
            ArchiveElite,
            elite_to_archive_record,
            new_elite_metadata,
        )
        from worldspace.specs.spec import WorldSpec

        from dashboard.components.archive_loader import load_archive_bundle
        from dashboard.utils.config import load_config

        spec = WorldSpec(
            birth=[1],
            survival=[2],
            noise=0.0,
            resource_regen=0.0,
            predation=0.0,
            cell_types=["life", "food"],
            grid_size=50,
            steps=200,
            seed=1,
        )
        record = elite_to_archive_record(
            ArchiveElite(
                bin=(0, 0),
                fitness=0.5,
                world_spec=spec,
                measures={"stability": 0.5, "diversity": 0.5},
                metadata=new_elite_metadata(
                    generated_by="random",
                    emitter_type="random",
                    elite_id="legacy-null-pv",
                    timestamp="2026-01-01T00:00:00+00:00",
                ),
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy_prompt_version.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for index in range(160):
                    copy = json.loads(json.dumps(record))
                    copy["bin"] = [index % 50, (index // 50) % 50]
                    copy["metadata"]["prompt_version"] = (
                        None if index < 120 else "llm-prompt-hash"
                    )
                    handle.write(json.dumps(copy) + "\n")
            cfg = load_config()
            performance = dict(cfg.get("performance") or {})
            performance["prefer_polars"] = True
            cfg = {**cfg, "performance": performance}
            bundle = load_archive_bundle(path, path.stat().st_mtime, cfg)
            self.assertEqual(bundle.line_count_raw, 160)
            self.assertGreater(len(bundle.collapsed), 0)

    def test_synthetic_jsonl_triggers_large_mode(self) -> None:
        from dashboard.components.archive_loader import load_archive_bundle
        from dashboard.utils.config import load_config

        record = json.loads(_SMOKE_ARCHIVE.read_text(encoding="utf-8").splitlines()[0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for index in range(5001):
                    copy = json.loads(json.dumps(record))
                    copy["bin"] = [index % 50, (index // 50) % 50]
                    copy["fitness"] = float(index % 100) / 100.0
                    handle.write(json.dumps(copy) + "\n")
            cfg = load_config()
            bundle = load_archive_bundle(path, path.stat().st_mtime, cfg)
            self.assertTrue(bundle.large_archive_mode)
            self.assertEqual(bundle.line_count_raw, 5001)


if __name__ == "__main__":
    unittest.main()

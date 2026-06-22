"""Unit tests for dashboard archive loading and pivots."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


from worldspace.illuminators.archive import (
    ARCHIVE_SCHEMA_VERSION_V1_3,
    ArchiveElite,
    elite_to_archive_record,
    new_elite_metadata,
)
from worldspace.illuminators.cvt import save_centroids
from worldspace.specs.spec import WorldSpec

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_ARCHIVE = (
    _REPO_ROOT / "artifacts" / "map_elites_smoke" / "map_elites_archive.jsonl"
)


def _cvt_fixture_dir(tmp: str, *, n_centroids: int = 9) -> Path:
    from worldspace.illuminators.cvt import generate_centroids

    root = Path(tmp)
    centroids = generate_centroids(n_centroids, seed=0, lloyd_iterations=5)
    save_centroids(root / "cvt_centroids.json", centroids)
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
    archive_path = root / "map_elites_archive.jsonl"
    with archive_path.open("w", encoding="utf-8") as handle:
        for cell_id in (0, 0, 3, 5):
            record = elite_to_archive_record(
                ArchiveElite(
                    bin=(cell_id, 0),
                    fitness=0.3 + 0.1 * cell_id,
                    world_spec=spec,
                    measures={"stability": 0.4, "diversity": 0.6},
                    metadata=new_elite_metadata(
                        generated_by="random",
                        emitter_type="random",
                        elite_id=f"cvt-{cell_id}",
                        timestamp="2026-01-01T00:00:00+00:00",
                    ),
                ),
                archive_type="cvt",
                schema_version=ARCHIVE_SCHEMA_VERSION_V1_3,
            )
            handle.write(json.dumps(record) + "\n")
    return archive_path


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

    def test_detect_grid_on_smoke_archive(self) -> None:
        from dashboard.components.archive_loader import detect_archive_type_from_jsonl

        self.assertEqual(detect_archive_type_from_jsonl(_SMOKE_ARCHIVE), "grid")

    def test_load_cvt_bundle_has_centroid_columns(self) -> None:
        from dashboard.components.archive_loader import load_archive_bundle
        from dashboard.utils.config import load_config

        with tempfile.TemporaryDirectory() as tmp:
            path = _cvt_fixture_dir(tmp)
            cfg = load_config()
            bundle = load_archive_bundle(path, path.stat().st_mtime, cfg)
        self.assertEqual(bundle.archive_type, "cvt")
        self.assertEqual(bundle.n_cells, 9)
        self.assertFalse(bundle.centroids_missing)
        self.assertIsNotNone(bundle.centroids)
        self.assertIn("centroid_s", bundle.collapsed.columns)
        self.assertIn("centroid_d", bundle.collapsed.columns)
        self.assertEqual(len(bundle.collapsed), 3)

    def test_cvt_missing_centroids_sets_flag(self) -> None:
        from dashboard.components.archive_loader import load_archive_bundle
        from dashboard.utils.config import load_config

        with tempfile.TemporaryDirectory() as tmp:
            path = _cvt_fixture_dir(tmp)
            (Path(tmp) / "cvt_centroids.json").unlink()
            cfg = load_config()
            bundle = load_archive_bundle(path, path.stat().st_mtime, cfg)
        self.assertTrue(bundle.centroids_missing)
        self.assertIsNone(bundle.centroids)

    def test_worldspace_fallback_marks_cvt_archive_type(self) -> None:
        from dashboard.components.archive_loader import _read_jsonl_via_worldspace

        archive_path = (
            _REPO_ROOT
            / "artifacts"
            / "map_elites_smoke_cvt"
            / "map_elites_archive.jsonl"
        )
        if not archive_path.is_file():
            self.skipTest("CVT smoke archive missing")
        frame, _ = _read_jsonl_via_worldspace(archive_path, archive_type="cvt")
        self.assertFalse(frame.empty)
        self.assertTrue((frame["archive_type"] == "cvt").all())


if __name__ == "__main__":
    unittest.main()

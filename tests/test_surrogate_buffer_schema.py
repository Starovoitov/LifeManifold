"""Tests for strict schema 2.0 surrogate buffer loading and migration."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from worldspace.illuminators.evaluation import apply_canonical_seed
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec
from worldspace.surrogate.backfill import backfill_buffer_from_archive
from worldspace.surrogate.buffer import buffer_record, world_spec_dict_for_buffer
from worldspace.surrogate.buffer_migrate import re_featurize_buffer
from worldspace.surrogate.feature_extractor import (
    FEATURE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION_20,
    extract,
)
from worldspace.surrogate.genome_features import FEATURE_DIM, FEATURE_DIM_V21
from worldspace.surrogate.synthetic_buffer import write_synthetic_buffer
from worldspace.surrogate.training import (
    BUFFER_FEATURE_DIM,
    BUFFER_SCHEMA_VERSION,
    load_buffer,
    scan_buffer_rows,
)


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


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


class TestSurrogateBufferSchema(unittest.TestCase):
    def test_load_buffer_accepts_schema_v2_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer.jsonl"
            write_synthetic_buffer(path, n_samples=8, seed=3)
            features, targets = load_buffer(path)
        self.assertEqual(features.shape, (8, BUFFER_FEATURE_DIM))
        self.assertEqual(len(targets["stability"]), 8)

    def test_load_buffer_rejects_legacy_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer.jsonl"
            _write_rows(
                path,
                [
                    {
                        "feature_schema_version": "1.0",
                        "emitter_type": "random",
                        "features": [1.0, 2.0],
                        "targets": _sample_targets(),
                    }
                ],
            )
            with self.assertRaisesRegex(
                ValueError, "Unsupported feature_schema_version"
            ):
                load_buffer(path)

    def test_load_buffer_rejects_wrong_feature_dim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer.jsonl"
            _write_rows(
                path,
                [
                    buffer_record(
                        features=np.zeros(8, dtype=float),
                        targets=_sample_targets(),
                        emitter_type="random",
                        world_spec=_sample_world_spec_dict(),
                    )
                ],
            )
            with self.assertRaisesRegex(ValueError, "Invalid feature dimension"):
                load_buffer(path)

    def test_load_buffer_rejects_missing_world_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer.jsonl"
            row = buffer_record(
                features=np.zeros(FEATURE_DIM_V21, dtype=float),
                targets=_sample_targets(),
                emitter_type="random",
                world_spec=_sample_world_spec_dict(),
            )
            del row["world_spec"]
            _write_rows(path, [row])
            with self.assertRaisesRegex(ValueError, "Missing world_spec"):
                load_buffer(path)

    def test_scan_buffer_rows_reports_invalid_legacy_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer.jsonl"
            write_synthetic_buffer(path, n_samples=2, seed=1)
            path.write_text(
                path.read_text(encoding="utf-8")
                + json.dumps(
                    {
                        "feature_schema_version": "1.0",
                        "emitter_type": "random",
                        "features": [1.0, 2.0],
                        "targets": _sample_targets(),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            stats = scan_buffer_rows(path)
        self.assertEqual(stats["valid_rows"], 2)
        self.assertEqual(stats["invalid_rows"], 1)
        self.assertEqual(stats["feature_schema_version"], BUFFER_SCHEMA_VERSION)

    def test_migrate_script_writes_trainable_buffer(self) -> None:
        archive = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "map_elites_smoke"
            / "map_elites_archive.jsonl"
        )
        if not archive.is_file():
            self.skipTest("smoke archive missing")
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "buffer.jsonl"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/migrate_surrogate_buffer.py",
                    "--archive",
                    str(archive),
                    "--output",
                    str(output),
                    "--overwrite",
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(result.stdout.strip())
            self.assertEqual(payload["feature_schema_version"], "2.1")
            self.assertEqual(payload["feature_dim"], BUFFER_FEATURE_DIM)
            self.assertGreater(payload["loaded_rows"], 0)
            features, _ = load_buffer(output)
            self.assertEqual(features.shape[1], BUFFER_FEATURE_DIM)

    def test_train_summary_includes_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            buffer_path = root / "buffer.jsonl"
            write_synthetic_buffer(buffer_path, n_samples=160, seed=9)
            checkpoint_path = root / "micro.pkl"
            summary_path = root / "micro.summary.json"
            repo_root = Path(__file__).resolve().parents[1]
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_surrogate.py",
                    "--buffer-path",
                    str(buffer_path),
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--summary-path",
                    str(summary_path),
                    "--micro",
                    "--no-quality-gate",
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["feature_schema_version"], "2.1")
        self.assertEqual(summary["feature_dim"], BUFFER_FEATURE_DIM)

    def test_backfill_output_is_loadable(self) -> None:
        archive = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "map_elites_smoke"
            / "map_elites_archive.jsonl"
        )
        if not archive.is_file():
            self.skipTest("smoke archive missing")
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            backfill_buffer_from_archive(archive, buffer_path)
            features, targets = load_buffer(buffer_path)
        self.assertEqual(features.shape[1], BUFFER_FEATURE_DIM)
        self.assertEqual(features.shape[0], len(targets["stability"]))

    def test_load_buffer_accepts_schema_20_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer_v20.jsonl"
            write_synthetic_buffer(path, n_samples=6, seed=4)
            downgraded: list[str] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                spec = WorldSpec.from_json_dict(row["world_spec"])
                apply_canonical_seed(spec)
                row["feature_schema_version"] = FEATURE_SCHEMA_VERSION_20
                row["features"] = [
                    float(value)
                    for value in extract(spec, schema_version=FEATURE_SCHEMA_VERSION_20)
                ]
                downgraded.append(json.dumps(row, sort_keys=True))
            path.write_text("\n".join(downgraded) + "\n", encoding="utf-8")
            features, targets = load_buffer(path)
        self.assertEqual(features.shape, (6, FEATURE_DIM))
        self.assertEqual(len(targets["stability"]), 6)

    def test_load_buffer_rejects_mixed_schema_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mixed.jsonl"
            write_synthetic_buffer(path, n_samples=4, seed=2)
            lines = path.read_text(encoding="utf-8").splitlines()
            row = json.loads(lines[0])
            spec = WorldSpec.from_json_dict(row["world_spec"])
            apply_canonical_seed(spec)
            row["feature_schema_version"] = FEATURE_SCHEMA_VERSION_20
            row["features"] = [
                float(value)
                for value in extract(spec, schema_version=FEATURE_SCHEMA_VERSION_20)
            ]
            lines[0] = json.dumps(row, sort_keys=True)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Mixed feature_schema_version"):
                load_buffer(path)

    def test_re_featurize_upgrades_schema_20_to_21(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "buffer_v20.jsonl"
            target = root / "buffer_v21.jsonl"
            write_synthetic_buffer(source, n_samples=10, seed=6)
            downgraded: list[str] = []
            for line in source.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                spec = WorldSpec.from_json_dict(row["world_spec"])
                apply_canonical_seed(spec)
                row["feature_schema_version"] = FEATURE_SCHEMA_VERSION_20
                row["features"] = [
                    float(value)
                    for value in extract(spec, schema_version=FEATURE_SCHEMA_VERSION_20)
                ]
                downgraded.append(json.dumps(row, sort_keys=True))
            source.write_text("\n".join(downgraded) + "\n", encoding="utf-8")
            stats = re_featurize_buffer(
                source,
                target,
                target_schema=FEATURE_SCHEMA_VERSION,
                overwrite=True,
            )
            features, targets = load_buffer(target)
            self.assertEqual(stats["rows_written"], 10)
            self.assertEqual(features.shape, (10, FEATURE_DIM_V21))
            self.assertEqual(len(targets["stability"]), 10)
            first = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first["feature_schema_version"], "2.1")
            self.assertEqual(len(first["features"]), FEATURE_DIM_V21)

    def test_migrate_script_re_featurize_mode(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "buffer_v20.jsonl"
            target = root / "buffer_v21.jsonl"
            write_synthetic_buffer(source, n_samples=5, seed=8)
            downgraded: list[str] = []
            for line in source.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                spec = WorldSpec.from_json_dict(row["world_spec"])
                apply_canonical_seed(spec)
                row["feature_schema_version"] = FEATURE_SCHEMA_VERSION_20
                row["features"] = [
                    float(value)
                    for value in extract(spec, schema_version=FEATURE_SCHEMA_VERSION_20)
                ]
                downgraded.append(json.dumps(row, sort_keys=True))
            source.write_text("\n".join(downgraded) + "\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/migrate_surrogate_buffer.py",
                    "--buffer",
                    str(source),
                    "--re-featurize",
                    "--target-schema",
                    "2.1",
                    "--output",
                    str(target),
                    "--overwrite",
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(result.stdout.strip())
            self.assertEqual(payload["mode"], "re_featurize")
            self.assertEqual(payload["feature_schema_version"], "2.1")
            self.assertEqual(payload["feature_dim"], FEATURE_DIM_V21)
            features, _ = load_buffer(target)
            self.assertEqual(features.shape[1], FEATURE_DIM_V21)


if __name__ == "__main__":
    unittest.main()

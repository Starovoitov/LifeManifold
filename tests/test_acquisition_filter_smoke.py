"""End-to-end smoke: filter mode skips at least one candidate slot."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import yaml

from worldspace.illuminators.illuminator import MapElitesIlluminator, archive_jsonl_path
from worldspace.illuminators.scheduler import (
    DEFAULT_MINI_SCHEDULER_PATH,
    load_scheduler,
)

_MINI_SEED = 42
_MINI_GRID_SIZE = 8
_MINI_STEPS = 200
_FAST_ITERATIONS = 2
_MAX_SECONDS = 30.0


def _write_filter_scheduler(
    path: Path, *, buffer_path: Path, checkpoint_path: Path
) -> None:
    """Mini scheduler with stub predictions that trigger threshold_gate skips."""
    raw = yaml.safe_load(DEFAULT_MINI_SCHEDULER_PATH.read_text(encoding="utf-8"))
    raw["iterations"] = _FAST_ITERATIONS
    raw["initial_random_candidates"] = 0
    raw["llm"] = {"enabled": False}
    raw["surrogate"] = {
        "enabled": True,
        "model_type": "lightgbm",
        "checkpoint": str(checkpoint_path),
        "buffer_path": str(buffer_path),
        "stub_mean": 0.1,
        "stub_uncertainty": 0.1,
        "acquisition": {
            "mode": "filter",
            "policy": "threshold_gate",
            "min_predicted_fitness": 0.25,
            "max_uncertainty_to_skip": 0.40,
            "never_skip_empty_bin": False,
        },
        "retrain": {"enabled": False},
    }
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


class TestAcquisitionFilterSmoke(unittest.TestCase):
    def test_filter_run_skips_at_least_one_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scheduler_path = root / "scheduler_filter.yaml"
            buffer_path = root / "buffer.jsonl"
            checkpoint_path = root / "missing_checkpoint.pkl"
            output_dir = root / "run_filter"
            _write_filter_scheduler(
                scheduler_path,
                buffer_path=buffer_path,
                checkpoint_path=checkpoint_path,
            )
            config = load_scheduler(scheduler_path)
            expected_slots = config.iterations * config.batch_size

            started = time.perf_counter()
            result = MapElitesIlluminator().run(
                scheduler_path=scheduler_path,
                output_dir=output_dir,
                seed=_MINI_SEED,
                grid_size=_MINI_GRID_SIZE,
                steps=_MINI_STEPS,
            )
            elapsed = time.perf_counter() - started

            self.assertLess(elapsed, _MAX_SECONDS)
            self.assertLess(result.evaluations, expected_slots)

            archive_path = archive_jsonl_path(output_dir)
            archive_lines: list[str] = []
            if archive_path.is_file():
                archive_lines = [
                    line
                    for line in archive_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            self.assertEqual(len(archive_lines), result.evaluations)

            surrogate_archive = result.surrogate_archive_jsonl_path
            self.assertIsNotNone(surrogate_archive)
            assert surrogate_archive is not None
            records = [
                json.loads(line)
                for line in surrogate_archive.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), expected_slots)
            skip_count = sum(
                1 for record in records if record.get("decision") == "skip"
            )
            self.assertGreaterEqual(skip_count, 1)
            self.assertEqual(skip_count, expected_slots - result.evaluations)
            if result.evaluations > 0:
                self.assertTrue(buffer_path.is_file())
                buffer_lines = [
                    line
                    for line in buffer_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertEqual(len(buffer_lines), result.evaluations)
            else:
                self.assertFalse(buffer_path.is_file())


if __name__ == "__main__":
    unittest.main()

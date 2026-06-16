"""Smoke test with surrogate.enabled=true and micro-checkpoint."""

from __future__ import annotations

import os
import subprocess
import sys
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
from worldspace.surrogate import get_surrogate
from worldspace.surrogate.synthetic_buffer import write_synthetic_buffer
from worldspace.surrogate.types import SurrogateConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MAX_SECONDS = 120.0
_FAST_ITERATIONS = 2


def _write_scheduler_with_surrogate(
    path: Path,
    *,
    enabled: bool,
    checkpoint_path: Path,
    buffer_path: Path,
    model_type: str = "lightgbm",
) -> None:
    raw = yaml.safe_load(DEFAULT_MINI_SCHEDULER_PATH.read_text(encoding="utf-8"))
    raw["iterations"] = _FAST_ITERATIONS
    raw["initial_random_candidates"] = 0
    raw["llm"] = {"enabled": False}
    raw["surrogate"] = {
        "enabled": enabled,
        "model_type": model_type,
        "checkpoint": str(checkpoint_path),
        "buffer_path": str(buffer_path),
        "stub_mean": 0.5,
        "stub_uncertainty": 1.0,
    }
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


class TestSurrogateEnabledSmoke(unittest.TestCase):
    def test_mini_run_with_micro_checkpoint_completes_quickly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            buffer_path = root / "train_buffer.jsonl"
            write_synthetic_buffer(buffer_path, n_samples=150, seed=99)
            checkpoint_path = root / "micro.pkl"
            summary_path = root / "micro.summary.json"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(_REPO_ROOT)
            train = subprocess.run(
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
                ],
                cwd=_REPO_ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(train.returncode, 0, msg=train.stderr)

            scheduler_path = root / "scheduler_surrogate.yaml"
            run_buffer = root / "run_buffer.jsonl"
            _write_scheduler_with_surrogate(
                scheduler_path,
                enabled=True,
                checkpoint_path=checkpoint_path,
                buffer_path=run_buffer,
            )
            config = load_scheduler(scheduler_path)
            self.assertTrue(config.surrogate_enabled)

            out_dir = root / "smoke_out"
            started = time.perf_counter()
            result = MapElitesIlluminator().run(
                scheduler_path=scheduler_path,
                output_dir=out_dir,
                seed=42,
                grid_size=8,
                steps=200,
            )
            elapsed = time.perf_counter() - started
            self.assertLess(elapsed, _MAX_SECONDS)
            jsonl_path = archive_jsonl_path(out_dir)
            self.assertTrue(jsonl_path.is_file())
            self.assertGreater(result.filled_cells, 0)

            surrogate = get_surrogate(
                SurrogateConfig(
                    enabled=True,
                    model_type="lightgbm",
                    checkpoint=str(checkpoint_path),
                    stub_mean=0.5,
                    stub_uncertainty=1.0,
                )
            )
            from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec

            spec = WorldSpec(
                birth=[1, 3],
                survival=[2, 3],
                noise=0.02,
                resource_regen=0.05,
                predation=0.1,
                cell_types=CANONICAL_CELL_TYPES.copy(),
                grid_size=8,
                steps=200,
                seed=0,
            )
            prediction = surrogate.predict(spec)
            self.assertGreaterEqual(prediction.fitness, 0.0)
            self.assertLessEqual(prediction.fitness, 1.0)

    def test_mini_run_with_mlp_checkpoint_completes_quickly(self) -> None:
        from importlib.util import find_spec

        if find_spec("torch") is None:
            self.skipTest("torch not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            buffer_path = root / "train_buffer.jsonl"
            write_synthetic_buffer(buffer_path, n_samples=150, seed=99)
            checkpoint_path = root / "micro_mlp.pkl"
            summary_path = root / "micro_mlp.summary.json"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(_REPO_ROOT)
            train = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_surrogate.py",
                    "--model-type",
                    "mlp",
                    "--buffer-path",
                    str(buffer_path),
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--summary-path",
                    str(summary_path),
                    "--micro",
                ],
                cwd=_REPO_ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(train.returncode, 0, msg=train.stderr)

            scheduler_path = root / "scheduler_mlp.yaml"
            run_buffer = root / "run_buffer.jsonl"
            _write_scheduler_with_surrogate(
                scheduler_path,
                enabled=True,
                checkpoint_path=checkpoint_path,
                buffer_path=run_buffer,
                model_type="mlp",
            )
            config = load_scheduler(scheduler_path)
            self.assertEqual(config.surrogate_model_type, "mlp")

            out_dir = root / "smoke_mlp_out"
            result = MapElitesIlluminator().run(
                scheduler_path=scheduler_path,
                output_dir=out_dir,
                seed=42,
                grid_size=8,
                steps=200,
            )
            self.assertGreater(result.filled_cells, 0)
            surrogate = get_surrogate(
                SurrogateConfig(
                    enabled=True,
                    model_type="mlp",
                    checkpoint=str(checkpoint_path),
                    stub_mean=0.5,
                    stub_uncertainty=1.0,
                )
            )
            from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec

            spec = WorldSpec(
                birth=[1, 3],
                survival=[2, 3],
                noise=0.02,
                resource_regen=0.05,
                predation=0.1,
                cell_types=CANONICAL_CELL_TYPES.copy(),
                grid_size=8,
                steps=200,
                seed=0,
            )
            prediction = surrogate.predict(spec)
            import numpy as np

            self.assertTrue(np.isfinite(prediction.fitness))
            self.assertGreaterEqual(prediction.uncertainty, 0.0)


if __name__ == "__main__":
    unittest.main()

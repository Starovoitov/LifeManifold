"""MAP-Elites reproducibility with surrogate on/off when LLM is disabled."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from worldspace.illuminators.archive import GridArchive, load_and_collapse_jsonl
from worldspace.illuminators.archive_protocol import ArchiveProtocol
from worldspace.illuminators.emitters.base import EmitterOutput, MapElitesEmitter
from worldspace.illuminators.emitters.llm_emitter import LlmEmitter
from worldspace.illuminators.illuminator import MapElitesIlluminator
from worldspace.illuminators.loop import run_scheduler
from worldspace.illuminators.scheduler import (
    DEFAULT_MINI_SCHEDULER_PATH,
    TargetCell,
    load_scheduler,
    surrogate_config_from_scheduler,
)
from worldspace.surrogate import get_surrogate
from worldspace.surrogate.buffer import SurrogateBuffer
from worldspace.surrogate.genome_features import FEATURE_DIM_V21

_MINI_SEED = 42
_MINI_GRID_SIZE = 8
_MINI_STEPS = 200
_FAST_ITERATIONS = 2
AcquisitionModeLiteral = Literal["off", "shadow", "filter"]


def _write_mini_scheduler(
    path: Path,
    *,
    surrogate_enabled: bool,
    buffer_path: Path,
    checkpoint_path: Path,
    iterations: int = _FAST_ITERATIONS,
    initial_random_candidates: int = 0,
    acquisition_mode: AcquisitionModeLiteral | None = None,
    stub_mean: float = 0.5,
    stub_uncertainty: float = 1.0,
    min_predicted_fitness: float = 0.25,
    max_uncertainty_to_skip: float = 0.40,
    never_skip_empty_bin: bool = True,
) -> None:
    """Write a mini-scheduler variant with isolated surrogate paths."""
    raw = yaml.safe_load(DEFAULT_MINI_SCHEDULER_PATH.read_text(encoding="utf-8"))
    raw["iterations"] = iterations
    raw["initial_random_candidates"] = initial_random_candidates
    raw["llm"] = {"enabled": False}
    surrogate_block: dict[str, Any] = {
        "enabled": surrogate_enabled,
        "model_type": "lightgbm",
        "checkpoint": str(checkpoint_path),
        "buffer_path": str(buffer_path),
        "stub_mean": stub_mean,
        "stub_uncertainty": stub_uncertainty,
    }
    if acquisition_mode is not None:
        surrogate_block["acquisition"] = {
            "mode": acquisition_mode,
            "policy": "threshold_gate",
            "min_predicted_fitness": min_predicted_fitness,
            "max_uncertainty_to_skip": max_uncertainty_to_skip,
            "never_skip_empty_bin": never_skip_empty_bin,
        }
    raw["surrogate"] = surrogate_block
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _run_illuminator(
    scheduler_path: Path,
    *,
    output_dir: Path,
    seed: int = _MINI_SEED,
):
    return MapElitesIlluminator().run(
        scheduler_path=scheduler_path,
        output_dir=output_dir,
        seed=seed,
        grid_size=_MINI_GRID_SIZE,
        steps=_MINI_STEPS,
    )


def _semantic_archive_snapshot(
    archive_path: Path, *, resolution: int
) -> dict[tuple[int, int], dict[str, Any]]:
    """Compare archive outcomes excluding runtime UUID/timestamp metadata."""
    archive = load_and_collapse_jsonl(archive_path, resolution=resolution)
    snapshot: dict[tuple[int, int], dict[str, Any]] = {}
    for i in range(resolution):
        for j in range(resolution):
            elite = archive.get_cell(archive.cell_id_from_bin((i, j)))
            if elite is None:
                continue
            if elite.world_spec is None or elite.measures is None:
                msg = "elite missing world_spec or measures"
                raise ValueError(msg)
            snapshot[(i, j)] = {
                "fitness": elite.fitness,
                "measures": dict(elite.measures),
                "world_spec": elite.world_spec.to_canonical_dict(),
                "emitter_type": (
                    elite.metadata.emitter_type if elite.metadata else None
                ),
                "generated_by": (
                    elite.metadata.generated_by if elite.metadata else None
                ),
            }
    return snapshot


class TestSurrogateReproducibilityLlmOff(unittest.TestCase):
    def test_archive_jsonl_identical_surrogate_on_vs_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scheduler_off = root / "scheduler_off.yaml"
            scheduler_on = root / "scheduler_on.yaml"
            buffer_off = root / "buffer_off.jsonl"
            buffer_on = root / "buffer_on.jsonl"
            checkpoint = root / "missing_checkpoint.pkl"
            _write_mini_scheduler(
                scheduler_off,
                surrogate_enabled=False,
                buffer_path=buffer_off,
                checkpoint_path=checkpoint,
            )
            _write_mini_scheduler(
                scheduler_on,
                surrogate_enabled=True,
                buffer_path=buffer_on,
                checkpoint_path=checkpoint,
            )
            out_off = root / "run_off"
            out_on = root / "run_on"
            result_off = _run_illuminator(scheduler_off, output_dir=out_off)
            result_on = _run_illuminator(scheduler_on, output_dir=out_on)
            path_off = result_off.archive_jsonl_path
            path_on = result_on.archive_jsonl_path
            config = load_scheduler(scheduler_off)
            snap_off = _semantic_archive_snapshot(
                path_off, resolution=config.grid_resolution
            )
            snap_on = _semantic_archive_snapshot(
                path_on, resolution=config.grid_resolution
            )
            self.assertEqual(snap_off, snap_on)
            self.assertGreater(len(snap_off), 0)
            self.assertFalse(buffer_off.exists())
            self.assertTrue(buffer_on.exists())

    def test_surrogate_on_off_same_seed_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scheduler_path = root / "scheduler.yaml"
            _write_mini_scheduler(
                scheduler_path,
                surrogate_enabled=True,
                buffer_path=root / "buffer_a.jsonl",
                checkpoint_path=root / "missing.pkl",
            )
            path_a = _run_illuminator(
                scheduler_path, output_dir=root / "run_a"
            ).archive_jsonl_path
            scheduler_path_2 = root / "scheduler_b.yaml"
            _write_mini_scheduler(
                scheduler_path_2,
                surrogate_enabled=True,
                buffer_path=root / "buffer_b.jsonl",
                checkpoint_path=root / "missing.pkl",
            )
            path_b = _run_illuminator(
                scheduler_path_2, output_dir=root / "run_b"
            ).archive_jsonl_path
            config = load_scheduler(scheduler_path)
            self.assertEqual(
                _semantic_archive_snapshot(path_a, resolution=config.grid_resolution),
                _semantic_archive_snapshot(path_b, resolution=config.grid_resolution),
            )


class _FailingLlmEmitter(LlmEmitter):
    emit_calls = 0

    def emit(
        self,
        *,
        target: TargetCell,
        archive: ArchiveProtocol,
        rng,
        grid_size: int,
        steps: int,
    ) -> EmitterOutput:
        del target, archive, rng, grid_size, steps
        _FailingLlmEmitter.emit_calls += 1
        raise AssertionError("LlmEmitter.emit must not run when llm.enabled is false")


class TestSurrogateEnabledLlmDisabledIntegration(unittest.TestCase):
    def setUp(self) -> None:
        _FailingLlmEmitter.emit_calls = 0

    def test_surrogate_enabled_never_invokes_llm_emitter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scheduler_path = root / "scheduler.yaml"
            _write_mini_scheduler(
                scheduler_path,
                surrogate_enabled=True,
                buffer_path=root / "buffer.jsonl",
                checkpoint_path=root / "missing.pkl",
                initial_random_candidates=0,
            )
            config = load_scheduler(scheduler_path)
            self.assertTrue(config.surrogate_enabled)
            self.assertFalse(config.llm_enabled)
            archive = GridArchive(config.grid_resolution)
            emitter = MapElitesEmitter(
                scheduler=config,
                surrogate=get_surrogate(surrogate_config_from_scheduler(config)),
                llm_emitter=_FailingLlmEmitter(
                    grid_resolution=config.grid_resolution,
                ),
            )
            buffer = SurrogateBuffer(config.surrogate_buffer_path, flush_every=32)
            run_scheduler(
                config,
                archive,
                np.random.default_rng(_MINI_SEED),
                emitter,
                grid_size=_MINI_GRID_SIZE,
                steps=_MINI_STEPS,
                surrogate_buffer=buffer,
            )
            buffer.flush()
            self.assertEqual(_FailingLlmEmitter.emit_calls, 0)
            self.assertGreater(archive.filled_count(), 0)


class TestAcquisitionReproducibility(unittest.TestCase):
    """Acquisition modes must not change MAP-Elites archive outcomes (off vs shadow)."""

    def test_explicit_acquisition_off_matches_default_surrogate_on(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "missing_checkpoint.pkl"
            scheduler_default = root / "scheduler_default.yaml"
            scheduler_explicit_off = root / "scheduler_off.yaml"
            _write_mini_scheduler(
                scheduler_default,
                surrogate_enabled=True,
                buffer_path=root / "buffer_default.jsonl",
                checkpoint_path=checkpoint,
            )
            _write_mini_scheduler(
                scheduler_explicit_off,
                surrogate_enabled=True,
                buffer_path=root / "buffer_off.jsonl",
                checkpoint_path=checkpoint,
                acquisition_mode="off",
            )
            result_default = _run_illuminator(
                scheduler_default, output_dir=root / "run_default"
            )
            result_off = _run_illuminator(
                scheduler_explicit_off, output_dir=root / "run_off"
            )
            config = load_scheduler(scheduler_default)
            resolution = config.grid_resolution
            snap_default = _semantic_archive_snapshot(
                result_default.archive_jsonl_path, resolution=resolution
            )
            snap_off = _semantic_archive_snapshot(
                result_off.archive_jsonl_path, resolution=resolution
            )
            self.assertEqual(snap_default, snap_off)
            self.assertEqual(
                result_default.evaluations,
                result_off.evaluations,
            )
            self.assertIsNone(result_off.surrogate_archive_jsonl_path)

    def test_shadow_archive_identical_to_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "missing_checkpoint.pkl"
            scheduler_off = root / "scheduler_off.yaml"
            scheduler_shadow = root / "scheduler_shadow.yaml"
            _write_mini_scheduler(
                scheduler_off,
                surrogate_enabled=True,
                buffer_path=root / "buffer_off.jsonl",
                checkpoint_path=checkpoint,
                acquisition_mode="off",
            )
            _write_mini_scheduler(
                scheduler_shadow,
                surrogate_enabled=True,
                buffer_path=root / "buffer_shadow.jsonl",
                checkpoint_path=checkpoint,
                acquisition_mode="shadow",
            )
            result_off = _run_illuminator(scheduler_off, output_dir=root / "run_off")
            result_shadow = _run_illuminator(
                scheduler_shadow, output_dir=root / "run_shadow"
            )
            config = load_scheduler(scheduler_off)
            resolution = config.grid_resolution
            expected_evaluations = config.iterations * config.batch_size
            snap_off = _semantic_archive_snapshot(
                result_off.archive_jsonl_path, resolution=resolution
            )
            snap_shadow = _semantic_archive_snapshot(
                result_shadow.archive_jsonl_path, resolution=resolution
            )
            self.assertEqual(snap_off, snap_shadow)
            self.assertEqual(result_off.evaluations, expected_evaluations)
            self.assertEqual(result_shadow.evaluations, expected_evaluations)
            self.assertEqual(result_off.evaluations, result_shadow.evaluations)
            self.assertIsNone(result_off.surrogate_archive_jsonl_path)
            shadow_archive = result_shadow.surrogate_archive_jsonl_path
            self.assertIsNotNone(shadow_archive)
            assert shadow_archive is not None
            self.assertTrue(shadow_archive.is_file())
            lines = [
                line
                for line in shadow_archive.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(lines), expected_evaluations)
            skip_logged = sum(
                1 for line in lines if json.loads(line).get("decision") == "skip"
            )
            self.assertGreaterEqual(
                skip_logged,
                0,
                "shadow may log policy skip recommendations without skipping eval",
            )

    def test_shadow_archive_identical_to_off_with_v2_checkpoint(self) -> None:
        import os
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            buffer_path = root / "train_buffer.jsonl"
            from worldspace.surrogate.synthetic_buffer import write_synthetic_buffer

            write_synthetic_buffer(buffer_path, n_samples=150, seed=99)
            checkpoint_path = root / "micro_v2.pkl"
            summary_path = root / "micro_v2.summary.json"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
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
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(train.returncode, 0, msg=train.stderr)
            scheduler_off = root / "scheduler_off.yaml"
            scheduler_shadow = root / "scheduler_shadow.yaml"
            _write_mini_scheduler(
                scheduler_off,
                surrogate_enabled=True,
                buffer_path=root / "buffer_off.jsonl",
                checkpoint_path=checkpoint_path,
                acquisition_mode="off",
            )
            _write_mini_scheduler(
                scheduler_shadow,
                surrogate_enabled=True,
                buffer_path=root / "buffer_shadow.jsonl",
                checkpoint_path=checkpoint_path,
                acquisition_mode="shadow",
            )
            result_off = _run_illuminator(scheduler_off, output_dir=root / "run_off")
            result_shadow = _run_illuminator(
                scheduler_shadow, output_dir=root / "run_shadow"
            )
            config = load_scheduler(scheduler_off)
            resolution = config.grid_resolution
            snap_off = _semantic_archive_snapshot(
                result_off.archive_jsonl_path, resolution=resolution
            )
            snap_shadow = _semantic_archive_snapshot(
                result_shadow.archive_jsonl_path, resolution=resolution
            )
            self.assertEqual(snap_off, snap_shadow)
            shadow_buffer = root / "buffer_shadow.jsonl"
            self.assertTrue(shadow_buffer.is_file())
            buffer_lines = [
                line
                for line in shadow_buffer.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(buffer_lines), result_shadow.evaluations)
            for line in buffer_lines:
                row = json.loads(line)
                self.assertEqual(row["feature_schema_version"], "2.1")
                self.assertEqual(len(row["features"]), FEATURE_DIM_V21)


if __name__ == "__main__":
    unittest.main()

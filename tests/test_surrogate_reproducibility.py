"""MAP-Elites reproducibility with surrogate on/off when LLM is disabled (E6.2)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from worldspace.illuminators.archive import GridArchive, load_and_collapse_jsonl
from worldspace.illuminators.emitters.base import EmitterOutput, MapElitesEmitter
from worldspace.illuminators.emitters.llm_emitter import LlmEmitter
from worldspace.illuminators.illuminator import MapElitesIlluminator, archive_jsonl_path
from worldspace.illuminators.loop import run_scheduler
from worldspace.illuminators.scheduler import (
    DEFAULT_MINI_SCHEDULER_PATH,
    TargetBin,
    load_scheduler,
    surrogate_config_from_scheduler,
)
from worldspace.surrogate import get_surrogate
from worldspace.surrogate.buffer import SurrogateBuffer

_MINI_SEED = 42
_MINI_GRID_SIZE = 8
_MINI_STEPS = 200
_FAST_ITERATIONS = 2


def _write_mini_scheduler(
    path: Path,
    *,
    surrogate_enabled: bool,
    buffer_path: Path,
    checkpoint_path: Path,
    iterations: int = _FAST_ITERATIONS,
    initial_random_candidates: int = 0,
) -> None:
    """Write a mini-scheduler variant with isolated surrogate paths."""
    raw = yaml.safe_load(DEFAULT_MINI_SCHEDULER_PATH.read_text(encoding="utf-8"))
    raw["iterations"] = iterations
    raw["initial_random_candidates"] = initial_random_candidates
    raw["llm"] = {"enabled": False}
    raw["surrogate"] = {
        "enabled": surrogate_enabled,
        "model_type": "lightgbm",
        "checkpoint": str(checkpoint_path),
        "buffer_path": str(buffer_path),
        "stub_mean": 0.5,
        "stub_uncertainty": 1.0,
    }
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _run_illuminator(
    scheduler_path: Path,
    *,
    output_dir: Path,
    seed: int = _MINI_SEED,
) -> Path:
    MapElitesIlluminator().run(
        scheduler_path=scheduler_path,
        output_dir=output_dir,
        seed=seed,
        grid_size=_MINI_GRID_SIZE,
        steps=_MINI_STEPS,
    )
    return archive_jsonl_path(output_dir)


def _semantic_archive_snapshot(
    archive_path: Path, *, resolution: int
) -> dict[tuple[int, int], dict[str, Any]]:
    """Compare archive outcomes excluding runtime UUID/timestamp metadata."""
    archive = load_and_collapse_jsonl(archive_path, resolution=resolution)
    snapshot: dict[tuple[int, int], dict[str, Any]] = {}
    for i in range(resolution):
        for j in range(resolution):
            elite = archive.get(i, j)
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
            path_off = _run_illuminator(scheduler_off, output_dir=out_off)
            path_on = _run_illuminator(scheduler_on, output_dir=out_on)
            config = load_scheduler(scheduler_off)
            snap_off = _semantic_archive_snapshot(
                path_off, resolution=config.grid_resolution
            )
            snap_on = _semantic_archive_snapshot(
                path_on, resolution=config.grid_resolution
            )
            self.assertEqual(snap_off, snap_on)
            self.assertGreater(len(snap_off), 0)
            self.assertTrue(buffer_off.exists())
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
            path_a = _run_illuminator(scheduler_path, output_dir=root / "run_a")
            scheduler_path_2 = root / "scheduler_b.yaml"
            _write_mini_scheduler(
                scheduler_path_2,
                surrogate_enabled=True,
                buffer_path=root / "buffer_b.jsonl",
                checkpoint_path=root / "missing.pkl",
            )
            path_b = _run_illuminator(scheduler_path_2, output_dir=root / "run_b")
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
        target: TargetBin,
        archive: GridArchive,
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


if __name__ == "__main__":
    unittest.main()

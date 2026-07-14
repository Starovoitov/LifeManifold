"""MAP-Elites illuminator entrypoint: scheduler, archive, loop, JSONL output."""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from worldspace.illuminators.archive import load_and_collapse_jsonl
from worldspace.illuminators.archive_trace import ARCHIVE_TRACE_FILENAME
from worldspace.illuminators.archive_factory import (
    archive_factory_config_from_scheduler,
    create_archive,
)
from worldspace.illuminators.archive_protocol import ArchiveProtocol
from worldspace.illuminators.cvt import centroids_path_for_output
from worldspace.illuminators.emitters.base import CandidateEmitter, MapElitesEmitter
from worldspace.illuminators.evaluation import ILLUMINATOR_MIN_STEPS
from worldspace.illuminators.loop import run_scheduler
from worldspace.illuminators.scheduler import (
    DEFAULT_SCHEDULER_PATH,
    RunCounters,
    SchedulerConfig,
    load_scheduler,
    surrogate_config_from_scheduler,
)


@dataclass(frozen=True)
class MapElitesRunResult:
    """Summary after a full MAP-Elites illuminator run."""

    iterations: int
    evaluations: int
    filled_cells: int
    archive_jsonl_path: Path
    counters: RunCounters
    surrogate_archive_jsonl_path: Path | None = None


class MapElitesIlluminator:
    """Orchestrate scheduler load, optional archive resume, and the iteration loop."""

    def run(
        self,
        *,
        scheduler_path: str | Path | None = None,
        output_dir: str | Path = "output",
        seed: int = 0,
        grid_resolution: int | None = None,
        grid_size: int = 50,
        steps: int = 300,
        iterations: int | None = None,
        load_archive_path: str | Path | None = None,
        archive_type: Literal["grid", "cvt"] | None = None,
        emitter: CandidateEmitter | None = None,
        llm_spec_path: str | Path | None = None,
        require_surrogate_quality_gate: bool = False,
        surrogate_checkpoint_override: str | Path | None = None,
    ) -> MapElitesRunResult:
        """Run MAP-Elites for ``iterations × batch_size`` candidate slots."""
        config = load_scheduler(
            scheduler_path or DEFAULT_SCHEDULER_PATH,
            iterations_override=iterations,
        )
        if archive_type is not None:
            if config.schema_version != "1.3":
                msg = (
                    "--archive-type override requires scheduler schema_version 1.3, "
                    f"got {config.schema_version!r}"
                )
                raise ValueError(msg)
            config = replace(config, archive_type=archive_type)
        if grid_resolution is not None and config.archive_type == "grid":
            config = replace(config, grid_resolution=grid_resolution)
        if surrogate_checkpoint_override is not None:
            config = replace(
                config,
                surrogate_checkpoint=str(surrogate_checkpoint_override),
            )
        effective_steps = normalize_illuminator_steps(steps, min_steps=config.min_steps)
        out_dir = Path(output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = archive_jsonl_path(out_dir)
        _clear_stale_run_artifacts(
            out_dir,
            jsonl_path,
            load_archive_path=load_archive_path,
        )

        archive, counters = _load_archive_and_counters(
            config,
            load_archive_path=load_archive_path,
            output_dir=out_dir,
        )
        start_evaluated = counters.candidates_evaluated
        rng = np.random.default_rng(seed)
        from worldspace.surrogate import get_surrogate
        from worldspace.surrogate.buffer import SurrogateBuffer
        from worldspace.surrogate.surrogate_archive import (
            open_surrogate_archive,
            surrogate_archive_path_for_output,
        )

        surrogate = get_surrogate(
            surrogate_config_from_scheduler(
                config,
                require_quality_gate=require_surrogate_quality_gate,
            )
        )
        surrogate_buffer = None
        if config.surrogate_enabled:
            surrogate_buffer = SurrogateBuffer(
                config.surrogate_buffer_path,
                flush_every=32,
            )
        retrain_state = None
        if config.retrain.enabled:
            from worldspace.surrogate.buffer import count_buffer_rows
            from worldspace.surrogate.retrain import RetrainState

            retrain_state = RetrainState(
                buffer_row_count_at_last_retrain=count_buffer_rows(
                    config.surrogate_buffer_path,
                ),
            )
        effective_emitter = emitter or MapElitesEmitter(
            scheduler=config,
            surrogate=surrogate,
            llm_spec_path=llm_spec_path,
        )
        run_id = uuid.uuid4().hex
        surrogate_archive_path = surrogate_archive_path_for_output(out_dir)
        archive_logging_enabled = (
            config.surrogate_enabled and config.acquisition.mode != "off"
        )
        surrogate_archive = open_surrogate_archive(
            surrogate_archive_path,
            run_id=run_id,
            enabled=archive_logging_enabled,
        )
        try:
            counters = run_scheduler(
                config,
                archive,
                rng,
                effective_emitter,
                grid_size=grid_size,
                steps=effective_steps,
                jsonl_path=jsonl_path,
                counters=counters,
                surrogate_buffer=surrogate_buffer,
                surrogate=surrogate,
                retrain_state=retrain_state,
                surrogate_archive=surrogate_archive,
            )
        finally:
            surrogate_archive.close()
        if surrogate_buffer is not None:
            surrogate_buffer.flush()
        expected_slots = config.iterations * config.batch_size
        run_evaluations = counters.candidates_evaluated - start_evaluated
        if run_evaluations > expected_slots:
            msg = (
                f"expected at most {expected_slots} evaluations this run, "
                f"got {run_evaluations}"
            )
            raise RuntimeError(msg)
        if config.acquisition.mode in ("off", "shadow"):
            if run_evaluations != expected_slots:
                msg = (
                    f"expected {expected_slots} evaluations this run, "
                    f"got {run_evaluations}"
                )
                raise RuntimeError(msg)
        return MapElitesRunResult(
            iterations=config.iterations,
            evaluations=run_evaluations,
            filled_cells=archive.filled_count(),
            archive_jsonl_path=jsonl_path,
            surrogate_archive_jsonl_path=(
                surrogate_archive_path if archive_logging_enabled else None
            ),
            counters=counters,
        )


def archive_jsonl_path(output_dir: str | Path) -> Path:
    """Return the canonical MAP-Elites archive JSONL path under ``output_dir``."""
    return Path(output_dir).expanduser() / _ARCHIVE_JSONL_NAME


def normalize_illuminator_steps(
    steps: int,
    *,
    min_steps: int = ILLUMINATOR_MIN_STEPS,
) -> int:
    """Apply illuminator minimum simulation length (§5)."""
    return max(int(steps), int(min_steps))


__all__ = [
    "MapElitesIlluminator",
    "MapElitesRunResult",
    "archive_jsonl_path",
    "normalize_illuminator_steps",
]

_ARCHIVE_JSONL_NAME = "map_elites_archive.jsonl"
_RUN_ARTIFACT_NAMES = (
    _ARCHIVE_JSONL_NAME,
    "surrogate_archive.jsonl",
    "iteration_timing.jsonl",
    ARCHIVE_TRACE_FILENAME,
)


def _clear_stale_run_artifacts(
    out_dir: Path,
    jsonl_path: Path,
    *,
    load_archive_path: str | Path | None,
) -> None:
    """Drop prior per-run JSONL when warming from an external baseline archive.

    Experiment batches load a shared baseline and write run-only deltas under
    ``output_dir``. Restarting without removing stale files would append a second
    run and break ``filled_cells`` validation.
    """
    if load_archive_path is None:
        return
    load_path = Path(load_archive_path).expanduser()
    if load_path.resolve() == jsonl_path.resolve():
        return
    for name in _RUN_ARTIFACT_NAMES:
        path = out_dir / name
        if path.is_file():
            path.unlink()
    # Ensure delta JSONL exists even if no elite beats the warm-start baseline
    # (append_archive_line only creates the file on accepted inserts).
    jsonl_path.touch(exist_ok=True)


def _load_archive_and_counters(
    config: SchedulerConfig,
    *,
    load_archive_path: str | Path | None,
    output_dir: Path,
) -> tuple[ArchiveProtocol, RunCounters]:
    if load_archive_path is not None:
        load_path = Path(load_archive_path).expanduser()
        source_centroids = _centroids_path_for_archive_dir(load_path.parent, config)
        _ensure_cvt_centroids_in_output_dir(
            config,
            output_dir=output_dir,
            source_centroids_path=source_centroids,
        )
        archive = load_and_collapse_jsonl(
            load_path,
            archive_type=config.archive_type,
            resolution=config.grid_resolution,
            centroids_path=_centroids_path_for_archive_dir(load_path.parent, config),
        )
        if archive.filled_count() > 0:
            counters = RunCounters(
                candidates_evaluated=config.initial_random_candidates
            )
        else:
            counters = RunCounters()
        return archive, counters
    archive = create_archive(
        archive_factory_config_from_scheduler(config),
        output_dir=output_dir,
    )
    return archive, RunCounters()


def _centroids_path_for_archive_dir(
    archive_dir: Path,
    config: SchedulerConfig,
) -> Path | None:
    if config.archive_type != "cvt":
        return None
    return centroids_path_for_output(archive_dir)


def _ensure_cvt_centroids_in_output_dir(
    config: SchedulerConfig,
    *,
    output_dir: Path,
    source_centroids_path: Path | None,
) -> None:
    """Copy CVT centroids beside run output when warm-starting from an external baseline."""
    if config.archive_type != "cvt" or source_centroids_path is None:
        return
    if not source_centroids_path.is_file():
        return
    dest = centroids_path_for_output(output_dir)
    if dest.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_centroids_path, dest)


def _cli_main() -> None:
    """Delegate to package CLI (see ``python -m worldspace.illuminators``)."""
    import sys

    print(
        "Note: prefer `python -m worldspace.illuminators` " "(no import warning).\n",
        file=sys.stderr,
    )
    from worldspace.illuminators.cli import main

    main()


if __name__ == "__main__":
    _cli_main()

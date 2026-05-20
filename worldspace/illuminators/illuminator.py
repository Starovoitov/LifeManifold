"""MAP-Elites illuminator entrypoint: scheduler, archive, loop, JSONL output."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from worldspace.illuminators.archive import (
    GridArchive,
    load_and_collapse_jsonl,
)
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
        emitter: CandidateEmitter | None = None,
    ) -> MapElitesRunResult:
        """Run MAP-Elites for ``iterations × batch_size`` candidate evaluations."""
        config = load_scheduler(
            scheduler_path or DEFAULT_SCHEDULER_PATH,
            iterations_override=iterations,
        )
        if grid_resolution is not None:
            config = replace(config, grid_resolution=grid_resolution)
        effective_steps = normalize_illuminator_steps(steps, min_steps=config.min_steps)
        out_dir = Path(output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = archive_jsonl_path(out_dir)

        archive, counters = _load_archive_and_counters(
            config,
            load_archive_path=load_archive_path,
        )
        start_evaluated = counters.candidates_evaluated
        rng = np.random.default_rng(seed)
        from worldspace.surrogate import get_surrogate
        from worldspace.surrogate.buffer import SurrogateBuffer

        surrogate = get_surrogate(surrogate_config_from_scheduler(config))
        surrogate_buffer = SurrogateBuffer(
            config.surrogate_buffer_path,
            flush_every=32,
        )
        effective_emitter = emitter or MapElitesEmitter(
            scheduler=config,
            surrogate=surrogate,
        )
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
        )
        surrogate_buffer.flush()
        expected_evaluations = config.iterations * config.batch_size
        run_evaluations = counters.candidates_evaluated - start_evaluated
        if run_evaluations != expected_evaluations:
            msg = (
                f"expected {expected_evaluations} evaluations this run, "
                f"got {run_evaluations}"
            )
            raise RuntimeError(msg)
        return MapElitesRunResult(
            iterations=config.iterations,
            evaluations=expected_evaluations,
            filled_cells=archive.filled_count(),
            archive_jsonl_path=jsonl_path,
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


def _load_archive_and_counters(
    config: SchedulerConfig,
    *,
    load_archive_path: str | Path | None,
) -> tuple[GridArchive, RunCounters]:
    if load_archive_path is not None:
        archive = load_and_collapse_jsonl(
            load_archive_path,
            resolution=config.grid_resolution,
        )
        if archive.filled_count() > 0:
            counters = RunCounters(
                candidates_evaluated=config.initial_random_candidates
            )
        else:
            counters = RunCounters()
        return archive, counters
    return GridArchive(config.grid_resolution), RunCounters()

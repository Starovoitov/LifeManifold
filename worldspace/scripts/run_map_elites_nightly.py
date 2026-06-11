"""Run MAP-Elites nightly: baseline → backfill buffer → surrogate run → train."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from worldspace.illuminators.illuminator import MapElitesIlluminator
from worldspace.illuminators.nightly_report import (
    NightlyRunReport,
    build_nightly_report,
    log_nightly_report,
    write_nightly_summary,
)
from worldspace.illuminators.scheduler import (
    DEFAULT_NIGHTLY_SCHEDULER_PATH,
    DEFAULT_NIGHTLY_SURROGATE_SCHEDULER_PATH,
    DEFAULT_SCHEDULER_PATH,
    load_scheduler,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "artifacts" / "map_elites_nightly"
_BASELINE_SUBDIR = "baseline"
_SURROGATE_SUBDIR = "surrogate"
_NIGHTLY_BUFFER_PATH = _REPO_ROOT / "artifacts" / "surrogate" / "buffer_nightly.jsonl"
_NIGHTLY_CHECKPOINT_PATH = (
    _REPO_ROOT / "artifacts" / "surrogate" / "checkpoints" / "nightly_v2.pkl"
)
_NIGHTLY_TRAINING_SUMMARY_PATH = (
    _REPO_ROOT / "artifacts" / "surrogate" / "checkpoints" / "nightly_v2.summary.json"
)
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train_surrogate.py"
_DEFAULT_GRID_SIZE = 50
_DEFAULT_STEPS = 200
_DEFAULT_SEED = 0

__all__ = [
    "NightlyPipelineResult",
    "ensure_nightly_buffer_backfill",
    "main",
    "run_map_elites_nightly",
    "run_nightly_pipeline",
    "train_nightly_surrogate",
]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NightlyPipelineResult:
    """Artifacts from the default two-phase nightly job."""

    baseline: NightlyRunReport
    surrogate: NightlyRunReport
    training_summary_path: Path
    checkpoint_path: Path
    pipeline_summary_path: Path


def run_map_elites_nightly(
    *,
    scheduler_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int = _DEFAULT_SEED,
    grid_resolution: int | None = None,
    grid_size: int = _DEFAULT_GRID_SIZE,
    steps: int = _DEFAULT_STEPS,
    iterations: int | None = None,
    load_archive_path: str | Path | None = None,
) -> NightlyRunReport:
    """Execute one illuminator run and write summary artifacts (not removed after return)."""
    sched_path = Path(scheduler_path or DEFAULT_NIGHTLY_SCHEDULER_PATH)
    out_dir = Path(output_dir or _DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_scheduler(sched_path, iterations_override=iterations)

    started = time.perf_counter()
    result = MapElitesIlluminator().run(
        scheduler_path=sched_path,
        output_dir=out_dir,
        seed=seed,
        grid_resolution=grid_resolution,
        grid_size=grid_size,
        steps=steps,
        iterations=iterations,
        load_archive_path=load_archive_path,
    )
    elapsed = time.perf_counter() - started

    report = build_nightly_report(
        result=result,
        config=config,
        scheduler_path=sched_path,
        seed=seed,
        elapsed_seconds=elapsed,
        resume_archive_path=load_archive_path,
    )
    summary_path = out_dir / "nightly_run_summary.json"
    write_nightly_summary(summary_path, report)
    log_nightly_report(report)
    return report


def train_nightly_surrogate(
    *,
    buffer_path: Path | None = None,
    checkpoint_path: Path | None = None,
    summary_path: Path | None = None,
) -> Path:
    """Train LightGBM surrogate from the nightly buffer; returns training summary path."""
    buffer = buffer_path or _NIGHTLY_BUFFER_PATH
    checkpoint = checkpoint_path or _NIGHTLY_CHECKPOINT_PATH
    summary = summary_path or _NIGHTLY_TRAINING_SUMMARY_PATH
    cmd = [
        sys.executable,
        str(_TRAIN_SCRIPT),
        "--model-type",
        "lightgbm",
        "--buffer-path",
        str(buffer),
        "--checkpoint-path",
        str(checkpoint),
        "--summary-path",
        str(summary),
        "--no-quality-gate",
    ]
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    logger.info("Training nightly surrogate: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=_REPO_ROOT, env=env, check=True)
    return summary


def ensure_nightly_buffer_backfill(
    baseline_archive: Path,
    *,
    buffer_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, int] | None:
    """Backfill nightly buffer from baseline archive without destroying live_eval rows."""
    from worldspace.surrogate.backfill import (
        backfill_buffer_from_archive,
        buffer_has_archive_backfill_rows,
        buffer_has_live_eval_rows,
    )

    target_buffer = buffer_path or _NIGHTLY_BUFFER_PATH
    target_buffer.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Ensuring nightly surrogate buffer from baseline archive: %s -> %s (overwrite=%s)",
        baseline_archive,
        target_buffer,
        overwrite,
    )

    if overwrite:
        stats = backfill_buffer_from_archive(
            baseline_archive,
            target_buffer,
            overwrite=True,
        )
        logger.info("Nightly buffer backfill stats: %s", stats)
        return stats

    if buffer_has_live_eval_rows(target_buffer):
        if buffer_has_archive_backfill_rows(target_buffer):
            logger.info(
                "Buffer already has live_eval and archive_backfill rows; skipping backfill"
            )
            return None
        stats = backfill_buffer_from_archive(
            baseline_archive,
            target_buffer,
            overwrite=False,
        )
        logger.info(
            "Nightly buffer backfill stats (append, preserving live_eval): %s", stats
        )
        return stats

    stats = backfill_buffer_from_archive(
        baseline_archive,
        target_buffer,
        overwrite=True,
    )
    logger.info("Nightly buffer backfill stats (rebuild): %s", stats)
    return stats


def _write_pipeline_summary(
    path: Path,
    *,
    baseline: NightlyRunReport,
    surrogate: NightlyRunReport,
    training_summary_path: Path,
    checkpoint_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    training_payload: dict | None = None
    if training_summary_path.is_file():
        training_payload = json.loads(training_summary_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": baseline.schema_version,
        "seed": baseline.seed,
        "buffer_path": str(_NIGHTLY_BUFFER_PATH.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "training_summary": str(training_summary_path.resolve()),
        "training": training_payload,
        "baseline": {
            "scheduler": baseline.scheduler_path,
            "output_summary": str(
                (
                    Path(baseline.archive_jsonl_path).parent
                    / "nightly_run_summary.json"
                ).resolve()
            ),
            "evaluations": baseline.evaluations,
            "filled_cells": baseline.filled_cells,
            "coverage": round(baseline.coverage, 6),
            "elapsed_seconds": round(baseline.elapsed_seconds, 3),
            "surrogate_enabled": baseline.surrogate_enabled,
        },
        "surrogate_run": {
            "scheduler": surrogate.scheduler_path,
            "output_summary": str(
                (
                    Path(surrogate.archive_jsonl_path).parent
                    / "nightly_run_summary.json"
                ).resolve()
            ),
            "evaluations": surrogate.evaluations,
            "filled_cells": surrogate.filled_cells,
            "coverage": round(surrogate.coverage, 6),
            "elapsed_seconds": round(surrogate.elapsed_seconds, 3),
            "surrogate_enabled": surrogate.surrogate_enabled,
            "resumed_from": baseline.archive_jsonl_path,
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


def run_nightly_pipeline(
    *,
    output_dir: str | Path | None = None,
    seed: int = _DEFAULT_SEED,
    grid_resolution: int | None = None,
    grid_size: int = _DEFAULT_GRID_SIZE,
    steps: int = _DEFAULT_STEPS,
    iterations: int | None = None,
    skip_training: bool = False,
    overwrite_buffer: bool = False,
) -> NightlyPipelineResult:
    """Baseline MAP-Elites → backfill buffer → surrogate run → train surrogate."""
    root = Path(output_dir or _DEFAULT_OUTPUT_DIR)
    baseline_dir = root / _BASELINE_SUBDIR
    surrogate_dir = root / _SURROGATE_SUBDIR

    logger.info("Nightly step 1/4: baseline (surrogate disabled)")
    baseline = run_map_elites_nightly(
        scheduler_path=DEFAULT_NIGHTLY_SCHEDULER_PATH,
        output_dir=baseline_dir,
        seed=seed,
        grid_resolution=grid_resolution,
        grid_size=grid_size,
        steps=steps,
        iterations=iterations,
    )

    logger.info("Nightly step 2/4: backfill surrogate buffer from baseline archive")
    ensure_nightly_buffer_backfill(
        Path(baseline.archive_jsonl_path),
        overwrite=overwrite_buffer,
    )

    logger.info("Nightly step 3/4: surrogate-enabled run (resume baseline archive)")
    surrogate = run_map_elites_nightly(
        scheduler_path=DEFAULT_NIGHTLY_SURROGATE_SCHEDULER_PATH,
        output_dir=surrogate_dir,
        seed=seed,
        grid_resolution=grid_resolution,
        grid_size=grid_size,
        steps=steps,
        iterations=iterations,
        load_archive_path=baseline.archive_jsonl_path,
    )

    training_summary = _NIGHTLY_TRAINING_SUMMARY_PATH
    if not skip_training:
        logger.info(
            "Nightly step 4/4: train surrogate on full buffer -> %s",
            _NIGHTLY_CHECKPOINT_PATH,
        )
        training_summary = train_nightly_surrogate()
    else:
        logger.info("Skipping surrogate training (--skip-training)")

    pipeline_summary = root / "nightly_pipeline_summary.json"
    _write_pipeline_summary(
        pipeline_summary,
        baseline=baseline,
        surrogate=surrogate,
        training_summary_path=training_summary,
        checkpoint_path=_NIGHTLY_CHECKPOINT_PATH,
    )
    logger.info("Wrote pipeline summary: %s", pipeline_summary)
    return NightlyPipelineResult(
        baseline=baseline,
        surrogate=surrogate,
        training_summary_path=training_summary,
        checkpoint_path=_NIGHTLY_CHECKPOINT_PATH,
        pipeline_summary_path=pipeline_summary,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI for scheduled or manual nightly MAP-Elites runs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Default: nightly pipeline (baseline → backfill buffer → surrogate run → "
            "train). Each phase uses 650 iterations (steps=200) unless overridden. "
            "Use --single-run for one phase only."
        ),
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run only one illuminator pass (see --scheduler).",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=str(DEFAULT_NIGHTLY_SCHEDULER_PATH),
        help=(
            "Scheduler YAML for --single-run (default: nightly baseline). "
            f"Production: {DEFAULT_SCHEDULER_PATH}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Root output directory (pipeline uses baseline/ and surrogate/ subdirs).",
    )
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=None,
        help="Archive resolution override (default: from YAML).",
    )
    parser.add_argument("--grid", type=int, default=_DEFAULT_GRID_SIZE)
    parser.add_argument("--steps", type=int, default=_DEFAULT_STEPS)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override YAML iterations (nightly default: 650 per phase, ~3h pipeline).",
    )
    parser.add_argument(
        "--load-archive",
        type=str,
        default="",
        help="Resume archive for --single-run only.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Pipeline only: skip train step (requires existing nightly_v2.pkl).",
    )
    parser.add_argument(
        "--overwrite-buffer",
        action="store_true",
        help="Rebuild nightly surrogate buffer from baseline archive (drops existing rows).",
    )
    args = parser.parse_args(argv)

    if args.single_run:
        load_path = args.load_archive.strip() or None
        run_map_elites_nightly(
            scheduler_path=args.scheduler,
            output_dir=args.output_dir,
            seed=args.seed,
            grid_resolution=args.grid_resolution,
            grid_size=args.grid,
            steps=args.steps,
            iterations=args.iterations,
            load_archive_path=load_path,
        )
        return

    run_nightly_pipeline(
        output_dir=args.output_dir,
        seed=args.seed,
        grid_resolution=args.grid_resolution,
        grid_size=args.grid,
        steps=args.steps,
        iterations=args.iterations,
        skip_training=args.skip_training,
        overwrite_buffer=args.overwrite_buffer,
    )


if __name__ == "__main__":
    main()

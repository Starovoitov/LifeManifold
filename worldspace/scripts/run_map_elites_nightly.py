"""Run a MAP-Elites nightly job (reduced default iterations; optional full budget)."""

from __future__ import annotations

import argparse
import logging
import time
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
    DEFAULT_SCHEDULER_PATH,
    load_scheduler,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "artifacts" / "map_elites_nightly"
_DEFAULT_GRID_SIZE = 50
_DEFAULT_STEPS = 300
_DEFAULT_SEED = 0

__all__ = ["main", "run_map_elites_nightly"]


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
    """Execute illuminator run and write summary artifacts (not removed after return)."""
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
    )
    summary_path = out_dir / "nightly_run_summary.json"
    write_nightly_summary(summary_path, report)
    log_nightly_report(report)
    return report


def main(argv: list[str] | None = None) -> None:
    """CLI for scheduled or manual nightly MAP-Elites runs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Run MAP-Elites with the nightly scheduler (default 100 iterations; "
            "use production YAML and --iterations 10000 for full 500k budget)."
        ),
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=str(DEFAULT_NIGHTLY_SCHEDULER_PATH),
        help=(
            "Scheduler YAML (default: map_elites_scheduler_nightly.yaml). "
            f"Full production: {DEFAULT_SCHEDULER_PATH}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Directory for map_elites_archive.jsonl and nightly_run_summary.json",
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
        help="Override YAML iterations (full production: 10000).",
    )
    parser.add_argument(
        "--load-archive",
        type=str,
        default="",
        help="Optional existing archive JSONL to resume from.",
    )
    args = parser.parse_args(argv)
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


if __name__ == "__main__":
    main()

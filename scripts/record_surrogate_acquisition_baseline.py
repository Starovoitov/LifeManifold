"""Backfill buffer from nightly archive, train checkpoint, write baseline manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from worldspace.illuminators.archive import (  # noqa: E402
    count_archive_jsonl_lines,
    load_and_collapse_jsonl,
)
from worldspace.illuminators.illuminator import MapElitesIlluminator  # noqa: E402
from worldspace.illuminators.scheduler import (  # noqa: E402
    DEFAULT_NIGHTLY_SCHEDULER_PATH,
    load_scheduler,
)
from worldspace.surrogate.backfill import (  # noqa: E402
    backfill_buffer_from_collapsed_archive,
)
from worldspace.surrogate.evaluation import (  # noqa: E402
    MIN_TRAIN_SAMPLES_FULL,
    QUALITY_MAE_FITNESS_MAX,
    QUALITY_MAE_STABILITY_MAX,
    QUALITY_R2_FITNESS_MIN,
)

_DEFAULT_ARCHIVE = (
    _REPO_ROOT / "artifacts" / "map_elites_nightly" / "map_elites_archive.jsonl"
)
_DEFAULT_SURROGATE_ROOT = _REPO_ROOT / "artifacts" / "surrogate"
_DEFAULT_BUFFER = _DEFAULT_SURROGATE_ROOT / "buffer_nightly.jsonl"
_DEFAULT_CHECKPOINT = _DEFAULT_SURROGATE_ROOT / "checkpoints" / "latest.pkl"
_DEFAULT_MANIFEST = _DEFAULT_SURROGATE_ROOT / "baseline_manifest.json"
_DEFAULT_BASELINE_OUT = _DEFAULT_SURROGATE_ROOT / "baseline"
_ACQUISITION_BASELINE_SCHEDULER = (
    _REPO_ROOT
    / "worldspace"
    / "specs"
    / "map_elites_scheduler_surrogate_acquisition_baseline.yaml"
)
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train_surrogate.py"
_MANIFEST_SCHEMA = "1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record Surrogate Acquisition baseline",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=_DEFAULT_ARCHIVE,
        help="MAP-Elites archive JSONL (nightly artifact)",
    )
    parser.add_argument(
        "--buffer-path",
        type=Path,
        default=_DEFAULT_BUFFER,
        help="Output surrogate training buffer",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=_DEFAULT_CHECKPOINT,
        help="Output surrogate checkpoint",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="Output baseline manifest JSON",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only backfill buffer and write manifest (checkpoint must exist)",
    )
    parser.add_argument(
        "--skip-mini-run",
        action="store_true",
        help="Skip surrogate-enabled mini illuminator baseline run",
    )
    parser.add_argument(
        "--allow-quality-fail",
        action="store_true",
        help="Pass --no-quality-gate to train when hold-out misses MVP thresholds",
    )
    parser.add_argument(
        "--all-archive-lines",
        action="store_true",
        help="Backfill every JSONL line (default: one row per collapsed bin)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for optional mini baseline run",
    )
    return parser.parse_args()


def run_train(
    *,
    buffer_path: Path,
    checkpoint_path: Path,
    summary_path: Path,
    allow_quality_fail: bool,
) -> None:
    cmd = [
        sys.executable,
        str(_TRAIN_SCRIPT),
        "--model-type",
        "lightgbm",
        "--buffer-path",
        str(buffer_path),
        "--checkpoint-path",
        str(checkpoint_path),
        "--summary-path",
        str(summary_path),
    ]
    if allow_quality_fail:
        cmd.append("--no-quality-gate")
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    subprocess.run(cmd, cwd=_REPO_ROOT, env=env, check=True)


def run_mini_baseline(*, output_dir: Path, seed: int) -> dict[str, object]:
    if not _ACQUISITION_BASELINE_SCHEDULER.is_file():
        msg = f"Surrogate Acquisition baseline scheduler missing: {_ACQUISITION_BASELINE_SCHEDULER}"
        raise FileNotFoundError(msg)
    config = load_scheduler(_ACQUISITION_BASELINE_SCHEDULER)
    result = MapElitesIlluminator().run(
        scheduler_path=_ACQUISITION_BASELINE_SCHEDULER,
        output_dir=output_dir,
        seed=seed,
    )
    return {
        "scheduler_path": str(_ACQUISITION_BASELINE_SCHEDULER.resolve()),
        "output_dir": str(output_dir.resolve()),
        "archive_jsonl": str(result.archive_jsonl_path.resolve()),
        "evaluations": result.evaluations,
        "filled_cells": result.filled_cells,
        "iterations": result.iterations,
        "surrogate_enabled": config.surrogate_enabled,
        "llm_enabled": config.llm_enabled,
    }


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def write_manifest(
    path: Path,
    *,
    archive_path: Path,
    buffer_path: Path,
    checkpoint_path: Path,
    summary_path: Path,
    backfill_stats: dict[str, int],
    training_summary: dict | None,
    nightly_scheduler_path: Path,
    mini_run: dict[str, object] | None,
    seed: int,
) -> None:
    nightly_config = load_scheduler(nightly_scheduler_path)
    collapsed = load_and_collapse_jsonl(
        archive_path, resolution=nightly_config.grid_resolution
    )
    archive_lines = count_archive_jsonl_lines(archive_path)
    holdout = (training_summary or {}).get("holdout_metrics") or {}
    payload = {
        "schema_version": _MANIFEST_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "acquisition_mode": "off",
        "nightly_reference": {
            "description": "MAP-Elites archive used as historical baseline and buffer source",
            "scheduler_path": str(nightly_scheduler_path.resolve()),
            "archive_jsonl": str(archive_path.resolve()),
            "seed_documented": seed,
            "surrogate_enabled_at_run": nightly_config.surrogate_enabled,
            "llm_enabled_at_run": nightly_config.llm_enabled,
            "iterations": nightly_config.iterations,
            "batch_size": nightly_config.batch_size,
            "grid_resolution": nightly_config.grid_resolution,
            "jsonl_raw_lines": archive_lines,
            "collapsed_filled_cells": collapsed.filled_count(),
        },
        "surrogate_training": {
            "backfill_mode": (
                "all_archive_lines"
                if backfill_stats.get("collapsed_filled_cells") is None
                else "collapsed_bins"
            ),
            "buffer_path": str(buffer_path.resolve()),
            "buffer_rows": backfill_stats["buffer_rows_written"],
            "checkpoint_path": str(checkpoint_path.resolve()),
            "summary_path": (
                str(summary_path.resolve()) if summary_path.is_file() else None
            ),
            "holdout_metrics": holdout,
            "quality_thresholds": {
                "r2_fitness_min": QUALITY_R2_FITNESS_MIN,
                "mae_fitness_max": QUALITY_MAE_FITNESS_MAX,
                "mae_stability_max": QUALITY_MAE_STABILITY_MAX,
            },
            "quality_passed": (training_summary or {}).get("quality_passed"),
            "min_samples_required": MIN_TRAIN_SAMPLES_FULL,
        },
        "acquisition_baseline_run": mini_run,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    archive_path = args.archive_path.expanduser()
    buffer_path = args.buffer_path.expanduser()
    checkpoint_path = args.checkpoint_path.expanduser()
    summary_path = checkpoint_path.with_name(checkpoint_path.stem + ".summary.json")
    manifest_path = args.manifest_path.expanduser()

    nightly_config = load_scheduler(DEFAULT_NIGHTLY_SCHEDULER_PATH)
    print(f"Backfilling buffer from {archive_path} ...")
    if args.all_archive_lines:
        from worldspace.surrogate.backfill import backfill_buffer_from_archive

        backfill_stats = backfill_buffer_from_archive(archive_path, buffer_path)
    else:
        backfill_stats = backfill_buffer_from_collapsed_archive(
            archive_path,
            buffer_path,
            resolution=nightly_config.grid_resolution,
        )
    print(backfill_stats)

    if backfill_stats["buffer_rows_written"] < MIN_TRAIN_SAMPLES_FULL:
        raise SystemExit(
            f"Need at least {MIN_TRAIN_SAMPLES_FULL} buffer rows after backfill, "
            f"got {backfill_stats['buffer_rows_written']}"
        )

    training_summary: dict | None = None
    if not args.skip_train:
        print(f"Training checkpoint -> {checkpoint_path} ...")
        allow_fail = args.allow_quality_fail
        try:
            run_train(
                buffer_path=buffer_path,
                checkpoint_path=checkpoint_path,
                summary_path=summary_path,
                allow_quality_fail=allow_fail,
            )
        except subprocess.CalledProcessError:
            if not allow_fail:
                print(
                    "Hold-out quality gate failed; retry with --allow-quality-fail "
                    "or improve buffer/archive data.",
                    file=sys.stderr,
                )
            raise
        training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    elif not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint missing: {checkpoint_path} (remove --skip-train)")

    dashboard_buffer = _DEFAULT_SURROGATE_ROOT / "buffer.jsonl"
    if dashboard_buffer.resolve() != buffer_path.resolve():
        dashboard_buffer.write_bytes(buffer_path.read_bytes())
        print(f"Synced dashboard buffer -> {dashboard_buffer}")

    mini_run: dict[str, object] | None = None
    if not args.skip_mini_run:
        print(f"Mini baseline run (surrogate enabled) -> {_DEFAULT_BASELINE_OUT} ...")
        mini_run = run_mini_baseline(output_dir=_DEFAULT_BASELINE_OUT, seed=args.seed)

    write_manifest(
        manifest_path,
        archive_path=archive_path,
        buffer_path=buffer_path,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
        backfill_stats=backfill_stats,
        training_summary=training_summary,
        nightly_scheduler_path=DEFAULT_NIGHTLY_SCHEDULER_PATH,
        mini_run=mini_run,
        seed=args.seed,
    )
    print(f"Wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()

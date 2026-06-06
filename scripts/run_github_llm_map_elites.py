"""MAP-Elites run for GitHub Actions: Qwen LLM + surrogate checkpoint from last nightly."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from worldspace.illuminators.illuminator import MapElitesIlluminator
from worldspace.illuminators.nightly_report import (
    build_nightly_report,
    log_nightly_report,
    write_nightly_summary,
)
from worldspace.illuminators.scheduler import (
    DEFAULT_GITHUB_LLM_SCHEDULER_PATH,
    DEFAULT_QWEN_LLM_SPEC_PATH,
    load_scheduler,
)
from worldspace.scripts.run_map_elites_nightly import (
    _NIGHTLY_BUFFER_PATH,
    _NIGHTLY_CHECKPOINT_PATH,
    train_nightly_surrogate,
)

_NIGHTLY_ROOT = _REPO_ROOT / "artifacts" / "map_elites_nightly"
_BASELINE_SUBDIR = "baseline"

_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "artifacts" / "map_elites_github_llm"
_DEFAULT_GRID_RESOLUTION = 50
_DEFAULT_GRID_SIZE = 50
_DEFAULT_STEPS = 200
_DEFAULT_SEED = 0

__all__ = [
    "main",
    "resolve_llm_spec_path",
    "resolve_nightly_grid_resolution",
    "resolve_nightly_resume_archive",
]


logger = logging.getLogger(__name__)


def resolve_llm_spec_path(provider: str) -> Path:
    """Map provider name to a spec under ``worldspace/specs/``."""
    name = provider.strip().lower()
    if name == "qwen":
        return DEFAULT_QWEN_LLM_SPEC_PATH
    candidate = _REPO_ROOT / "worldspace" / "specs" / f"llm_world_generator_{name}.yaml"
    if candidate.is_file():
        return candidate
    msg = (
        f"unknown LLM provider {provider!r}; use qwen or add {candidate.name} "
        f"(default: {DEFAULT_QWEN_LLM_SPEC_PATH.name})"
    )
    raise FileNotFoundError(msg)


def resolve_nightly_resume_archive() -> Path | None:
    """Baseline archive from a downloaded or local nightly pipeline."""
    baseline = _NIGHTLY_ROOT / _BASELINE_SUBDIR / "map_elites_archive.jsonl"
    if baseline.is_file():
        return baseline
    return None


def resolve_nightly_grid_resolution(archive_path: Path | str) -> int | None:
    """``grid_resolution`` from ``nightly_run_summary.json`` beside the archive."""
    summary_path = Path(archive_path).parent / "nightly_run_summary.json"
    if not summary_path.is_file():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    raw = payload.get("grid_resolution")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def ensure_nightly_surrogate_checkpoint(*, train_if_missing: bool) -> Path:
    """Require ``nightly_v2.pkl``; train (and backfill buffer) when missing if allowed."""
    checkpoint = _NIGHTLY_CHECKPOINT_PATH
    if checkpoint.is_file():
        return checkpoint
    if not train_if_missing:
        msg = (
            f"surrogate checkpoint not found: {checkpoint}. "
            "Download map-elites-nightly artifacts or run with --train-surrogate-if-missing."
        )
        raise FileNotFoundError(msg)
    if not _ensure_nightly_buffer_from_baseline():
        baseline = _NIGHTLY_ROOT / _BASELINE_SUBDIR / "map_elites_archive.jsonl"
        if _NIGHTLY_BUFFER_PATH.is_file() and _NIGHTLY_BUFFER_PATH.stat().st_size == 0:
            detail = f"buffer is empty at {_NIGHTLY_BUFFER_PATH}"
        elif not baseline.is_file():
            detail = f"no baseline archive at {baseline}"
        else:
            detail = "buffer backfill produced no rows"
        msg = (
            f"cannot train surrogate: {detail}. "
            "Run MAP-Elites nightly first or download map-elites-nightly artifacts."
        )
        raise FileNotFoundError(msg)
    logger.info("Training nightly surrogate from %s", _NIGHTLY_BUFFER_PATH)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    train_nightly_surrogate()
    if not checkpoint.is_file():
        msg = f"training finished but checkpoint missing: {checkpoint}"
        raise FileNotFoundError(msg)
    return checkpoint


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "MAP-Elites with LLM emitter (Qwen by default) and nightly surrogate "
            "checkpoint for CI / workflow_dispatch."
        ),
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=str(DEFAULT_GITHUB_LLM_SCHEDULER_PATH),
        help="Scheduler YAML (default: map_elites_scheduler_github_llm.yaml).",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="qwen",
        help="LLM provider key (default: qwen → llm_world_generator_qwen.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Run output directory.",
    )
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=_DEFAULT_GRID_RESOLUTION,
        help="Archive grid side length (default: 50, same as nightly).",
    )
    parser.add_argument("--grid", type=int, default=_DEFAULT_GRID_SIZE)
    parser.add_argument("--steps", type=int, default=_DEFAULT_STEPS)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument(
        "--load-archive",
        type=str,
        default="",
        help="Resume archive JSONL (default: nightly baseline if present).",
    )
    parser.add_argument(
        "--no-resume-nightly",
        action="store_true",
        help="Do not resume from artifacts/map_elites_nightly/baseline/.",
    )
    parser.add_argument(
        "--train-surrogate-if-missing",
        action="store_true",
        help=(
            "Train nightly_v2.pkl when checkpoint is absent; backfill buffer from "
            "nightly baseline archive if buffer_nightly.jsonl is missing."
        ),
    )
    args = parser.parse_args(argv)

    ensure_nightly_surrogate_checkpoint(
        train_if_missing=args.train_surrogate_if_missing
    )
    llm_spec = resolve_llm_spec_path(args.llm_provider)

    load_archive: str | Path | None = None
    if args.load_archive.strip():
        load_archive = args.load_archive.strip()
    elif not args.no_resume_nightly:
        load_archive = resolve_nightly_resume_archive()

    sched_path = Path(args.scheduler)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_scheduler(sched_path, iterations_override=args.iterations)

    import time

    started = time.perf_counter()
    result = MapElitesIlluminator().run(
        scheduler_path=sched_path,
        output_dir=out_dir,
        seed=args.seed,
        grid_resolution=args.grid_resolution,
        grid_size=args.grid,
        steps=args.steps,
        iterations=args.iterations,
        load_archive_path=load_archive,
        llm_spec_path=llm_spec,
    )
    elapsed = time.perf_counter() - started

    report = build_nightly_report(
        result=result,
        config=config,
        scheduler_path=sched_path,
        seed=args.seed,
        elapsed_seconds=elapsed,
        resume_archive_path=load_archive,
    )
    summary_path = out_dir / "nightly_run_summary.json"
    write_nightly_summary(summary_path, report)
    log_nightly_report(report)
    logger.info("LLM MAP-Elites run complete: %s", result.archive_jsonl_path)


def _nightly_buffer_has_rows() -> bool:
    """True when the nightly buffer file exists and is non-empty."""
    return _NIGHTLY_BUFFER_PATH.is_file() and _NIGHTLY_BUFFER_PATH.stat().st_size > 0


def _ensure_nightly_buffer_from_baseline() -> bool:
    """Backfill ``buffer_nightly.jsonl`` from the nightly baseline archive if needed."""
    if _nightly_buffer_has_rows():
        return True
    baseline = resolve_nightly_resume_archive()
    if baseline is None:
        return False
    from worldspace.surrogate.backfill import backfill_buffer_from_archive

    _NIGHTLY_BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Backfilling surrogate buffer from nightly baseline: %s", baseline)
    stats = backfill_buffer_from_archive(
        baseline,
        _NIGHTLY_BUFFER_PATH,
        overwrite=True,
    )
    logger.info("Buffer backfill stats: %s", stats)
    return _NIGHTLY_BUFFER_PATH.is_file() and _NIGHTLY_BUFFER_PATH.stat().st_size > 0


if __name__ == "__main__":
    main()

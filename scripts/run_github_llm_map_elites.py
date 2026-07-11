"""MAP-Elites run for GitHub Actions: Qwen LLM + surrogate checkpoint from last nightly."""

from __future__ import annotations

import argparse
import json
import logging
import os
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
from worldspace.surrogate.checkpoint_quality import checkpoint_quality_allows_hints

_NIGHTLY_ROOT = _REPO_ROOT / "artifacts" / "map_elites_nightly"
_BASELINE_ARCHIVE_NAME = "map_elites_archive.jsonl"
_GRID_BASELINE_LEGACY_SUBDIR = Path("baseline")
_CVT_BASELINE_SUBDIR = Path("cvt") / "baseline"

_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "artifacts" / "map_elites_github_llm"
_DEFAULT_GRID_RESOLUTION = 50
_DEFAULT_GRID_SIZE = 50
_DEFAULT_STEPS = 200
_DEFAULT_SEED = 0
_QUALITY_GATE_ENV = "SURROGATE_REQUIRE_QUALITY_GATE"
from worldspace.surrogate.checkpoint_paths import STUB_CHECKPOINT_SENTINEL

__all__ = [
    "main",
    "resolve_baseline_archive_for_scheduler",
    "resolve_effective_surrogate_checkpoint",
    "resolve_llm_spec_path",
    "resolve_nightly_baseline_archive",
    "resolve_nightly_grid_resolution",
    "resolve_nightly_resume_archive",
    "resolve_surrogate_quality_gate",
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


def resolve_nightly_baseline_archive(archive_type: str) -> Path | None:
    """Baseline archive for ``grid`` or ``cvt`` under the nightly artifact root."""
    typed = _NIGHTLY_ROOT / archive_type / "baseline" / _BASELINE_ARCHIVE_NAME
    if typed.is_file():
        return typed
    if archive_type == "grid":
        legacy = _NIGHTLY_ROOT / _GRID_BASELINE_LEGACY_SUBDIR / _BASELINE_ARCHIVE_NAME
        if legacy.is_file():
            return legacy
    return None


def resolve_baseline_archive_for_scheduler(scheduler_path: str | Path) -> Path:
    """Return the nightly baseline archive matching the scheduler ``archive_type``."""
    sched_path = Path(scheduler_path)
    config = load_scheduler(sched_path)
    baseline = resolve_nightly_baseline_archive(config.archive_type)
    if baseline is None:
        msg = (
            f"no {config.archive_type} baseline archive for scheduler {sched_path}; "
            f"expected {_NIGHTLY_ROOT / config.archive_type / 'baseline' / _BASELINE_ARCHIVE_NAME}"
        )
        raise FileNotFoundError(msg)
    return baseline


def resolve_nightly_resume_archive() -> Path | None:
    """Default resume archive for GitHub LLM runs (CVT nightly baseline)."""
    return resolve_nightly_baseline_archive("cvt")


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


def resolve_surrogate_quality_gate(*, cli_flag: bool | None = None) -> bool:
    """Resolve whether runtime LLM hints require a gated checkpoint summary."""
    if cli_flag is not None:
        return cli_flag
    raw = os.environ.get(_QUALITY_GATE_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def resolve_effective_surrogate_checkpoint(
    checkpoint: Path,
    *,
    override: Path | None,
    require_quality_gate: bool,
    allow_ungated: bool,
) -> Path | None:
    """Return checkpoint path for runtime hints, or ``None`` to force stub values."""
    candidate = override or checkpoint
    if not candidate.is_file():
        return None
    if require_quality_gate and not allow_ungated:
        if not checkpoint_quality_allows_hints(candidate):
            logger.warning(
                "Surrogate checkpoint failed quality gate; LLM hints will use stub: %s",
                candidate,
            )
            return None
    return candidate


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
        baseline = resolve_nightly_baseline_archive("cvt")
        if _NIGHTLY_BUFFER_PATH.is_file() and _NIGHTLY_BUFFER_PATH.stat().st_size == 0:
            detail = f"buffer is empty at {_NIGHTLY_BUFFER_PATH}"
        elif baseline is None:
            detail = (
                f"no CVT baseline archive under "
                f"{_NIGHTLY_ROOT / 'cvt' / 'baseline' / _BASELINE_ARCHIVE_NAME}"
            )
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
        "--replicate",
        type=int,
        default=None,
        help="Optional within-seed replicate index (q1-repeat variance floor runs).",
    )
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
        help="Do not resume from artifacts/map_elites_nightly/cvt/baseline/.",
    )
    parser.add_argument(
        "--train-surrogate-if-missing",
        action="store_true",
        help=(
            "Train nightly_v2.pkl when checkpoint is absent; backfill buffer from "
            "nightly baseline archive if buffer_nightly.jsonl is missing."
        ),
    )
    parser.add_argument(
        "--surrogate-checkpoint",
        type=str,
        default="",
        help=(
            "Override surrogate checkpoint path. When quality gate is required, "
            "the override must pass nightly_v2.summary.json checks."
        ),
    )
    parser.add_argument(
        "--require-surrogate-quality-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Require quality_passed in checkpoint summary for LLM hints. "
            f"Default: env {_QUALITY_GATE_ENV} or false."
        ),
    )
    parser.add_argument(
        "--allow-ungated-checkpoint",
        action="store_true",
        help="Use checkpoint for LLM hints even when quality gate fails (local only).",
    )
    args = parser.parse_args(argv)

    checkpoint = ensure_nightly_surrogate_checkpoint(
        train_if_missing=args.train_surrogate_if_missing
    )
    require_quality_gate = resolve_surrogate_quality_gate(
        cli_flag=args.require_surrogate_quality_gate,
    )
    if args.allow_ungated_checkpoint:
        require_quality_gate = False
    override = (
        Path(args.surrogate_checkpoint.strip())
        if args.surrogate_checkpoint.strip()
        else None
    )
    effective_checkpoint = resolve_effective_surrogate_checkpoint(
        checkpoint,
        override=override,
        require_quality_gate=require_quality_gate,
        allow_ungated=args.allow_ungated_checkpoint,
    )
    checkpoint_override = (
        STUB_CHECKPOINT_SENTINEL
        if effective_checkpoint is None
        else str(effective_checkpoint)
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

    grid_resolution = args.grid_resolution
    if config.archive_type == "cvt":
        grid_resolution = None

    started = time.perf_counter()
    result = MapElitesIlluminator().run(
        scheduler_path=sched_path,
        output_dir=out_dir,
        seed=args.seed,
        grid_resolution=grid_resolution,
        grid_size=args.grid,
        steps=args.steps,
        iterations=args.iterations,
        load_archive_path=load_archive,
        llm_spec_path=llm_spec,
        require_surrogate_quality_gate=require_quality_gate,
        surrogate_checkpoint_override=checkpoint_override,
    )
    elapsed = time.perf_counter() - started

    report = build_nightly_report(
        result=result,
        config=config,
        scheduler_path=sched_path,
        seed=args.seed,
        elapsed_seconds=elapsed,
        replicate=args.replicate,
        resume_archive_path=load_archive,
        llm_spec_path=llm_spec,
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

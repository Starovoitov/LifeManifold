"""Post-run validation and summary for MAP-Elites nightly jobs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from worldspace.illuminators.archive import (
    ARCHIVE_SCHEMA_VERSION,
    count_archive_jsonl_lines,
    load_and_collapse_jsonl,
    merge_archives,
)
from worldspace.illuminators.archive_protocol import ArchiveProtocol
from worldspace.illuminators.cvt import centroids_path_for_output
from worldspace.illuminators.illuminator import MapElitesRunResult
from worldspace.illuminators.scheduler import RunCounters, SchedulerConfig

logger = logging.getLogger(__name__)

LLM_STACK_VERSION = "v2"

__all__ = [
    "LLM_STACK_VERSION",
    "LlmRunInfo",
    "NightlyRunReport",
    "build_llm_run_info",
    "build_nightly_report",
    "log_nightly_report",
    "write_nightly_summary",
]


@dataclass(frozen=True)
class LlmRunInfo:
    """LLM stack metadata for nightly summaries (stack v2 audit)."""

    stack_version: str
    model: str | None
    max_tokens: int | None
    llm_parallel_emit: bool
    llm_parallel_workers: int | None
    prompt_version: str | None


@dataclass(frozen=True)
class NightlyRunReport:
    """Metrics collected after a nightly MAP-Elites run."""

    schema_version: str
    scheduler_path: str
    seed: int
    iterations: int
    evaluations: int
    filled_cells: int
    grid_resolution: int
    archive_type: str
    n_cells: int
    coverage: float
    jsonl_raw_lines: int
    jsonl_collapsed_cells: int
    elapsed_seconds: float
    llm_enabled: bool
    surrogate_enabled: bool
    archive_jsonl_path: str
    llm_stack_version: str | None = None
    llm_model: str | None = None
    max_tokens: int | None = None
    llm_parallel_emit: bool | None = None
    llm_parallel_workers: int | None = None
    prompt_version: str | None = None
    llm_emit_attempts: int | None = None
    llm_emit_fallbacks: int | None = None
    llm_fallback_rate_pct: float | None = None
    emit_llm_seconds: float | None = None
    eval_seconds: float | None = None


def build_llm_run_info(
    config: SchedulerConfig,
    *,
    llm_spec_path: str | Path | None = None,
) -> LlmRunInfo | None:
    """Build LLM stack metadata when the scheduler has LLM emit enabled."""
    if not config.llm_enabled:
        return None
    from worldspace.generators.llm_config import load_llm_config
    from worldspace.illuminators.emitters.llm_prompts import emitter_prompt_version

    llm_cfg = load_llm_config(llm_spec_path)
    provider = llm_cfg.providers.get(llm_cfg.active_provider, {})
    model_raw = provider.get("model")
    model = str(model_raw) if model_raw is not None else llm_cfg.active_provider
    max_llm_slots = sum(1 for kind in config.batch_emitters if kind == "llm")
    parallel_workers: int | None = None
    if config.performance.llm_parallel_emit and max_llm_slots >= 1:
        parallel_workers = max_llm_slots
        if config.performance.llm_parallel_workers > 0:
            parallel_workers = min(
                parallel_workers,
                config.performance.llm_parallel_workers,
            )
        parallel_workers = max(1, parallel_workers)
    return LlmRunInfo(
        stack_version=LLM_STACK_VERSION,
        model=model,
        max_tokens=llm_cfg.max_tokens,
        llm_parallel_emit=config.performance.llm_parallel_emit,
        llm_parallel_workers=parallel_workers,
        prompt_version=emitter_prompt_version(archive_type=config.archive_type),
    )


def _llm_fallback_rate_pct(counters: RunCounters) -> float | None:
    if counters.llm_emit_attempts <= 0:
        return None
    return round(
        100.0 * counters.llm_emit_fallbacks / counters.llm_emit_attempts,
        6,
    )


def build_nightly_report(
    *,
    result: MapElitesRunResult,
    config: SchedulerConfig,
    scheduler_path: str | Path,
    seed: int,
    elapsed_seconds: float,
    resume_archive_path: str | Path | None = None,
    llm_spec_path: str | Path | None = None,
) -> NightlyRunReport:
    """Validate on-disk JSONL and compute fill metrics."""
    jsonl_path = result.archive_jsonl_path
    n_cells = config.n_cells
    raw_lines = count_archive_jsonl_lines(
        jsonl_path,
        resolution=config.grid_resolution,
    )
    run_only = _load_collapsed_archive(jsonl_path, config=config)
    collapsed = _collapsed_archive_for_validation(
        jsonl_path,
        config=config,
        resume_archive_path=resume_archive_path,
    )
    collapsed_cells = collapsed.filled_count()
    if collapsed_cells != result.filled_cells:
        msg = (
            f"filled_cells mismatch: run reported {result.filled_cells}, "
            f"collapsed archive has {collapsed_cells}"
        )
        raise RuntimeError(msg)
    if collapsed_cells > n_cells:
        msg = f"collapsed cells {collapsed_cells} exceed archive capacity {n_cells}"
        raise RuntimeError(msg)
    run_only_cells = run_only.filled_count()
    if raw_lines < run_only_cells:
        msg = "raw JSONL line count must be >= collapsed cell count for this run"
        raise RuntimeError(msg)
    coverage = float(collapsed_cells) / float(n_cells) if n_cells else 0.0
    llm_info = build_llm_run_info(config, llm_spec_path=llm_spec_path)
    counters = result.counters
    return NightlyRunReport(
        schema_version=ARCHIVE_SCHEMA_VERSION,
        scheduler_path=str(Path(scheduler_path).resolve()),
        seed=int(seed),
        iterations=result.iterations,
        evaluations=result.evaluations,
        filled_cells=collapsed_cells,
        grid_resolution=config.grid_resolution,
        archive_type=config.archive_type,
        n_cells=n_cells,
        coverage=coverage,
        jsonl_raw_lines=raw_lines,
        jsonl_collapsed_cells=collapsed_cells,
        elapsed_seconds=float(elapsed_seconds),
        llm_enabled=config.llm_enabled,
        surrogate_enabled=config.surrogate_enabled,
        archive_jsonl_path=str(jsonl_path.resolve()),
        llm_stack_version=llm_info.stack_version if llm_info is not None else None,
        llm_model=llm_info.model if llm_info is not None else None,
        max_tokens=llm_info.max_tokens if llm_info is not None else None,
        llm_parallel_emit=llm_info.llm_parallel_emit if llm_info is not None else None,
        llm_parallel_workers=(
            llm_info.llm_parallel_workers if llm_info is not None else None
        ),
        prompt_version=llm_info.prompt_version if llm_info is not None else None,
        llm_emit_attempts=counters.llm_emit_attempts if config.llm_enabled else None,
        llm_emit_fallbacks=counters.llm_emit_fallbacks if config.llm_enabled else None,
        llm_fallback_rate_pct=(
            _llm_fallback_rate_pct(counters) if config.llm_enabled else None
        ),
        emit_llm_seconds=(
            round(counters.emit_llm_seconds, 3) if config.llm_enabled else None
        ),
        eval_seconds=round(counters.eval_seconds, 3) if config.llm_enabled else None,
    )


def write_nightly_summary(path: str | Path, report: NightlyRunReport) -> None:
    """Write ``nightly_run_summary.json`` next to the archive JSONL."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": report.schema_version,
        "scheduler": report.scheduler_path,
        "seed": report.seed,
        "iterations": report.iterations,
        "evaluations": report.evaluations,
        "filled_cells": report.filled_cells,
        "grid_resolution": report.grid_resolution,
        "archive_type": report.archive_type,
        "n_cells": report.n_cells,
        "coverage": round(report.coverage, 6),
        "jsonl_raw_lines": report.jsonl_raw_lines,
        "jsonl_collapsed_cells": report.jsonl_collapsed_cells,
        "elapsed_seconds": round(report.elapsed_seconds, 3),
        "llm_enabled": report.llm_enabled,
        "surrogate_enabled": report.surrogate_enabled,
        "archive_jsonl": report.archive_jsonl_path,
    }
    if report.llm_enabled:
        payload.update(_llm_summary_fields(report))
    target.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


def _llm_summary_fields(report: NightlyRunReport) -> dict[str, object]:
    fields: dict[str, object] = {
        "llm_emit_attempts": report.llm_emit_attempts,
        "llm_emit_fallbacks": report.llm_emit_fallbacks,
        "llm_fallback_rate_pct": report.llm_fallback_rate_pct,
        "emit_llm_seconds": report.emit_llm_seconds,
        "eval_seconds": report.eval_seconds,
    }
    if report.llm_stack_version is not None:
        fields["llm_stack_version"] = report.llm_stack_version
    if report.llm_model is not None:
        fields["llm_model"] = report.llm_model
    if report.max_tokens is not None:
        fields["max_tokens"] = report.max_tokens
    if report.llm_parallel_emit is not None:
        fields["llm_parallel_emit"] = report.llm_parallel_emit
    if report.llm_parallel_workers is not None:
        fields["llm_parallel_workers"] = report.llm_parallel_workers
    if report.prompt_version is not None:
        fields["prompt_version"] = report.prompt_version
    return fields


def log_nightly_report(report: NightlyRunReport) -> None:
    """Log archive fill metrics for nightly operators."""
    logger.info(
        "MAP-Elites nightly: evaluations=%s filled_cells=%s coverage=%.4f "
        "jsonl_raw_lines=%s jsonl_collapsed_cells=%s elapsed_s=%.1f "
        "llm_enabled=%s surrogate_enabled=%s archive_type=%s",
        report.evaluations,
        report.filled_cells,
        report.coverage,
        report.jsonl_raw_lines,
        report.jsonl_collapsed_cells,
        report.elapsed_seconds,
        report.llm_enabled,
        report.surrogate_enabled,
        report.archive_type,
    )
    if report.llm_enabled and report.llm_fallback_rate_pct is not None:
        logger.info(
            "MAP-Elites LLM: model=%s stack=%s fallback_rate=%.3f%% "
            "attempts=%s parallel_emit=%s workers=%s",
            report.llm_model,
            report.llm_stack_version,
            report.llm_fallback_rate_pct,
            report.llm_emit_attempts,
            report.llm_parallel_emit,
            report.llm_parallel_workers,
        )
    summary_line = (
        f"MAP-Elites nightly: {report.evaluations} evaluations, "
        f"{report.filled_cells}/{report.n_cells} cells "
        f"({report.coverage * 100:.2f}% coverage), "
        f"jsonl lines={report.jsonl_raw_lines} (collapsed={report.jsonl_collapsed_cells}), "
        f"elapsed={report.elapsed_seconds:.1f}s, "
        f"archive={report.archive_jsonl_path}"
    )
    if report.llm_enabled and report.llm_fallback_rate_pct is not None:
        summary_line += (
            f", llm_fallback={report.llm_fallback_rate_pct:.2f}% "
            f"({report.llm_model})"
        )
    print(summary_line)


def _collapsed_archive_for_validation(
    jsonl_path: str | Path,
    *,
    config: SchedulerConfig,
    resume_archive_path: str | Path | None = None,
) -> ArchiveProtocol:
    """Collapse run JSONL; when resuming, merge with the prior archive snapshot."""
    run_archive = _load_collapsed_archive(jsonl_path, config=config)
    if resume_archive_path is None:
        return run_archive
    base = _load_collapsed_archive(resume_archive_path, config=config)
    merge_archives(base, run_archive)
    return base


def _load_collapsed_archive(
    path: str | Path,
    *,
    config: SchedulerConfig,
) -> ArchiveProtocol:
    archive_path = Path(path).expanduser()
    centroids_path = None
    if config.archive_type == "cvt":
        centroids_path = centroids_path_for_output(archive_path.parent)
    return load_and_collapse_jsonl(
        archive_path,
        archive_type=config.archive_type,
        resolution=config.grid_resolution,
        centroids_path=centroids_path,
    )

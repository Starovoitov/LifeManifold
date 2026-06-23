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
from worldspace.illuminators.scheduler import SchedulerConfig

logger = logging.getLogger(__name__)

__all__ = [
    "NightlyRunReport",
    "build_nightly_report",
    "log_nightly_report",
    "write_nightly_summary",
]


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


def build_nightly_report(
    *,
    result: MapElitesRunResult,
    config: SchedulerConfig,
    scheduler_path: str | Path,
    seed: int,
    elapsed_seconds: float,
    resume_archive_path: str | Path | None = None,
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
    )


def write_nightly_summary(path: str | Path, report: NightlyRunReport) -> None:
    """Write ``nightly_run_summary.json`` next to the archive JSONL."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
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
    target.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


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
    print(
        f"MAP-Elites nightly: {report.evaluations} evaluations, "
        f"{report.filled_cells}/{report.n_cells} cells "
        f"({report.coverage * 100:.2f}% coverage), "
        f"jsonl lines={report.jsonl_raw_lines} (collapsed={report.jsonl_collapsed_cells}), "
        f"elapsed={report.elapsed_seconds:.1f}s, "
        f"archive={report.archive_jsonl_path}"
    )


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

"""Eval-indexed archive snapshots for anytime QD curves (coverage / fitness vs budget)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from worldspace.illuminators.archive_protocol import ArchiveProtocol

ARCHIVE_TRACE_FILENAME = "archive_trace.jsonl"

__all__ = [
    "ARCHIVE_TRACE_FILENAME",
    "archive_trace_metrics",
    "qd_score_from_archive",
    "write_archive_trace_line",
]


def qd_score_from_archive(archive: ArchiveProtocol) -> float:
    """Canonical QD-score: sum of elite fitness over all filled archive cells."""
    total = 0.0
    for cell_id in range(archive.n_cells):
        elite = archive.get_cell(cell_id)
        if elite is not None:
            total += float(elite.fitness)
    return total


def archive_trace_metrics(
    archive: ArchiveProtocol,
) -> tuple[int, float, float | None, float]:
    """Return ``(filled_cells, coverage_fraction, mean_best_fitness, qd_score)``."""
    filled = archive.filled_count()
    n_cells = archive.n_cells
    coverage = float(filled) / float(n_cells) if n_cells else 0.0
    qd_score = qd_score_from_archive(archive)
    mean_fit = qd_score / float(filled) if filled else None
    return filled, coverage, mean_fit, qd_score


def write_archive_trace_line(trace_file: TextIO, payload: dict[str, object]) -> None:
    """Append one NDJSON snapshot row and flush."""
    trace_file.write(json.dumps(payload, ensure_ascii=True) + "\n")
    trace_file.flush()

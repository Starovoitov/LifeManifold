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
    "write_archive_trace_line",
]


def archive_trace_metrics(archive: ArchiveProtocol) -> tuple[int, float, float | None]:
    """Return ``(filled_cells, coverage_fraction, mean_best_fitness)``."""
    filled = archive.filled_count()
    n_cells = archive.n_cells
    coverage = float(filled) / float(n_cells) if n_cells else 0.0
    fitnesses: list[float] = []
    for cell_id in range(n_cells):
        elite = archive.get_cell(cell_id)
        if elite is not None:
            fitnesses.append(float(elite.fitness))
    mean_fit = sum(fitnesses) / len(fitnesses) if fitnesses else None
    return filled, coverage, mean_fit


def write_archive_trace_line(trace_file: TextIO, payload: dict[str, object]) -> None:
    """Append one NDJSON snapshot row and flush."""
    trace_file.write(json.dumps(payload, ensure_ascii=True) + "\n")
    trace_file.flush()

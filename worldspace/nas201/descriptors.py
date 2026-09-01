"""Frozen 20×20 equal-width bins on log10(params) × log10(flops)."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from worldspace.nas201.table import Nas201SearchRecord

DEFAULT_RESOLUTION = 20


@dataclass(frozen=True)
class Nas201BinEdges:
    """Equal-width edges computed from a pinned table; freeze after P2.1."""

    resolution: int
    log_params_min: float
    log_params_max: float
    log_flops_min: float
    log_flops_max: float
    n_architectures: int
    source_sha256: str

    def __post_init__(self) -> None:
        if self.resolution < 1:
            raise ValueError("resolution must be positive")
        if self.log_params_max < self.log_params_min:
            raise ValueError("log_params_max must be >= log_params_min")
        if self.log_flops_max < self.log_flops_min:
            raise ValueError("log_flops_max must be >= log_flops_min")


def log10_positive(value: float) -> float:
    if value <= 0.0:
        raise ValueError("log10 is undefined for non-positive values")
    return math.log10(value)


def equal_width_index(value: float, vmin: float, vmax: float, resolution: int) -> int:
    if vmax <= vmin:
        return 0
    scaled = (value - vmin) / (vmax - vmin)
    index = int(scaled * resolution)
    return min(resolution - 1, max(0, index))


def bin_for_record(
    record: Nas201SearchRecord, edges: Nas201BinEdges
) -> tuple[int, int]:
    log_params = log10_positive(record.params)
    log_flops = log10_positive(record.flops)
    return (
        equal_width_index(
            log_params,
            edges.log_params_min,
            edges.log_params_max,
            edges.resolution,
        ),
        equal_width_index(
            log_flops,
            edges.log_flops_min,
            edges.log_flops_max,
            edges.resolution,
        ),
    )


def measures_for_record(record: Nas201SearchRecord) -> tuple[float, float]:
    return (log10_positive(record.params), log10_positive(record.flops))


def bin_edges_from_records(
    records: Iterable[Nas201SearchRecord],
    *,
    resolution: int = DEFAULT_RESOLUTION,
    source_sha256: str,
) -> Nas201BinEdges:
    rows = list(records)
    if not rows:
        raise ValueError("cannot compute bin edges from an empty table")
    log_params = [log10_positive(row.params) for row in rows]
    log_flops = [log10_positive(row.flops) for row in rows]
    return Nas201BinEdges(
        resolution=resolution,
        log_params_min=min(log_params),
        log_params_max=max(log_params),
        log_flops_min=min(log_flops),
        log_flops_max=max(log_flops),
        n_architectures=len(rows),
        source_sha256=source_sha256,
    )


def occupancy_counts(
    records: Iterable[Nas201SearchRecord],
    edges: Nas201BinEdges,
) -> list[int]:
    counts = [0] * (edges.resolution * edges.resolution)
    for record in records:
        row, column = bin_for_record(record, edges)
        counts[row * edges.resolution + column] += 1
    return counts


def max_bin_fraction(counts: list[int], n_architectures: int) -> float:
    if n_architectures < 1:
        raise ValueError("n_architectures must be positive")
    return max(counts) / float(n_architectures) if counts else 0.0

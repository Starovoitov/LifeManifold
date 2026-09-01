"""10×10 equal-width bins for PCG Benchmark behavior characteristics."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RESOLUTION = 10


@dataclass(frozen=True)
class PcgBinEdges:
    resolution: int
    measure_names: tuple[str, str]
    axis0_min: float
    axis0_max: float
    axis1_min: float
    axis1_max: float
    n_samples: int
    problem_name: str

    def __post_init__(self) -> None:
        if self.resolution < 1:
            raise ValueError("resolution must be positive")
        if self.axis0_max < self.axis0_min or self.axis1_max < self.axis1_min:
            raise ValueError("bin maxima must be >= minima")


def equal_width_index(value: float, vmin: float, vmax: float, resolution: int) -> int:
    if vmax <= vmin:
        return 0
    scaled = (value - vmin) / (vmax - vmin)
    index = int(scaled * resolution)
    return min(resolution - 1, max(0, index))


def bin_for_measures(
    measures: tuple[float, float], edges: PcgBinEdges
) -> tuple[int, int]:
    return (
        equal_width_index(
            measures[0], edges.axis0_min, edges.axis0_max, edges.resolution
        ),
        equal_width_index(
            measures[1], edges.axis1_min, edges.axis1_max, edges.resolution
        ),
    )


def bin_edges_from_measures(
    measures: Iterable[tuple[float, float]],
    *,
    measure_names: tuple[str, str],
    problem_name: str,
    resolution: int = DEFAULT_RESOLUTION,
) -> PcgBinEdges:
    rows = list(measures)
    if not rows:
        raise ValueError("cannot compute bin edges from an empty sample")
    axis0 = [row[0] for row in rows]
    axis1 = [row[1] for row in rows]
    return PcgBinEdges(
        resolution=resolution,
        measure_names=measure_names,
        axis0_min=min(axis0),
        axis0_max=max(axis0),
        axis1_min=min(axis1),
        axis1_max=max(axis1),
        n_samples=len(rows),
        problem_name=problem_name,
    )


def occupancy_counts(
    measures: Iterable[tuple[float, float]],
    edges: PcgBinEdges,
) -> list[int]:
    counts = [0] * (edges.resolution * edges.resolution)
    for item in measures:
        row, column = bin_for_measures(item, edges)
        counts[row * edges.resolution + column] += 1
    return counts


def load_frozen_bin_edges(path: Path) -> PcgBinEdges:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload["measure_names"]
    return PcgBinEdges(
        resolution=int(payload["resolution"]),
        measure_names=(str(names[0]), str(names[1])),
        axis0_min=float(payload["axis0_min"]),
        axis0_max=float(payload["axis0_max"]),
        axis1_min=float(payload["axis1_min"]),
        axis1_max=float(payload["axis1_max"]),
        n_samples=int(payload["n_samples"]),
        problem_name=str(payload["problem_name"]),
    )


def dump_frozen_bin_edges(edges: PcgBinEdges, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "resolution": edges.resolution,
                "measure_names": list(edges.measure_names),
                "axis0_min": edges.axis0_min,
                "axis0_max": edges.axis0_max,
                "axis1_min": edges.axis1_min,
                "axis1_max": edges.axis1_max,
                "n_samples": edges.n_samples,
                "problem_name": edges.problem_name,
                "stage": "pcg_smoke",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

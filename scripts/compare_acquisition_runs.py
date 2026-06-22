"""Compare MAP-Elites runs for Surrogate Acquisition A/B analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NIGHTLY_SUMMARY_FILENAME = "nightly_run_summary.json"
MAP_ELITES_ARCHIVE_FILENAME = "map_elites_archive.jsonl"
CVT_CENTROIDS_FILENAME = "cvt_centroids.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs candidate illuminator output directories",
    )
    parser.add_argument(
        "--baseline-dir",
        required=True,
        help="Output directory for baseline (e.g. acquisition off)",
    )
    parser.add_argument(
        "--candidate-dir",
        required=True,
        help="Output directory for candidate (e.g. filter mode)",
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=10,
        help="Grid resolution fallback when run metadata is missing",
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        default=None,
        help="Override archive niche count when summary/JSONL metadata is missing",
    )
    return parser.parse_args()


class RunArchiveMeta(TypedDict):
    archive_type: str
    n_cells: int
    grid_resolution: int | None


class _RunSummary(TypedDict):
    archive_type: str
    n_cells: int
    evaluations: float
    filled_cells: float
    filled_cells_pct: float
    mean_best_fitness: float
    recommended_skip_rate: float | None


def main() -> None:
    args = parse_args()
    baseline = _summarize_run(
        Path(args.baseline_dir),
        grid_resolution=args.grid_resolution,
        n_cells_override=args.n_cells,
    )
    candidate = _summarize_run(
        Path(args.candidate_dir),
        grid_resolution=args.grid_resolution,
        n_cells_override=args.n_cells,
    )
    if baseline["archive_type"] != candidate["archive_type"]:
        print(
            "WARNING: archive_type mismatch "
            f"(baseline={baseline['archive_type']}, "
            f"candidate={candidate['archive_type']}); "
            "filled-cell coverage is not directly comparable.",
            file=sys.stderr,
        )
    eval_reduction_pct = 0.0
    if baseline["evaluations"] > 0:
        eval_reduction_pct = (
            100.0
            * (baseline["evaluations"] - candidate["evaluations"])
            / baseline["evaluations"]
        )
    filled_delta_pct = 0.0
    if baseline["filled_cells_pct"] > 0:
        filled_delta_pct = (
            100.0
            * (candidate["filled_cells_pct"] - baseline["filled_cells_pct"])
            / baseline["filled_cells_pct"]
        )
    fitness_delta_pct = 0.0
    if baseline["mean_best_fitness"] > 0:
        fitness_delta_pct = (
            100.0
            * (candidate["mean_best_fitness"] - baseline["mean_best_fitness"])
            / baseline["mean_best_fitness"]
        )
    print("=== Acquisition A/B ===")
    print(
        f"baseline archive: {baseline['archive_type']} "
        f"({baseline['n_cells']} niches)"
    )
    print(
        f"candidate archive: {candidate['archive_type']} "
        f"({candidate['n_cells']} niches)"
    )
    print(f"baseline evaluations: {baseline['evaluations']}")
    print(f"candidate evaluations: {candidate['evaluations']}")
    print(f"eval reduction: {eval_reduction_pct:.1f}%")
    print(
        f"filled cells: baseline {baseline['filled_cells_pct']:.1f}% "
        f"candidate {candidate['filled_cells_pct']:.1f}% "
        f"(delta {filled_delta_pct:+.1f}%)"
    )
    print(
        f"mean best fitness: baseline {baseline['mean_best_fitness']:.4f} "
        f"candidate {candidate['mean_best_fitness']:.4f} "
        f"(delta {fitness_delta_pct:+.1f}%)"
    )
    if candidate.get("recommended_skip_rate") is not None:
        print(
            f"candidate policy skip rate (archive): "
            f"{candidate['recommended_skip_rate']:.1f}%"
        )


def resolve_run_archive_meta(
    run_dir: Path,
    *,
    grid_resolution: int,
    n_cells_override: int | None = None,
) -> RunArchiveMeta:
    """Resolve archive type and niche count for coverage denominators."""
    if n_cells_override is not None:
        if n_cells_override < 1:
            msg = f"n_cells must be >= 1, got {n_cells_override}"
            raise ValueError(msg)
        summary_meta = _meta_from_nightly_summary(run_dir)
        archive_type = (
            summary_meta["archive_type"] if summary_meta is not None else "grid"
        )
        grid_res = (
            summary_meta["grid_resolution"]
            if summary_meta is not None
            else grid_resolution
        )
        return {
            "archive_type": archive_type,
            "n_cells": int(n_cells_override),
            "grid_resolution": grid_res,
        }

    summary_meta = _meta_from_nightly_summary(run_dir)
    if summary_meta is not None:
        return summary_meta

    jsonl_meta = _meta_from_archive_jsonl(run_dir)
    if jsonl_meta is not None:
        return jsonl_meta

    if grid_resolution < 1:
        msg = f"grid_resolution must be >= 1, got {grid_resolution}"
        raise ValueError(msg)
    return {
        "archive_type": "grid",
        "n_cells": grid_resolution * grid_resolution,
        "grid_resolution": grid_resolution,
    }


def _meta_from_nightly_summary(run_dir: Path) -> RunArchiveMeta | None:
    summary_path = run_dir / NIGHTLY_SUMMARY_FILENAME
    if not summary_path.is_file():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    raw_n_cells = payload.get("n_cells")
    if raw_n_cells is None:
        raw_resolution = payload.get("grid_resolution")
        if raw_resolution is None:
            return None
        try:
            resolution = int(raw_resolution)
        except (TypeError, ValueError):
            return None
        if resolution < 1:
            return None
        n_cells = resolution * resolution
    else:
        try:
            n_cells = int(raw_n_cells)
        except (TypeError, ValueError):
            return None
        if n_cells < 1:
            return None
        resolution_raw = payload.get("grid_resolution")
        resolution = int(resolution_raw) if resolution_raw is not None else None
    archive_type = str(payload.get("archive_type", "grid"))
    return {
        "archive_type": archive_type,
        "n_cells": n_cells,
        "grid_resolution": resolution,
    }


def _meta_from_archive_jsonl(run_dir: Path) -> RunArchiveMeta | None:
    archive_path = run_dir / MAP_ELITES_ARCHIVE_FILENAME
    if not archive_path.is_file():
        return None
    first_record: dict | None = None
    for line in archive_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            first_record = parsed
            break
    if first_record is None:
        return None

    schema_version = str(first_record.get("schema_version", "1.2"))
    if schema_version == "1.2":
        return None

    archive_type = str(first_record.get("archive_type", "grid"))
    if archive_type == "cvt":
        centroids_path = run_dir / CVT_CENTROIDS_FILENAME
        if centroids_path.is_file():
            try:
                payload = json.loads(centroids_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            if isinstance(payload, dict):
                centroids = payload.get("centroids")
                if isinstance(centroids, list) and centroids:
                    return {
                        "archive_type": "cvt",
                        "n_cells": len(centroids),
                        "grid_resolution": None,
                    }
        return None

    cell_id = first_record.get("cell_id")
    if isinstance(cell_id, int) and cell_id >= 0:
        bin_value = first_record.get("bin")
        if isinstance(bin_value, list) and len(bin_value) == 2:
            try:
                i = int(bin_value[0])
                j = int(bin_value[1])
            except (TypeError, ValueError):
                return None
            resolution = _infer_grid_resolution_from_bin_cell(i, j, cell_id)
            if resolution is None:
                return None
            n_cells = resolution * resolution
            return {
                "archive_type": "grid",
                "n_cells": n_cells,
                "grid_resolution": resolution,
            }
    return None


def _infer_grid_resolution_from_bin_cell(
    i: int,
    j: int,
    cell_id: int,
) -> int | None:
    """Infer grid side length only when ``cell_id == i * resolution + j`` holds uniquely."""
    from worldspace.illuminators.archive import bin_ij_from_flat_cell_id, flat_cell_id

    if i < 0 or j < 0 or cell_id < 0:
        return None
    min_resolution = max(i, j) + 1
    if i == 0:
        if cell_id != j:
            return None
        return None
    if (cell_id - j) % i != 0:
        return None
    resolution = (cell_id - j) // i
    if resolution < min_resolution:
        return None
    try:
        if flat_cell_id((i, j), resolution=resolution) != cell_id:
            return None
        if bin_ij_from_flat_cell_id(cell_id, resolution=resolution) != (i, j):
            return None
    except (IndexError, ValueError):
        return None
    return resolution


def _summarize_run(
    path: Path,
    *,
    grid_resolution: int,
    n_cells_override: int | None = None,
) -> _RunSummary:
    meta = resolve_run_archive_meta(
        path,
        grid_resolution=grid_resolution,
        n_cells_override=n_cells_override,
    )
    archive_path = path / MAP_ELITES_ARCHIVE_FILENAME
    surrogate_path = path / "surrogate_archive.jsonl"
    evaluations = 0
    best_fitness_values: list[float] = []
    if archive_path.is_file():
        for line in archive_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            evaluations += 1
            best_fitness_values.append(float(record.get("fitness", 0.0)))
    filled_cells = len(best_fitness_values)
    total_cells = max(1, meta["n_cells"])
    skip_count = 0
    slot_count = 0
    if surrogate_path.is_file():
        for line in surrogate_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            slot_count += 1
            record = json.loads(line)
            if record.get("decision") == "skip":
                skip_count += 1
    return {
        "archive_type": meta["archive_type"],
        "n_cells": meta["n_cells"],
        "evaluations": float(evaluations),
        "filled_cells": float(filled_cells),
        "filled_cells_pct": 100.0 * filled_cells / total_cells,
        "mean_best_fitness": (
            float(sum(best_fitness_values) / len(best_fitness_values))
            if best_fitness_values
            else 0.0
        ),
        "recommended_skip_rate": (
            100.0 * skip_count / slot_count if slot_count > 0 else None
        ),
    }


if __name__ == "__main__":
    main()

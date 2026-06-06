"""Rebuild surrogate training buffer rows from MAP-Elites archive JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from worldspace.illuminators.archive import (
    ArchiveElite,
    archive_record_to_elite,
    count_archive_jsonl_lines,
    load_and_collapse_jsonl,
)
from worldspace.illuminators.evaluation import (
    apply_canonical_seed,
    extinction_probability,
)
from worldspace.surrogate.buffer import buffer_record
from worldspace.surrogate.feature_extractor import extract
from worldspace.surrogate.model import TARGET_KEYS

__all__ = [
    "backfill_buffer_from_archive",
    "backfill_buffer_from_collapsed_archive",
    "targets_from_archive_elite",
    "targets_from_archive_record",
]


def targets_from_archive_elite(elite: ArchiveElite) -> dict[str, float]:
    """Build Strategy A targets from a collapsed in-memory archive elite."""
    if elite.measures is None or elite.metrics is None:
        msg = "archive elite requires measures and metrics"
        raise ValueError(msg)
    metrics = elite.metrics
    final_density = float(metrics.density_mean)
    return {
        "stability": float(elite.measures["stability"]),
        "diversity": float(elite.measures["diversity"]),
        "oscillation_score": float(metrics.oscillation_score),
        "topology_interface_index": float(metrics.topology_interface_index),
        "topology_window_heterogeneity": float(metrics.topology_window_heterogeneity),
        "final_density": final_density,
        "early_extinction_prob": extinction_probability(final_density),
    }


def targets_from_archive_record(record: dict) -> dict[str, float]:
    """Build Strategy A targets from one archive JSONL record (schema 1.2)."""
    measures = record["measures"]
    metrics = record["metrics"]
    if not isinstance(measures, dict) or not isinstance(metrics, dict):
        msg = "archive record requires measures and metrics objects"
        raise ValueError(msg)
    final_density = float(metrics["density_mean"])
    return {
        "stability": float(measures["stability"]),
        "diversity": float(measures["diversity"]),
        "oscillation_score": float(metrics["oscillation_score"]),
        "topology_interface_index": float(metrics["topology_interface_index"]),
        "topology_window_heterogeneity": float(
            metrics["topology_window_heterogeneity"]
        ),
        "final_density": final_density,
        "early_extinction_prob": extinction_probability(final_density),
    }


def backfill_buffer_from_archive(
    archive_path: Path | str,
    buffer_path: Path | str,
    *,
    overwrite: bool = True,
) -> dict[str, int]:
    """Write training buffer JSONL from append-only archive lines.

      Each parseable archive row becomes one buffer row (features from ``world_spec``,
    targets from stored ``measures`` / ``metrics``).
    """
    archive = Path(archive_path)
    buffer = Path(buffer_path)
    if not archive.is_file():
        msg = f"archive file not found: {archive}"
        raise FileNotFoundError(msg)

    buffer.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and buffer.is_file():
        buffer.unlink()

    written = 0
    skipped = 0
    with (
        archive.open(encoding="utf-8") as src,
        buffer.open("w", encoding="utf-8") as out,
    ):
        for line_no, line in enumerate(src, start=1):
            stripped = line.strip()
            if not stripped:
                skipped += 1
                continue
            try:
                record = json.loads(stripped)
                elite = archive_record_to_elite(record)
                if elite.world_spec is None:
                    skipped += 1
                    continue
                spec = elite.world_spec
                apply_canonical_seed(spec)
                features = extract(spec)
                targets = targets_from_archive_record(record)
                _validate_targets_dict(targets)
                metadata = record.get("metadata") or {}
                emitter_type = str(
                    metadata.get("emitter_type")
                    or metadata.get("generated_by")
                    or "unknown"
                )
                row = buffer_record(
                    features=features,
                    targets=targets,
                    emitter_type=emitter_type,
                    world_spec=spec.to_json_dict(),
                    metadata={
                        "source": "archive_backfill",
                        "archive_path": str(archive.resolve()),
                        "archive_line": line_no,
                    },
                )
                out.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                written += 1
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                skipped += 1

    return {
        "archive_lines": count_archive_jsonl_lines(archive),
        "buffer_rows_written": written,
        "lines_skipped": skipped,
    }


def backfill_buffer_from_collapsed_archive(
    archive_path: Path | str,
    buffer_path: Path | str,
    *,
    resolution: int,
    overwrite: bool = True,
) -> dict[str, int]:
    """Write one buffer row per filled archive cell (best elite per bin)."""
    archive = Path(archive_path)
    buffer = Path(buffer_path)
    if not archive.is_file():
        msg = f"archive file not found: {archive}"
        raise FileNotFoundError(msg)

    collapsed = load_and_collapse_jsonl(archive, resolution=resolution)
    buffer.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and buffer.is_file():
        buffer.unlink()

    written = 0
    skipped = 0
    res = collapsed.resolution
    with buffer.open("w", encoding="utf-8") as out:
        for i in range(res):
            for j in range(res):
                elite = collapsed.get(i, j)
                if elite is None or elite.world_spec is None:
                    continue
                try:
                    spec = elite.world_spec
                    apply_canonical_seed(spec)
                    features = extract(spec)
                    targets = targets_from_archive_elite(elite)
                    _validate_targets_dict(targets)
                    emitter_type = "unknown"
                    if elite.metadata is not None:
                        emitter_type = (
                            elite.metadata.emitter_type or elite.metadata.generated_by
                        )
                    row = buffer_record(
                        features=features,
                        targets=targets,
                        emitter_type=str(emitter_type),
                        world_spec=spec.to_json_dict(),
                        metadata={
                            "source": "archive_backfill_collapsed",
                            "archive_path": str(archive.resolve()),
                            "bin": [i, j],
                        },
                    )
                    out.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                    written += 1
                except (KeyError, TypeError, ValueError):
                    skipped += 1

    return {
        "archive_lines": count_archive_jsonl_lines(archive),
        "collapsed_filled_cells": collapsed.filled_count(),
        "buffer_rows_written": written,
        "lines_skipped": skipped,
    }


def _validate_targets_dict(targets: dict[str, float]) -> None:
    missing = [key for key in TARGET_KEYS if key not in targets]
    if missing:
        msg = f"Missing required target keys: {missing}"
        raise ValueError(msg)

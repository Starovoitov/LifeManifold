"""Load SurrogateArchive acquisition JSONL (schema 1.0) for dashboard tables and charts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.utils.config import load_config, resolve_surrogate_archive_path

SURROGATE_ARCHIVE_SCHEMA_VERSION = "1.0"

__all__ = [
    "ArchiveLogBundle",
    "apply_archive_log_filters",
    "flatten_archive_record",
    "get_archive_log_bundle",
    "load_surrogate_archive",
    "read_surrogate_archive_jsonl",
    "try_flatten_archive_record",
]


@dataclass(frozen=True)
class ArchiveLogBundle:
    """Parsed SurrogateArchive rows plus raw records for export."""

    records: pd.DataFrame
    raw_records: list[dict[str, Any]]
    line_count_raw: int
    invalid_line_count: int
    source_path: str


def load_surrogate_archive(path: Path) -> pd.DataFrame:
    """Load SurrogateArchive JSONL into a flat DataFrame."""
    frame, _, _, _ = read_surrogate_archive_jsonl(path, load_config())
    return frame


def get_archive_log_bundle(path: Path | None = None) -> ArchiveLogBundle:
    """Load acquisition log with Streamlit disk cache (path + mtime)."""
    cfg = load_config()
    target = path if path is not None else resolve_surrogate_archive_path(cfg)
    if not target.is_file():
        msg = f"surrogate archive file not found: {target}"
        raise FileNotFoundError(msg)
    mtime = float(target.stat().st_mtime)
    return _cached_load_archive_log_bundle(str(target.resolve()), mtime)


@st.cache_data(show_spinner=False)
def _cached_load_archive_log_bundle(path_str: str, mtime: float) -> ArchiveLogBundle:
    del mtime
    return _load_archive_log_bundle(Path(path_str))


def _load_archive_log_bundle(path: Path) -> ArchiveLogBundle:
    cfg = load_config()
    frame, raw_records, line_count_raw, invalid_line_count = (
        read_surrogate_archive_jsonl(
            path,
            cfg,
        )
    )
    return ArchiveLogBundle(
        records=frame,
        raw_records=raw_records,
        line_count_raw=line_count_raw,
        invalid_line_count=invalid_line_count,
        source_path=str(path.resolve()),
    )


def read_surrogate_archive_jsonl(
    path: Path,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], int, int]:
    """Return ``(flat_frame, raw_records, raw_line_count, invalid_line_count)``."""
    del cfg
    rows: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    invalid = 0
    line_count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            line_count += 1
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if not isinstance(record, dict):
                invalid += 1
                continue
            flat = try_flatten_archive_record(record)
            if flat is None:
                invalid += 1
                continue
            rows.append(flat)
            raw_records.append(record)
    frame = pd.DataFrame(rows) if rows else pd.DataFrame()
    return frame, raw_records, line_count, invalid


def flatten_archive_record(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten one SurrogateArchive JSON object (raises on invalid shape)."""
    row = try_flatten_archive_record(record)
    if row is None:
        msg = "invalid surrogate archive record"
        raise ValueError(msg)
    return row


def try_flatten_archive_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return a flat row dict or ``None`` when the record is invalid."""
    schema = record.get("schema_version")
    if schema != SURROGATE_ARCHIVE_SCHEMA_VERSION:
        return None

    run_id = record.get("run_id")
    iteration = record.get("iteration")
    candidate_id = record.get("candidate_id")
    emitter_type = record.get("emitter_type")
    target_bin = record.get("target_bin")
    raw_target_cell_id = record.get("target_cell_id")
    decision = record.get("decision")
    decision_reason = record.get("decision_reason")
    acquisition_mode = record.get("acquisition_mode")
    world_spec_hash = record.get("world_spec_hash")
    prediction = record.get("prediction")

    if not isinstance(run_id, str) or not run_id.strip():
        return None
    if not isinstance(iteration, int):
        return None
    if not isinstance(candidate_id, int):
        return None
    if not isinstance(emitter_type, str) or not emitter_type.strip():
        return None
    has_target_bin = isinstance(target_bin, list) and len(target_bin) == 2
    if not has_target_bin and raw_target_cell_id is None:
        return None
    bin_i: int | None = None
    bin_j: int | None = None
    if has_target_bin:
        assert isinstance(target_bin, list)
        try:
            bin_i = int(target_bin[0])
            bin_j = int(target_bin[1])
        except (TypeError, ValueError):
            return None
    target_cell_id: int | None = None
    if raw_target_cell_id is not None:
        try:
            target_cell_id = int(raw_target_cell_id)
        except (TypeError, ValueError):
            return None
    if decision not in ("eval", "skip"):
        return None
    if not isinstance(decision_reason, str):
        return None
    if acquisition_mode not in ("off", "shadow", "filter"):
        return None
    if not isinstance(world_spec_hash, str):
        return None
    if not isinstance(prediction, dict):
        return None

    try:
        pred_fitness = float(prediction.get("fitness", 0.0))
        pred_uncertainty = float(prediction.get("uncertainty", 0.0))
    except (TypeError, ValueError):
        return None

    row: dict[str, Any] = {
        "schema_version": schema,
        "run_id": run_id,
        "iteration": iteration,
        "candidate_id": candidate_id,
        "emitter_type": emitter_type,
        "world_spec_hash": world_spec_hash,
        "pred_fitness": pred_fitness,
        "pred_uncertainty": pred_uncertainty,
        "decision": decision,
        "decision_reason": decision_reason,
        "acquisition_mode": acquisition_mode,
    }
    if bin_i is not None and bin_j is not None:
        row["target_bin_i"] = bin_i
        row["target_bin_j"] = bin_j
        row["target_bin_label"] = f"{bin_i},{bin_j}"
    if target_cell_id is not None:
        row["target_cell_id"] = target_cell_id

    outcome = record.get("eval_outcome")
    if outcome is None:
        row["has_eval"] = False
        row["eval_fitness"] = float("nan")
        row["eval_accepted"] = False
        row["eval_improved"] = False
    elif isinstance(outcome, dict):
        row["has_eval"] = True
        try:
            row["eval_fitness"] = float(outcome.get("fitness", float("nan")))
        except (TypeError, ValueError):
            row["eval_fitness"] = float("nan")
        row["eval_accepted"] = bool(outcome.get("accepted", False))
        row["eval_improved"] = bool(outcome.get("improved", False))
    else:
        return None

    return row


def apply_archive_log_filters(
    bundle: ArchiveLogBundle,
    *,
    decisions: list[str],
    acquisition_modes: list[str],
    emitter_types: list[str],
    iteration_range: tuple[int, int] | None,
) -> pd.DataFrame:
    """Filter acquisition log rows in memory (empty multiselect → no rows)."""
    frame = bundle.records
    if frame.empty:
        return frame.copy()
    if not decisions or not acquisition_modes or not emitter_types:
        return frame.iloc[0:0].copy()

    mask = pd.Series(True, index=frame.index)
    mask &= frame["decision"].isin(decisions)
    mask &= frame["acquisition_mode"].isin(acquisition_modes)
    mask &= frame["emitter_type"].isin(emitter_types)
    if iteration_range is not None:
        low, high = iteration_range
        mask &= frame["iteration"] >= low
        mask &= frame["iteration"] <= high
    return frame.loc[mask].reset_index(drop=True)

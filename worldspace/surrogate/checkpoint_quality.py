"""Hold-out quality gate checks for runtime surrogate checkpoint use."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldspace.surrogate.feature_extractor import (
    FEATURE_SCHEMA_VERSION,
    feature_dim_for_schema,
)

__all__ = [
    "checkpoint_quality_allows_hints",
    "default_summary_path",
    "load_checkpoint_summary",
]


def default_summary_path(checkpoint_path: Path | str) -> Path:
    """Return the default JSON summary path beside a checkpoint file."""
    path = Path(checkpoint_path).expanduser()
    return path.with_name(f"{path.stem}.summary.json")


def load_checkpoint_summary(checkpoint_path: Path | str) -> dict[str, Any] | None:
    """Load the JSON training summary beside a checkpoint, if present."""
    summary_path = default_summary_path(Path(checkpoint_path).expanduser())
    if not summary_path.is_file():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def checkpoint_quality_allows_hints(checkpoint_path: Path | str) -> bool:
    """Return whether a checkpoint passed the pilot hold-out hints tier."""
    summary = load_checkpoint_summary(checkpoint_path)
    if summary is None:
        return False
    if not _summary_schema_matches(summary):
        return False
    hints_ok = summary.get("hints_ok")
    if hints_ok is True:
        return True
    if hints_ok is False:
        return False
    return summary.get("quality_passed") is True


def _summary_schema_matches(summary: dict[str, Any]) -> bool:
    if summary.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        return False
    feature_dim = summary.get("feature_dim")
    return feature_dim == feature_dim_for_schema(FEATURE_SCHEMA_VERSION)

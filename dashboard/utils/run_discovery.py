"""Discover MAP-Elites run directories (archive JSONL + optional summary JSON)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dashboard.utils.config import (
    existing_archive_paths,
    load_config,
    resolve_repo_path,
)

ARCHIVE_JSONL_NAME = "map_elites_archive.jsonl"
SUMMARY_FILENAMES = ("nightly_run_summary.json", "smoke_run_summary.json")

_DEFAULT_SCAN_DIRS = (
    "output",
    "artifacts/map_elites_nightly",
    "artifacts/map_elites_smoke",
)

__all__ = [
    "RunInfo",
    "discover_runs",
    "load_summary_json",
    "summary_get",
]


@dataclass(frozen=True)
class RunInfo:
    """One MAP-Elites run: archive JSONL and optional summary sidecar."""

    run_dir: Path
    archive_path: Path
    summary_path: Path | None
    summary: dict[str, Any] | None
    archive_mtime: float


def discover_runs(cfg: dict[str, Any] | None = None) -> list[RunInfo]:
    """Find archive JSONL files under configured scan roots (newest first)."""
    config = cfg if cfg is not None else load_config()
    seen: set[str] = set()
    runs: list[RunInfo] = []

    for root in _scan_roots(config):
        if not root.exists():
            continue
        for archive_path in _find_archives_under(root):
            key = str(archive_path.resolve())
            if key in seen:
                continue
            seen.add(key)
            run_dir = archive_path.parent
            summary_path = _find_summary_in_dir(run_dir)
            summary = (
                load_summary_json(summary_path) if summary_path is not None else None
            )
            runs.append(
                RunInfo(
                    run_dir=run_dir,
                    archive_path=archive_path,
                    summary_path=summary_path,
                    summary=summary,
                    archive_mtime=float(archive_path.stat().st_mtime),
                )
            )

    runs.sort(key=lambda run: run.archive_mtime, reverse=True)
    return runs


def load_summary_json(path: Path) -> dict[str, Any] | None:
    """Load a nightly or smoke summary file; return None on missing or invalid JSON."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def summary_get(
    summary: dict[str, Any] | None,
    *keys: str,
    default: Any = None,
) -> Any:
    """Read the first present key from a summary dict (smoke vs nightly field names)."""
    if summary is None:
        return default
    for key in keys:
        if key in summary:
            return summary[key]
    return default


def _scan_roots(cfg: dict[str, Any]) -> list[Path]:
    paths_section = cfg.get("paths")
    dirs: list[str] = list(_DEFAULT_SCAN_DIRS)
    if isinstance(paths_section, dict):
        raw_dirs = paths_section.get("run_scan_dirs")
        if isinstance(raw_dirs, list):
            dirs = [str(entry) for entry in raw_dirs]

    roots: list[Path] = [_resolve_scan_path(relative) for relative in dirs]
    for archive in existing_archive_paths(cfg):
        roots.append(archive.parent)

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = str(root.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(root)
    return unique


def _find_archives_under(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(root.rglob(ARCHIVE_JSONL_NAME))


def _resolve_scan_path(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    return resolve_repo_path(relative)


def _find_summary_in_dir(run_dir: Path) -> Path | None:
    for name in SUMMARY_FILENAMES:
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    return None

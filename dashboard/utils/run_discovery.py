"""Discover MAP-Elites run directories (archive JSONL + optional summary JSON)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dashboard.utils.config import load_config, repo_root, resolve_repo_path

ARCHIVE_JSONL_NAME = "map_elites_archive.jsonl"
SUMMARY_FILENAMES = ("nightly_run_summary.json", "smoke_run_summary.json")

# Used only when ``paths.run_scan_dirs`` is missing or empty in config.yaml.
_DEFAULT_SCAN_DIRS = "artifacts/"

_GLOB_CHARS = frozenset("*?[]")

__all__ = [
    "RunInfo",
    "default_scan_dir_entries",
    "discover_runs",
    "expand_scan_dir_entries",
    "load_summary_json",
    "summary_get",
]


def default_scan_dir_entries() -> list[str]:
    """Fallback scan patterns when ``paths.run_scan_dirs`` is unset in config.yaml."""
    return list(_DEFAULT_SCAN_DIRS)


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


def _scan_dir_entries(cfg: dict[str, Any]) -> list[str]:
    """``paths.run_scan_dirs`` from config.yaml; code defaults if missing or empty."""
    paths_section = cfg.get("paths")
    if isinstance(paths_section, dict):
        raw_dirs = paths_section.get("run_scan_dirs")
        if isinstance(raw_dirs, list) and raw_dirs:
            return [text for entry in raw_dirs if (text := str(entry).strip())]
    return list(_DEFAULT_SCAN_DIRS)


def _scan_roots(cfg: dict[str, Any]) -> list[Path]:
    roots: list[Path] = expand_scan_dir_entries(_scan_dir_entries(cfg))

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


def expand_scan_dir_entries(entries: list[str]) -> list[Path]:
    """Expand ``run_scan_dirs`` entries; paths with ``*``, ``?``, or ``[]`` glob under repo root."""
    roots: list[Path] = []
    for entry in entries:
        roots.extend(_expand_scan_dir_entry(entry))
    return roots


def _expand_scan_dir_entry(entry: str) -> list[Path]:
    text = entry.strip()
    if not text:
        return []
    if not any(char in text for char in _GLOB_CHARS):
        return [_resolve_scan_path(text)]

    path = Path(text).expanduser()
    if path.is_absolute():
        matches = sorted(path.parent.glob(path.name))
    else:
        matches = sorted(repo_root().glob(text))

    return [match.resolve() for match in matches if match.is_dir()]


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

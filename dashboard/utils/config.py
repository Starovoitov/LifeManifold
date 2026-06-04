"""Load dashboard YAML config and resolve repo-relative paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.yaml"

__all__ = [
    "config_path",
    "configured_archive_paths",
    "existing_archive_paths",
    "load_config",
    "repo_root",
    "resolve_repo_path",
    "resolve_surrogate_archive_path",
    "resolve_surrogate_buffer_path",
    "sort_archive_paths_by_mtime",
]


def repo_root() -> Path:
    """Repository root (parent of ``dashboard/``)."""
    return Path(__file__).resolve().parents[2]


def config_path() -> Path:
    """Default ``config.yaml`` location."""
    return _DEFAULT_CONFIG_PATH


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load dashboard config from YAML."""
    target = path or _DEFAULT_CONFIG_PATH
    with target.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = f"Expected mapping in config file, got {type(data).__name__}"
        raise TypeError(msg)
    return data


def resolve_repo_path(relative: str) -> Path:
    """Resolve a path relative to the repository root."""
    return repo_root() / relative


def configured_archive_paths(cfg: dict[str, Any] | None = None) -> list[Path]:
    """Return ``paths.archives`` entries that exist on disk (explicit overrides only)."""
    config = cfg if cfg is not None else load_config()
    paths_section = config.get("paths")
    if not isinstance(paths_section, dict):
        return []
    raw_archives = paths_section.get("archives")
    if not isinstance(raw_archives, list):
        return []
    found: list[Path] = []
    for entry in raw_archives:
        if not isinstance(entry, str):
            continue
        resolved = resolve_repo_path(entry)
        if resolved.is_file():
            found.append(resolved)
    return found


def _archive_mtime(path: Path) -> float | None:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return None


def sort_archive_paths_by_mtime(entries: list[tuple[Path, float]]) -> list[Path]:
    """Sort by cached mtimes; drop paths that no longer exist (no second ``stat()``)."""
    ordered = sorted(entries, key=lambda item: item[1], reverse=True)
    return [path for path, _ in ordered if path.is_file()]


def existing_archive_paths(cfg: dict[str, Any] | None = None) -> list[Path]:
    """Archives for the sidebar: ``paths.archives`` plus ``paths.run_scan_dirs`` discovery."""
    config = cfg if cfg is not None else load_config()
    seen: set[str] = set()
    by_mtime: list[tuple[Path, float]] = []

    for path in configured_archive_paths(config):
        key = str(path.resolve())
        if key in seen:
            continue
        mtime = _archive_mtime(path)
        if mtime is None:
            continue
        seen.add(key)
        by_mtime.append((path, mtime))

    from dashboard.utils.run_discovery import discover_runs

    for run in discover_runs(config):
        key = str(run.archive_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        by_mtime.append((run.archive_path, run.archive_mtime))

    return sort_archive_paths_by_mtime(by_mtime)


def resolve_surrogate_archive_path(cfg: dict[str, Any] | None = None) -> Path:
    """Resolve configured SurrogateArchive JSONL path (acquisition log)."""
    config = cfg if cfg is not None else load_config()
    paths_section = config.get("paths")
    if not isinstance(paths_section, dict):
        msg = "config paths section missing"
        raise KeyError(msg)
    raw = paths_section.get("surrogate_archive")
    if not isinstance(raw, str) or not raw.strip():
        msg = "paths.surrogate_archive must be a non-empty string"
        raise KeyError(msg)
    return resolve_repo_path(raw)


def resolve_surrogate_buffer_path(cfg: dict[str, Any] | None = None) -> Path:
    """Resolve configured surrogate training buffer JSONL path."""
    config = cfg if cfg is not None else load_config()
    paths_section = config.get("paths")
    if not isinstance(paths_section, dict):
        msg = "config paths section missing"
        raise KeyError(msg)
    raw = paths_section.get("surrogate_buffer")
    if not isinstance(raw, str) or not raw.strip():
        msg = "paths.surrogate_buffer must be a non-empty string"
        raise KeyError(msg)
    return resolve_repo_path(raw)

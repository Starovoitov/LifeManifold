"""Load dashboard YAML config and resolve repo-relative paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.yaml"

__all__ = [
    "config_path",
    "existing_archive_paths",
    "load_config",
    "repo_root",
    "resolve_repo_path",
    "resolve_surrogate_archive_path",
    "resolve_surrogate_buffer_path",
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


def existing_archive_paths(cfg: dict[str, Any] | None = None) -> list[Path]:
    """Return configured archive JSONL paths that exist on disk."""
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

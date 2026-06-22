"""Load dashboard YAML config and resolve repo-relative paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.yaml"

DASHBOARD_ARCHIVE_SESSION_KEY = "dashboard_archive_select"

_CHECKPOINT_FILENAMES: tuple[str, ...] = (
    "nightly_v2.pkl",
    "nightly.pkl",
    "latest.pkl",
    "micro.pkl",
)

SURROGATE_ARCHIVE_JSONL_NAME = "surrogate_archive.jsonl"

__all__ = [
    "DASHBOARD_ARCHIVE_SESSION_KEY",
    "SURROGATE_ARCHIVE_JSONL_NAME",
    "active_archive_path",
    "checkpoint_dirs_for_archive",
    "checkpoint_paths_near_archive",
    "config_path",
    "configured_archive_paths",
    "configured_checkpoint_candidates",
    "existing_archive_paths",
    "load_config",
    "repo_root",
    "resolve_repo_path",
    "resolve_surrogate_archive_path",
    "resolve_surrogate_buffer_path",
    "resolve_surrogate_checkpoint_path",
    "sort_archive_paths_by_mtime",
    "surrogate_archive_path_for_map_elites_archive",
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


def surrogate_archive_path_for_map_elites_archive(archive_path: Path) -> Path:
    """Return expected co-located SurrogateArchive path for one MAP-Elites JSONL."""
    return archive_path.resolve().parent / SURROGATE_ARCHIVE_JSONL_NAME


def _optional_configured_surrogate_archive_path(cfg: dict[str, Any]) -> Path | None:
    paths_section = cfg.get("paths")
    if not isinstance(paths_section, dict):
        return None
    raw = paths_section.get("surrogate_archive")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return resolve_repo_path(raw)


def _configured_surrogate_archive_path(cfg: dict[str, Any]) -> Path:
    configured = _optional_configured_surrogate_archive_path(cfg)
    if configured is None:
        msg = "paths.surrogate_archive must be a non-empty string"
        raise KeyError(msg)
    return configured


def resolve_surrogate_archive_path(
    cfg: dict[str, Any] | None = None,
    *,
    archive_path: Path | None = None,
) -> Path:
    """Resolve SurrogateArchive JSONL: co-located with archive first, then config."""
    config = cfg if cfg is not None else load_config()

    if archive_path is not None and archive_path.is_file():
        co_located = surrogate_archive_path_for_map_elites_archive(archive_path)
        if co_located.is_file():
            return co_located

    configured = _optional_configured_surrogate_archive_path(config)
    if configured is not None and configured.is_file():
        return configured

    if archive_path is not None and archive_path.is_file():
        return surrogate_archive_path_for_map_elites_archive(archive_path)
    if configured is not None:
        return configured
    raise KeyError("paths.surrogate_archive must be a non-empty string")


def checkpoint_dirs_for_archive(archive_path: Path) -> list[Path]:
    """Return ``checkpoints/`` directories to search relative to an archive JSONL."""
    parent = archive_path.resolve().parent
    dirs: list[Path] = [parent / "checkpoints"]
    if parent.name in ("baseline", "surrogate"):
        run_root = parent.parent
        dirs.append(run_root / "checkpoints")
    dirs.extend(_artifact_bundle_checkpoint_dirs(archive_path))
    unique: list[Path] = []
    seen: set[str] = set()
    for directory in dirs:
        key = str(directory.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(directory)
    return unique


def _artifact_bundle_checkpoint_dirs(archive_path: Path) -> list[Path]:
    """Also search ``artifacts/*/checkpoints`` (e.g. downloaded GHA surrogate bundle)."""
    resolved = archive_path.resolve()
    parts = resolved.parts
    if "artifacts" not in parts:
        return []
    artifacts_root = Path(*parts[: parts.index("artifacts") + 1])
    if not artifacts_root.is_dir():
        return []
    dirs: list[Path] = []
    for child in sorted(artifacts_root.iterdir()):
        if child.is_dir():
            dirs.append(child / "checkpoints")
    return dirs


def checkpoint_paths_near_archive(archive_path: Path) -> list[Path]:
    """List checkpoint pickle paths beside the archive run (nearest dirs first)."""
    seen: set[str] = set()
    found: list[Path] = []
    for directory in checkpoint_dirs_for_archive(archive_path):
        if not directory.is_dir():
            continue
        for name in _CHECKPOINT_FILENAMES:
            candidate = directory / name
            resolved = str(candidate.resolve())
            if candidate.is_file() and resolved not in seen:
                seen.add(resolved)
                found.append(candidate)
        for candidate in sorted(directory.glob("*.pkl")):
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(candidate)
    return found


def configured_checkpoint_candidates(cfg: dict[str, Any] | None = None) -> list[Path]:
    """Return configured surrogate checkpoint paths (may be missing on disk)."""
    config = cfg if cfg is not None else load_config()
    paths_section = config.get("paths")
    surrogate_section = config.get("surrogate")
    relative_candidates: list[str] = []
    if isinstance(paths_section, dict):
        primary = paths_section.get("surrogate_checkpoint")
        if isinstance(primary, str) and primary.strip():
            relative_candidates.append(primary.strip())
    if isinstance(surrogate_section, dict):
        fallbacks = surrogate_section.get("checkpoint_fallbacks")
        if isinstance(fallbacks, list):
            for item in fallbacks:
                if isinstance(item, str) and item.strip():
                    relative_candidates.append(item.strip())
        legacy = surrogate_section.get("micro_checkpoint_fallback")
        if isinstance(legacy, str) and legacy.strip():
            relative_candidates.append(legacy.strip())
    seen: set[str] = set()
    resolved: list[Path] = []
    for relative in relative_candidates:
        if relative in seen:
            continue
        seen.add(relative)
        resolved.append(resolve_repo_path(relative))
    return resolved


def resolve_surrogate_checkpoint_path(
    cfg: dict[str, Any] | None = None,
    *,
    archive_path: Path | None = None,
) -> Path | None:
    """Resolve surrogate checkpoint: archive-adjacent ``checkpoints/`` first, then config."""
    config = cfg if cfg is not None else load_config()
    candidates: list[Path] = []
    archive = archive_path
    if archive is None:
        archive = active_archive_path(config)
    if archive is not None and archive.is_file():
        candidates.extend(checkpoint_paths_near_archive(archive))
    candidates.extend(configured_checkpoint_candidates(config))
    seen: set[str] = set()
    for candidate in candidates:
        resolved_key = str(candidate.resolve())
        if resolved_key in seen:
            continue
        seen.add(resolved_key)
        if candidate.is_file():
            return candidate
    return None


def active_archive_path(cfg: dict[str, Any] | None = None) -> Path | None:
    """Best-effort current archive JSONL from Streamlit session or config discovery."""
    config = cfg if cfg is not None else load_config()
    try:
        import streamlit as st

        session_keys = (
            DASHBOARD_ARCHIVE_SESSION_KEY,
            "surrogate_archive_select",
            "metrics_archive_select",
            "llm_prompt_archive",
        )
        for key in session_keys:
            value = st.session_state.get(key)
            if isinstance(value, Path) and value.is_file():
                return value
        explorer_path = st.session_state.get("explorer_archive_path")
        if isinstance(explorer_path, str):
            path = Path(explorer_path)
            if path.is_file():
                return path
    except (ImportError, RuntimeError, AttributeError):
        pass
    archives = existing_archive_paths(config)
    return archives[0] if archives else None


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

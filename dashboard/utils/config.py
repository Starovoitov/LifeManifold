"""Load dashboard YAML config and resolve repo-relative paths."""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any, Final, Literal

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.yaml"

DASHBOARD_ARCHIVE_SESSION_KEY = "dashboard_archive_select"
DASHBOARD_SURROGATE_CHECKPOINT_SESSION_KEY = "dashboard_surrogate_checkpoint_select"
DASHBOARD_BUFFER_SESSION_KEY = "dashboard_buffer_select"

MAP_ELITES_ARCHIVE_JSONL = "map_elites_archive.jsonl"

SURROGATE_ARCHIVE_JSONL_NAME = "surrogate_archive.jsonl"

CHECKPOINT_STUB_VALUE = ""
ARTIFACT_SEARCH_MAX_DEPTH = 4
CHECKPOINT_SEARCH_MAX_DEPTH = ARTIFACT_SEARCH_MAX_DEPTH
BUFFER_SEARCH_MAX_DEPTH = ARTIFACT_SEARCH_MAX_DEPTH


class UnsetType:
    """Marker for omitted optional values (distinct from explicit ``None``)."""

    __slots__ = ()


UNSET: Final = UnsetType()

__all__ = [
    "CHECKPOINT_STUB_VALUE",
    "ARTIFACT_SEARCH_MAX_DEPTH",
    "BUFFER_SEARCH_MAX_DEPTH",
    "CHECKPOINT_SEARCH_MAX_DEPTH",
    "MAP_ELITES_ARCHIVE_JSONL",
    "UNSET",
    "UnsetType",
    "DASHBOARD_ARCHIVE_SESSION_KEY",
    "DASHBOARD_BUFFER_SESSION_KEY",
    "DASHBOARD_SURROGATE_CHECKPOINT_SESSION_KEY",
    "SURROGATE_ARCHIVE_JSONL_NAME",
    "archive_adjacent_surrogate_checkpoint",
    "active_archive_path",
    "buffer_paths_near_archive",
    "buffer_search_roots",
    "buffer_session_key",
    "checkpoint_search_roots",
    "checkpoint_paths_near_archive",
    "checkpoint_session_key",
    "config_path",
    "existing_archive_paths",
    "list_surrogate_buffer_candidates",
    "list_surrogate_checkpoint_candidates",
    "load_config",
    "repo_root",
    "resolve_repo_path",
    "resolve_surrogate_archive_path",
    "resolve_surrogate_buffer_path",
    "resolve_surrogate_checkpoint_path",
    "session_surrogate_checkpoint_override",
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


def sort_archive_paths_by_mtime(entries: list[tuple[Path, float]]) -> list[Path]:
    """Sort by cached mtimes; drop paths that no longer exist (no second ``stat()``)."""
    ordered = sorted(entries, key=lambda item: item[1], reverse=True)
    return [path for path, _ in ordered if path.is_file()]


def existing_archive_paths(cfg: dict[str, Any] | None = None) -> list[Path]:
    """Archives for sidebar pickers discovered under ``paths.run_scan_dirs``."""
    config = cfg if cfg is not None else load_config()
    seen: set[str] = set()
    by_mtime: list[tuple[Path, float]] = []

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


def resolve_surrogate_archive_path(
    cfg: dict[str, Any] | None = None,
    *,
    archive_path: Path | None = None,
) -> Path:
    """Return SurrogateArchive JSONL path co-located with the selected MAP-Elites archive."""
    del cfg
    archive = archive_path if archive_path is not None else active_archive_path()
    if archive is None or not archive.is_file():
        msg = "Select a MAP-Elites archive JSONL first"
        raise KeyError(msg)
    return surrogate_archive_path_for_map_elites_archive(archive)


def buffer_search_roots(archive_path: Path) -> list[Path]:
    """Directories to scan for training buffer JSONL near an archive."""
    return checkpoint_search_roots(archive_path)


def _buffer_sort_key(
    buffer_path: Path,
    *,
    root_index: int,
) -> tuple[int, int, float, str]:
    name_rank = 0 if buffer_path.name == "buffer.jsonl" else 1
    try:
        mtime = float(buffer_path.stat().st_mtime)
    except OSError:
        mtime = 0.0
    return (root_index, name_rank, -mtime, str(buffer_path.resolve()))


def buffer_paths_near_archive(archive_path: Path) -> list[Path]:
    """List ``*buffer*.jsonl`` files under run dirs that contain the archive."""
    seen: set[str] = set()
    ranked: list[tuple[tuple[int, int, float, str], Path]] = []
    for root_index, root in enumerate(buffer_search_roots(archive_path)):
        if not root.is_dir():
            continue
        with suppress(OSError):
            for candidate in _iter_matching_files(
                root,
                pattern="*buffer*.jsonl",
                max_depth=BUFFER_SEARCH_MAX_DEPTH,
            ):
                if not _is_existing_file(candidate):
                    continue
                resolved = str(candidate.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                ranked.append(
                    (_buffer_sort_key(candidate, root_index=root_index), candidate)
                )
    ranked.sort(key=lambda item: item[0])
    return [path for _, path in ranked]


def buffer_session_key(archive_path: Path) -> str:
    """Streamlit session key for per-archive buffer override."""
    return f"{DASHBOARD_BUFFER_SESSION_KEY}:{archive_path.resolve()}"


def list_surrogate_buffer_candidates(archive_path: Path) -> list[Path]:
    """Training buffer JSONL files discovered beside the selected archive."""
    return buffer_paths_near_archive(archive_path)


def checkpoint_search_roots(archive_path: Path) -> list[Path]:
    """Directories to scan recursively for ``.pkl`` near an archive JSONL."""
    archive_path = archive_path.resolve()
    run_dir = archive_path.parent
    roots: list[Path] = [run_dir]
    if (run_dir / "nightly_run_summary.json").is_file():
        return roots
    parent = run_dir.parent
    if parent != run_dir and (
        (parent / "nightly_run_summary.json").is_file()
        or (parent / "checkpoints").is_dir()
    ):
        roots.append(parent)
    return roots


def _checkpoint_sort_key(
    pkl_path: Path,
    *,
    root_index: int,
) -> tuple[int, float, str]:
    try:
        mtime = float(pkl_path.stat().st_mtime)
    except OSError:
        mtime = 0.0
    return (root_index, -mtime, str(pkl_path.resolve()))


def _iter_matching_files(
    root: Path,
    *,
    pattern: str,
    max_depth: int,
) -> Iterator[Path]:
    """Yield files under ``root`` whose names match ``pattern`` up to ``max_depth`` levels."""
    resolved_root = root.resolve()
    depth_limit = max(0, max_depth)
    for current, dirnames, filenames in os.walk(
        resolved_root,
        topdown=True,
    ):
        current_path = Path(current)
        try:
            rel_depth = len(current_path.relative_to(resolved_root).parts)
        except ValueError:
            continue
        if rel_depth >= depth_limit:
            dirnames.clear()
        for name in filenames:
            if fnmatch.fnmatch(name, pattern):
                yield current_path / name


def _iter_pkl_files(root: Path, *, max_depth: int) -> Iterator[Path]:
    """Yield ``.pkl`` files under ``root`` up to ``max_depth`` subdirectory levels."""
    yield from _iter_matching_files(root, pattern="*.pkl", max_depth=max_depth)


def _is_existing_file(path: Path) -> bool:
    with suppress(OSError):
        return path.is_file()
    return False


def checkpoint_paths_near_archive(archive_path: Path) -> list[Path]:
    """List loadable ``SurrogateModel`` pickles under run dirs that contain the archive."""
    from dashboard.utils.surrogate_checkpoint import filter_surrogate_model_checkpoints

    seen: set[str] = set()
    ranked: list[tuple[tuple[int, float, str], Path]] = []
    for root_index, root in enumerate(checkpoint_search_roots(archive_path)):
        if not root.is_dir():
            continue
        with suppress(OSError):
            for candidate in _iter_pkl_files(
                root,
                max_depth=CHECKPOINT_SEARCH_MAX_DEPTH,
            ):
                if not _is_existing_file(candidate):
                    continue
                resolved = str(candidate.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                ranked.append(
                    (_checkpoint_sort_key(candidate, root_index=root_index), candidate)
                )
    ranked.sort(key=lambda item: item[0])
    return filter_surrogate_model_checkpoints([path for _, path in ranked])


def checkpoint_session_key(archive_path: Path) -> str:
    """Streamlit session key for per-archive surrogate checkpoint override."""
    return f"{DASHBOARD_SURROGATE_CHECKPOINT_SESSION_KEY}:{archive_path.resolve()}"


def archive_adjacent_surrogate_checkpoint(archive_path: Path) -> Path | None:
    """First loadable ``SurrogateModel`` checkpoint beside the archive run directory."""
    for candidate in checkpoint_paths_near_archive(archive_path):
        return candidate
    return None


def list_surrogate_checkpoint_candidates(
    cfg: dict[str, Any] | None = None,
    *,
    archive_path: Path | None = None,
) -> list[Path]:
    """Loadable ``SurrogateModel`` checkpoints discovered beside the selected archive."""
    del cfg
    archive = archive_path
    if archive is None:
        archive = active_archive_path()
    if archive is None or not archive.is_file():
        return []
    return checkpoint_paths_near_archive(archive)


def session_surrogate_checkpoint_override(
    archive_path: Path,
) -> Path | None | Literal["stub"]:
    """Return user-selected checkpoint path, explicit stub, or unset (``None``)."""
    try:
        import streamlit as st

        raw = st.session_state.get(checkpoint_session_key(archive_path), UNSET)
    except (ImportError, RuntimeError, AttributeError):
        return None
    if raw is UNSET:
        return None
    if raw == CHECKPOINT_STUB_VALUE:
        return "stub"
    if not isinstance(raw, str) or not raw.strip():
        return "stub"
    candidate = Path(raw)
    from dashboard.utils.surrogate_checkpoint import is_surrogate_model_checkpoint

    return candidate if is_surrogate_model_checkpoint(candidate) else None


def resolve_surrogate_checkpoint_path(
    cfg: dict[str, Any] | None = None,
    *,
    archive_path: Path | None = None,
) -> Path | None:
    """Resolve surrogate checkpoint beside the archive only (no global auto-fallback)."""
    config = cfg if cfg is not None else load_config()
    archive = archive_path
    if archive is None:
        archive = active_archive_path(config)
    if archive is None or not archive.is_file():
        return None
    override = session_surrogate_checkpoint_override(archive)
    if override == "stub":
        return None
    if isinstance(override, Path):
        return override
    return archive_adjacent_surrogate_checkpoint(archive)


def active_archive_path(cfg: dict[str, Any] | None = None) -> Path | None:
    """Best-effort current archive JSONL from Streamlit session or discovery."""
    del cfg
    try:
        import streamlit as st

        value = st.session_state.get(DASHBOARD_ARCHIVE_SESSION_KEY)
        if isinstance(value, Path) and value.is_file():
            return value
        if isinstance(value, str) and value.strip():
            path = Path(value)
            if path.is_file():
                return path
        explorer_path = st.session_state.get("explorer_archive_path")
        if isinstance(explorer_path, str):
            path = Path(explorer_path)
            if path.is_file():
                return path
    except (ImportError, RuntimeError, AttributeError):
        pass
    archives = existing_archive_paths()
    return archives[0] if archives else None


def resolve_surrogate_buffer_path(
    cfg: dict[str, Any] | None = None,
    *,
    archive_path: Path | None = None,
) -> Path | None:
    """Resolve training buffer JSONL from sidebar selection or archive-local discovery."""
    del cfg
    archive = archive_path if archive_path is not None else active_archive_path()
    if archive is None or not archive.is_file():
        return None
    try:
        import streamlit as st

        raw = st.session_state.get(buffer_session_key(archive))
        if isinstance(raw, str) and raw.strip():
            candidate = Path(raw)
            if candidate.is_file():
                return candidate
    except (ImportError, RuntimeError, AttributeError):
        pass
    candidates = list_surrogate_buffer_candidates(archive)
    return candidates[0] if candidates else None

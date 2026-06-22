"""Load dashboard YAML config and resolve repo-relative paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.yaml"

DASHBOARD_ARCHIVE_SESSION_KEY = "dashboard_archive_select"
DASHBOARD_SURROGATE_CHECKPOINT_SESSION_KEY = "dashboard_surrogate_checkpoint_select"

MAP_ELITES_ARCHIVE_JSONL = "map_elites_archive.jsonl"

SURROGATE_ARCHIVE_JSONL_NAME = "surrogate_archive.jsonl"

_UNSET = object()
CHECKPOINT_STUB_VALUE = ""

__all__ = [
    "CHECKPOINT_STUB_VALUE",
    "DASHBOARD_ARCHIVE_SESSION_KEY",
    "DASHBOARD_SURROGATE_CHECKPOINT_SESSION_KEY",
    "SURROGATE_ARCHIVE_JSONL_NAME",
    "archive_adjacent_surrogate_checkpoint",
    "active_archive_path",
    "checkpoint_search_roots",
    "checkpoint_paths_near_archive",
    "checkpoint_session_key",
    "config_path",
    "configured_archive_paths",
    "configured_checkpoint_candidates",
    "existing_archive_paths",
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


def checkpoint_paths_near_archive(archive_path: Path) -> list[Path]:
    """List loadable ``SurrogateModel`` pickles under run dirs that contain the archive."""
    from dashboard.utils.surrogate_checkpoint import filter_surrogate_model_checkpoints

    seen: set[str] = set()
    ranked: list[tuple[tuple[int, float, str], Path]] = []
    for root_index, root in enumerate(checkpoint_search_roots(archive_path)):
        if not root.is_dir():
            continue
        try:
            iterator = root.rglob("*.pkl")
            while True:
                try:
                    candidate = next(iterator)
                except StopIteration:
                    break
                except OSError:
                    continue
                try:
                    if not candidate.is_file():
                        continue
                except OSError:
                    continue
                resolved = str(candidate.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                ranked.append(
                    (_checkpoint_sort_key(candidate, root_index=root_index), candidate)
                )
        except OSError:
            continue
    ranked.sort(key=lambda item: item[0])
    return filter_surrogate_model_checkpoints([path for _, path in ranked])


def checkpoint_session_key(archive_path: Path) -> str:
    """Streamlit session key for per-archive surrogate checkpoint override."""
    return f"{DASHBOARD_SURROGATE_CHECKPOINT_SESSION_KEY}:{archive_path.resolve()}"


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
    """Loadable ``SurrogateModel`` checkpoints: beside archive first, then configured paths."""
    from dashboard.utils.surrogate_checkpoint import filter_surrogate_model_checkpoints

    config = cfg if cfg is not None else load_config()
    archive = archive_path
    if archive is None:
        archive = active_archive_path(config)
    ordered: list[Path] = []
    seen: set[str] = set()
    if archive is not None and archive.is_file():
        for candidate in checkpoint_paths_near_archive(archive):
            key = str(candidate.resolve())
            if key not in seen:
                seen.add(key)
                ordered.append(candidate)
    for candidate in configured_checkpoint_candidates(config):
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)
    return filter_surrogate_model_checkpoints(ordered)


def session_surrogate_checkpoint_override(
    archive_path: Path,
) -> Path | None | Literal["stub"]:
    """Return user-selected checkpoint path, explicit stub, or unset (``None``)."""
    try:
        import streamlit as st

        raw = st.session_state.get(checkpoint_session_key(archive_path), _UNSET)
    except (ImportError, RuntimeError, AttributeError):
        return None
    if raw is _UNSET:
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

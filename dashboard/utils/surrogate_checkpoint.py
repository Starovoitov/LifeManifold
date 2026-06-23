"""Validate surrogate model checkpoint pickles for the dashboard."""

from __future__ import annotations

import functools
from pathlib import Path

from dashboard.utils.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from worldspace.surrogate.checkpoint_io import (  # noqa: E402
    CHECKPOINT_LOAD_ERRORS,
    load_surrogate_checkpoint,
)

__all__ = [
    "filter_surrogate_model_checkpoints",
    "is_surrogate_model_checkpoint",
]


def _checkpoint_stat_key(path: Path) -> tuple[str, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path.resolve()), int(stat.st_mtime_ns))


@functools.lru_cache(maxsize=256)
def _is_surrogate_model_checkpoint_cached(stat_key: tuple[str, int]) -> bool:
    path = Path(stat_key[0])
    try:
        load_surrogate_checkpoint(path)
    except CHECKPOINT_LOAD_ERRORS:
        return False
    return True


def is_surrogate_model_checkpoint(path: Path) -> bool:
    """Return True when ``path`` is a readable pickle containing ``SurrogateModel``."""
    if not path.is_file():
        return False
    stat_key = _checkpoint_stat_key(path)
    if stat_key is None:
        return False
    return _is_surrogate_model_checkpoint_cached(stat_key)


def filter_surrogate_model_checkpoints(paths: list[Path]) -> list[Path]:
    """Keep only loadable ``SurrogateModel`` checkpoints, preserving order."""
    found: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if not is_surrogate_model_checkpoint(path):
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        found.append(path)
    return found

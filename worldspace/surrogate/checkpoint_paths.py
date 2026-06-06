"""Shared surrogate checkpoint path helpers."""

from __future__ import annotations

from pathlib import Path

STUB_CHECKPOINT_SENTINEL = "__stub__"

__all__ = [
    "STUB_CHECKPOINT_SENTINEL",
    "is_stub_checkpoint",
    "resolve_runtime_checkpoint_path",
]


def is_stub_checkpoint(value: str | Path | None) -> bool:
    """Return whether ``value`` is the runtime stub sentinel, not a real path."""
    if value is None:
        return False
    return str(value).strip() == STUB_CHECKPOINT_SENTINEL


def resolve_runtime_checkpoint_path(value: str | Path | None) -> Path | None:
    """Map scheduler checkpoint strings to a loadable path, if any."""
    if not value or is_stub_checkpoint(value):
        return None
    return Path(value).expanduser()

"""Ensure the repository root is on ``sys.path`` for ``worldspace`` imports."""

from __future__ import annotations

from pathlib import Path

from path_setup import install_paths

_DASHBOARD_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _DASHBOARD_DIR.parent

__all__ = ["ensure_repo_on_path", "repo_root"]


def repo_root() -> Path:
    """Return the LifeManifold repository root."""
    return _REPO_ROOT


def ensure_repo_on_path() -> Path:
    """Insert repo root and ``dashboard/`` on ``sys.path`` for imports."""
    return install_paths(__file__)

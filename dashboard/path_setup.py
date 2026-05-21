"""Stdlib-only sys.path setup before ``import dashboard`` (Streamlit entry points)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

__all__ = ["install_paths"]


def install_paths(from_file: PathLike) -> Path:
    """Add repository root and ``dashboard/`` to ``sys.path`` for package imports.

    Call this at the top of ``Home.py`` and every ``pages/*.py`` script, using
    ``__file__`` as ``from_file``, before any ``from dashboard...`` imports.
    """
    script = Path(from_file).resolve()
    if script.parent.name == "pages":
        dashboard_dir = script.parent.parent
    else:
        dashboard_dir = script.parent
    repo_root = dashboard_dir.parent
    for entry in (str(repo_root), str(dashboard_dir)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return repo_root

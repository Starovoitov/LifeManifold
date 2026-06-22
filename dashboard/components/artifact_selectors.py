"""Shared sidebar selectors for dashboard JSONL artifacts."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.utils.config import (
    DASHBOARD_ARCHIVE_SESSION_KEY,
    existing_archive_paths,
    list_surrogate_buffer_candidates,
    load_config,
    repo_root,
)

__all__ = [
    "format_repo_relative_path",
    "render_archive_selector",
    "render_buffer_selector",
]


def format_repo_relative_path(path: Path) -> str:
    """Human-readable path for selectbox labels."""
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        return str(path)


def render_archive_selector(cfg: dict | None = None) -> Path | None:
    """Sidebar MAP-Elites archive picker shared across dashboard pages."""
    config = cfg if cfg is not None else load_config()
    archives = existing_archive_paths(config)
    if not archives:
        st.sidebar.error("No archive JSONL found under scan roots.")
        return None
    selected = st.sidebar.selectbox(
        "Archive JSONL",
        archives,
        format_func=format_repo_relative_path,
        key=DASHBOARD_ARCHIVE_SESSION_KEY,
    )
    return selected


def render_buffer_selector(
    archive_path: Path,
    cfg: dict | None = None,
) -> Path | None:
    """Sidebar surrogate training buffer picker near the selected archive."""
    candidates = list_surrogate_buffer_candidates(archive_path)
    if not candidates:
        st.sidebar.warning("No buffer JSONL found near the selected archive.")
        return None

    option_values = [str(path) for path in candidates]

    def _label(value: str) -> str:
        path = Path(value)
        display = format_repo_relative_path(path)
        if path.is_symlink():
            target = path.resolve()
            return f"{display} -> {format_repo_relative_path(target)}"
        return display

    selected_value = st.sidebar.selectbox(
        "Training buffer JSONL",
        option_values,
        format_func=_label,
        key=f"dashboard_buffer_select:{archive_path.resolve()}",
    )
    selected_path = Path(selected_value)
    return selected_path if selected_path.is_file() else None

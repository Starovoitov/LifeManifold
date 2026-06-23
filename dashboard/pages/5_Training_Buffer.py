"""Surrogate training buffer statistics and paginated record viewer."""

from __future__ import annotations

import sys
from pathlib import Path

_dashboard_dir = Path(__file__).resolve().parent.parent
if str(_dashboard_dir) not in sys.path:
    sys.path.insert(0, str(_dashboard_dir))

import path_setup

path_setup.install_paths(__file__)

import streamlit as st

from dashboard.components.artifact_selectors import (
    format_repo_relative_path,
    render_archive_selector,
    render_buffer_selector,
)
from dashboard.components.training_buffer_loader import (
    apply_buffer_filters,
    get_buffer_bundle,
    show_large_buffer_warning,
)
from dashboard.components.training_buffer_view import (
    render_buffer_export,
    render_buffer_filters,
    render_buffer_stats,
    render_buffer_table,
)
from dashboard.utils.config import load_config

st.set_page_config(page_title="Training Buffer", layout="wide")
st.title("Training Buffer")
st.caption("View surrogate training buffer JSONL (read-only; no training from UI).")

cfg = load_config()

selected_archive = render_archive_selector(cfg)
if selected_archive is None:
    st.error("No archive JSONL found under scan roots.")
    st.stop()

buffer_path = render_buffer_selector(selected_archive, cfg)
if buffer_path is None:
    st.error("No training buffer JSONL found near the selected archive.")
    st.info(
        "Expected `buffer.jsonl` or other `*buffer*.jsonl` in the run directory "
        "or parent when checkpoints exist."
    )
    st.stop()

st.sidebar.caption(format_repo_relative_path(buffer_path))

try:
    bundle = get_buffer_bundle(buffer_path)
except (FileNotFoundError, ValueError, TypeError, KeyError) as exc:
    st.error(f"Failed to load buffer: {exc}")
    st.stop()

show_large_buffer_warning(bundle, cfg)

if bundle.records.empty:
    st.warning("Buffer file has no valid training records.")
    st.stop()

render_buffer_stats(bundle)
filter_state = render_buffer_filters(bundle)
filtered, filtered_raw = apply_buffer_filters(
    bundle,
    emitter_types=filter_state.emitter_types,
    schema_versions=filter_state.schema_versions,
)

performance = cfg.get("performance")
table_max_rows = 500
if isinstance(performance, dict) and "table_max_rows" in performance:
    table_max_rows = max(1, int(performance["table_max_rows"]))

render_buffer_table(filtered, table_max_rows=table_max_rows)
render_buffer_export(filtered_raw, source_name=buffer_path.stem)

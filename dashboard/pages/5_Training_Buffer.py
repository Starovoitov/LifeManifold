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
from dashboard.utils.config import load_config, repo_root, resolve_surrogate_buffer_path

st.set_page_config(page_title="Training Buffer", layout="wide")
st.title("Training Buffer")
st.caption("View surrogate training buffer JSONL (read-only; no training from UI).")

cfg = load_config()

try:
    buffer_path = resolve_surrogate_buffer_path(cfg)
except KeyError as exc:
    st.error(f"Dashboard config error: {exc}")
    st.stop()

if not buffer_path.is_file():
    st.error(f"Buffer file not found: {buffer_path.relative_to(repo_root())}")
    st.info(
        "Run MAP-Elites with surrogate buffer enabled or point "
        "`paths.surrogate_buffer` in dashboard config to an existing JSONL file."
    )
    st.stop()

st.sidebar.caption(str(buffer_path.relative_to(repo_root())))

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

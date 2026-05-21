"""Metric correlations and distributions across MAP-Elites archives."""

from __future__ import annotations

import sys
from pathlib import Path

_dashboard_dir = Path(__file__).resolve().parent.parent
if str(_dashboard_dir) not in sys.path:
    sys.path.insert(0, str(_dashboard_dir))

import path_setup

path_setup.install_paths(__file__)

import streamlit as st

from dashboard.utils.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from dashboard.components.archive_loader import (
    get_archive_bundle,
    show_large_archive_warning,
)
from dashboard.components.filters import apply_collapsed_filters, render_archive_filters
from dashboard.components.metrics_dashboard import (
    render_correlation_section,
    render_distributions_section,
)
from dashboard.utils.config import existing_archive_paths, load_config, repo_root

st.set_page_config(page_title="Metrics Dashboard", layout="wide")
st.title("Metrics Dashboard")
st.caption(
    "Correlations and distributions over collapsed archive elites (in-memory filters)."
)

cfg = load_config()
archives = existing_archive_paths(cfg)
if not archives:
    st.error("No archive JSONL found. Update dashboard config or run MAP-Elites.")
    st.stop()


def _archive_label(path: Path) -> str:
    return str(path.relative_to(repo_root()))


selected_path = st.sidebar.selectbox(
    "Archive JSONL",
    archives,
    format_func=_archive_label,
    key="metrics_archive_select",
)

bundle = get_archive_bundle(selected_path)
show_large_archive_warning(bundle, cfg)
filter_state = render_archive_filters(bundle, selected_path)
filtered = apply_collapsed_filters(bundle.collapsed, filter_state)

st.metric("Collapsed elites", len(bundle.collapsed))
st.metric("After filters", len(filtered))

if filtered.empty:
    st.warning("No elites match the current filters.")
    st.stop()

render_correlation_section(filtered)
render_distributions_section(filtered)

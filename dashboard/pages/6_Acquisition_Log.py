"""Surrogate acquisition decision log (SurrogateArchive JSONL)."""

from __future__ import annotations

import sys
from pathlib import Path

_dashboard_dir = Path(__file__).resolve().parent.parent
if str(_dashboard_dir) not in sys.path:
    sys.path.insert(0, str(_dashboard_dir))

import path_setup

path_setup.install_paths(__file__)

import streamlit as st

from dashboard.components.surrogate_acquisition_view import (
    render_acquisition_charts,
    render_acquisition_filters,
    render_acquisition_kpis,
    render_acquisition_table,
)
from dashboard.components.surrogate_archive_loader import (
    apply_archive_log_filters,
    get_archive_log_bundle,
)
from dashboard.components.artifact_selectors import render_archive_selector
from dashboard.utils.config import (
    load_config,
    repo_root,
    resolve_surrogate_archive_path,
    surrogate_archive_path_for_map_elites_archive,
)

st.set_page_config(page_title="Acquisition Log", layout="wide")
st.title("Acquisition Log")
st.caption(
    "SurrogateArchive JSONL: per-slot predictions and eval/skip decisions "
    "(read-only)."
)

cfg = load_config()

selected_archive = render_archive_selector(cfg)
if selected_archive is None:
    st.error("No archive JSONL found under scan roots.")
    st.stop()

try:
    log_path = resolve_surrogate_archive_path(cfg, archive_path=selected_archive)
except KeyError as exc:
    st.error(str(exc))
    st.stop()

if not log_path.is_file():
    co_located = surrogate_archive_path_for_map_elites_archive(selected_archive)
    st.error(f"Surrogate archive not found: {log_path.relative_to(repo_root())}")
    st.info(
        "Expected acquisition log next to the selected archive at "
        f"`{co_located.relative_to(repo_root())}` when the illuminator run used "
        "`acquisition.mode: shadow` or `filter`."
    )
    st.stop()

st.sidebar.caption(str(log_path.relative_to(repo_root())))

try:
    bundle = get_archive_log_bundle(log_path)
except (FileNotFoundError, ValueError, TypeError, KeyError) as exc:
    st.error(f"Failed to load surrogate archive: {exc}")
    st.stop()

if bundle.invalid_line_count:
    st.warning(f"Skipped {bundle.invalid_line_count} invalid JSONL line(s).")

if bundle.records.empty:
    st.warning("Archive file has no valid acquisition records.")
    st.stop()

filter_state = render_acquisition_filters(bundle)
filtered = apply_archive_log_filters(
    bundle,
    decisions=filter_state.decisions,
    acquisition_modes=filter_state.acquisition_modes,
    emitter_types=filter_state.emitter_types,
    iteration_range=filter_state.iteration_range,
)

render_acquisition_kpis(filtered)

st.subheader("Eval savings by iteration")
render_acquisition_charts(filtered)

performance = cfg.get("performance")
table_max_rows = 500
if isinstance(performance, dict) and "table_max_rows" in performance:
    table_max_rows = max(1, int(performance["table_max_rows"]))

render_acquisition_table(filtered, table_max_rows=table_max_rows)

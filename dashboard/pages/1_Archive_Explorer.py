"""MAP-Elites archive explorer (heatmap, bin detail, diagnostic panel)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.utils.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from dashboard.components.archive_explorer import (
    render_diagnostic_panel,
    reset_explorer_session_for_archive,
    sync_selected_bin_selectbox,
)
from dashboard.components.archive_loader import (
    get_archive_bundle,
    show_large_archive_warning,
)
from dashboard.components.filters import (
    apply_collapsed_filters,
    rebuild_pivots_from_collapsed,
    render_archive_filters,
)
from dashboard.components.visualizations import create_archive_heatmap
from dashboard.utils.config import existing_archive_paths, load_config, repo_root

st.set_page_config(page_title="Archive Explorer", layout="wide")
st.title("Archive Explorer")

cfg = load_config()
archives = existing_archive_paths(cfg)
if not archives:
    st.error("No archive JSONL found. Run MAP-Elites smoke or update dashboard config.")
    st.stop()


def _archive_label(path: Path) -> str:
    return str(path.relative_to(repo_root()))


selected_path = st.sidebar.selectbox(
    "Archive JSONL",
    archives,
    format_func=_archive_label,
)

reset_explorer_session_for_archive(str(selected_path))

bundle = get_archive_bundle(selected_path)
show_large_archive_warning(bundle, cfg)

filter_state = render_archive_filters(bundle, selected_path)
filtered = apply_collapsed_filters(bundle.collapsed, filter_state)
filtered_pivots = rebuild_pivots_from_collapsed(
    filtered,
    list(bundle.pivots.keys()),
    bundle.resolution,
)

st.metric("Collapsed elites", len(bundle.collapsed))
st.metric("After filters", len(filtered))
st.metric("Raw JSONL lines", bundle.line_count_raw)

st.subheader("Filtered elites (preview)")
performance = cfg.get("performance")
perf = performance if isinstance(performance, dict) else {}
table_max = int(perf.get("table_max_rows", 500))
display_columns = [
    column
    for column in (
        "bin_x",
        "bin_y",
        "fitness",
        "stability",
        "diversity",
        "emitter_type",
        "seed",
    )
    if column in filtered.columns
]
st.dataframe(
    filtered[display_columns].head(table_max),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Archive heatmap")
metric = filter_state.heatmap_metric
pivot = filtered_pivots.get(metric)
if pivot is None:
    st.info("No pivot data for the selected metric.")
else:
    heatmap_fig = create_archive_heatmap(
        pivot=pivot,
        metric=metric,
        resolution=filter_state.resolution,
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)

if filtered.empty:
    st.info("No elites match the current filters.")
elif "world_spec" not in filtered.columns:
    st.warning("Archive rows lack world_spec; cannot run diagnostics.")
else:
    st.subheader("Bin selection")
    elite_row = sync_selected_bin_selectbox(filtered)
    if elite_row is not None:
        st.subheader("Diagnostic")
        render_diagnostic_panel(elite_row)

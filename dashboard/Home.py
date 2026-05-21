"""LifeManifold research dashboard — overview and navigation."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.bootstrap import ensure_repo_on_path
from dashboard.utils.config import existing_archive_paths, load_config, repo_root
from dashboard.utils.plotting import apply_dark_theme, default_figure_height

ensure_repo_on_path()

st.set_page_config(
    page_title="LifeManifold Dashboard",
    page_icon="🧬",
    layout="wide",
)

st.title("LifeManifold Research Dashboard")
st.caption("MAP-Elites archives, surrogate analysis, and metrics (JSONL sources only).")

cfg = load_config()
defaults = cfg.get("defaults") or {}
resolution = defaults.get("grid_resolution", 50)
st.metric("Default grid resolution", int(resolution))

st.subheader("Configured archives on disk")
archives = existing_archive_paths(cfg)
if archives:
    for path in archives:
        rel = path.relative_to(repo_root())
        st.success(f"`{rel}`")
else:
    st.warning(
        "No archive JSONL from config found. Run MAP-Elites smoke or point paths in config."
    )

st.subheader("Pages")
st.page_link("pages/1_Archive_Explorer.py", label="Archive Explorer")
st.page_link("pages/2_Surrogate_Analysis.py", label="Surrogate Analysis")
st.page_link("pages/3_Metrics_Dashboard.py", label="Metrics Dashboard")
st.page_link("pages/4_LLM_Prompt_Tester.py", label="LLM Prompt Tester")
st.page_link("pages/5_Training_Buffer.py", label="Training Buffer")

st.subheader("Theme preview")
fig = apply_dark_theme(go.Figure())
fig.update_layout(
    title="Plotly dark theme (E0)",
    height=default_figure_height(),
)
st.plotly_chart(fig, use_container_width=True)

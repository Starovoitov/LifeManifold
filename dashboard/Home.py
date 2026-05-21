"""LifeManifold research dashboard — overview and navigation."""

from __future__ import annotations

import sys
from pathlib import Path

_dashboard_dir = Path(__file__).resolve().parent
if str(_dashboard_dir) not in sys.path:
    sys.path.insert(0, str(_dashboard_dir))

import path_setup

path_setup.install_paths(__file__)

import streamlit as st

from dashboard.utils.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from dashboard.components.home_overview import render_page_links, render_runs_overview
from dashboard.utils.config import load_config
from dashboard.utils.run_discovery import discover_runs

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

runs = discover_runs(cfg)
render_runs_overview(runs)
render_page_links()

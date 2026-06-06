"""Surrogate real vs predicted analysis and calibration."""

from __future__ import annotations

import sys
from pathlib import Path

_dashboard_dir = Path(__file__).resolve().parent.parent
if str(_dashboard_dir) not in sys.path:
    sys.path.insert(0, str(_dashboard_dir))

import path_setup

path_setup.install_paths(__file__)

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.utils.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from dashboard.components.archive_loader import (
    get_archive_bundle,
    show_large_archive_warning,
)
from dashboard.components.filters import apply_collapsed_filters, render_archive_filters
from dashboard.components.surrogate_widget import (
    feature_importance_from_model,
    load_surrogate,
    predict_world_spec_dict,
    render_surrogate_status_banner,
    resolve_checkpoint_path,
    surrogate_model_from_handle,
)
from dashboard.components.visualizations import (
    plot_calibration_by_uncertainty,
    plot_real_vs_predicted,
)
from dashboard.utils.config import (
    DASHBOARD_ARCHIVE_SESSION_KEY,
    existing_archive_paths,
    load_config,
    repo_root,
)
from dashboard.utils.surrogate_analysis import (
    build_prediction_frame,
    regression_metrics,
    sample_collapsed_rows,
)


@st.cache_data(show_spinner=False)
def _cached_prediction_frame(
    archive_path_str: str,
    mtime: float,
    min_fitness: float,
    seed: int | None,
    emitter_type: str | None,
    max_rows: int,
    checkpoint_key: str,
) -> pd.DataFrame:
    del mtime, checkpoint_key
    bundle = get_archive_bundle(Path(archive_path_str))
    frame = bundle.collapsed
    mask = frame["fitness"] >= min_fitness
    if seed is not None and "seed" in frame.columns:
        mask &= frame["seed"] == seed
    if emitter_type is not None and "emitter_type" in frame.columns:
        mask &= frame["emitter_type"] == emitter_type
    filtered = frame.loc[mask].reset_index(drop=True)
    sample = sample_collapsed_rows(filtered, max_rows=max_rows, seed=0)
    return build_prediction_frame(sample, predict_world_spec_dict)


st.set_page_config(page_title="Surrogate Analysis", layout="wide")
st.title("Surrogate Analysis")

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
    key=DASHBOARD_ARCHIVE_SESSION_KEY,
)

status = render_surrogate_status_banner(cfg, archive_path=selected_path)

bundle = get_archive_bundle(selected_path)
show_large_archive_warning(bundle, cfg)
filter_state = render_archive_filters(bundle, selected_path)
filtered = apply_collapsed_filters(bundle.collapsed, filter_state)

performance = cfg.get("performance")
perf = performance if isinstance(performance, dict) else {}
max_rows = int(perf.get("table_max_rows", 500))

st.metric("Elites after filters", len(filtered))

if status.is_stub:
    st.stop()

prediction_frame = _cached_prediction_frame(
    str(selected_path.resolve()),
    float(selected_path.stat().st_mtime),
    filter_state.min_fitness,
    filter_state.seed,
    filter_state.emitter_type,
    max_rows,
    str(resolve_checkpoint_path(cfg, archive_path=selected_path) or ""),
)

if prediction_frame.empty:
    st.warning("No predictions generated. Check world_spec columns in the archive.")
    st.stop()

y_true = prediction_frame["fitness"].to_numpy(dtype=np.float64)
y_pred = prediction_frame["pred_fitness"].to_numpy(dtype=np.float64)
uncertainty = prediction_frame["pred_uncertainty"].to_numpy(dtype=np.float64)

mae, r2 = regression_metrics(y_true, y_pred)
col_mae, col_r2, col_n = st.columns(3)
col_mae.metric("MAE", f"{mae:.4f}")
col_r2.metric("R²", f"{r2:.4f}" if np.isfinite(r2) else "—")
col_n.metric("Points", len(prediction_frame))

st.subheader("Real vs predicted")
scatter_fig = plot_real_vs_predicted(
    y_true,
    y_pred,
    uncertainty,
    metric_name="fitness",
)
st.plotly_chart(scatter_fig, width="stretch")

st.subheader("Calibration by uncertainty")
if len(prediction_frame) < 3:
    st.info("Need at least three points for calibration bins.")
else:
    calibration_fig = plot_calibration_by_uncertainty(
        y_true,
        y_pred,
        uncertainty,
        n_bins=min(8, len(prediction_frame)),
    )
    st.plotly_chart(calibration_fig, width="stretch")

model = surrogate_model_from_handle(load_surrogate(cfg, archive_path=selected_path))
importances = feature_importance_from_model(model) if model is not None else None
if importances:
    st.subheader("Feature importance (LightGBM)")
    importance_frame = pd.DataFrame(
        [{"feature": key, "importance": value} for key, value in importances.items()]
    ).sort_values("importance", ascending=True)
    st.bar_chart(importance_frame.set_index("feature"))

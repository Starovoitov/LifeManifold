"""Archive Explorer helpers: bin selection state and diagnostic panel."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import streamlit as st

from dashboard.components.metrics import metrics_dict_from_row
from dashboard.components.visualizations import (
    DIAGNOSTIC_PANEL_HELP,
    create_diagnostic_dashboard,
    format_diagnostic_interpretation,
)
from dashboard.components.world_renderer import run_and_cache_world_from_dict
from dashboard.utils.bootstrap import ensure_repo_on_path
from dashboard.utils.data_processing import canonical_world_spec_hash

ensure_repo_on_path()

from worldspace.simulator import SimulationResult

SESSION_KEY_SELECTED_BIN = "explorer_selected_bin"
SESSION_KEY_SIMULATED_HASHES = "explorer_simulated_hashes"
SESSION_KEY_ARCHIVE_PATH = "explorer_archive_path"

__all__ = [
    "SESSION_KEY_ARCHIVE_PATH",
    "SESSION_KEY_SELECTED_BIN",
    "SESSION_KEY_SIMULATED_HASHES",
    "elite_row_for_bin",
    "format_elite_bin_label",
    "get_selected_elite_row",
    "list_bins_from_frame",
    "render_diagnostic_panel",
    "reset_explorer_session_for_archive",
    "run_cached_simulation_with_ui",
    "sync_selected_bin_selectbox",
]


def reset_explorer_session_for_archive(archive_path: str) -> None:
    """Clear bin/simulation session keys when the user switches archive JSONL."""
    previous = st.session_state.get(SESSION_KEY_ARCHIVE_PATH)
    if previous == archive_path:
        return
    st.session_state[SESSION_KEY_ARCHIVE_PATH] = archive_path
    st.session_state.pop(SESSION_KEY_SELECTED_BIN, None)
    st.session_state.pop(SESSION_KEY_SIMULATED_HASHES, None)


def list_bins_from_frame(frame: pd.DataFrame) -> list[tuple[int, int]]:
    """Return ``(bin_x, bin_y)`` pairs present in a collapsed archive frame."""
    if frame.empty or "bin_x" not in frame.columns or "bin_y" not in frame.columns:
        return []
    bins: list[tuple[int, int]] = []
    for _, row in frame.iterrows():
        bins.append((_row_int(row, "bin_x"), _row_int(row, "bin_y")))
    return bins


def elite_row_for_bin(
    frame: pd.DataFrame,
    bin_x: int,
    bin_y: int,
) -> pd.Series | None:
    """Lookup the collapsed elite row for a MAP-Elites bin."""
    if frame.empty:
        return None
    match = frame[(frame["bin_x"] == bin_x) & (frame["bin_y"] == bin_y)]
    if match.empty:
        return None
    return match.iloc[0]


def get_selected_elite_row(frame: pd.DataFrame) -> pd.Series | None:
    """Return the elite for ``SESSION_KEY_SELECTED_BIN``, if set and present."""
    selected = st.session_state.get(SESSION_KEY_SELECTED_BIN)
    if not isinstance(selected, (list, tuple)) or len(selected) != 2:
        return None
    return elite_row_for_bin(frame, int(selected[0]), int(selected[1]))


def format_elite_bin_label(frame: pd.DataFrame, bin_xy: tuple[int, int]) -> str:
    """Short label for bin selectors (bin coordinates + fitness)."""
    row = elite_row_for_bin(frame, bin_xy[0], bin_xy[1])
    if row is None:
        return f"bin ({bin_xy[0]}, {bin_xy[1]})"
    fitness = _row_float(row, "fitness") if "fitness" in row.index else float("nan")
    seed = _optional_row_int(row, "seed")
    parts = [f"bin ({bin_xy[0]}, {bin_xy[1]})", f"fitness={fitness:.4f}"]
    if seed is not None:
        parts.append(f"seed={seed}")
    return " · ".join(parts)


def _normalize_bin_xy(value: object) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    return None


def sync_selected_bin_selectbox(
    frame: pd.DataFrame,
    *,
    label: str = "Selected bin",
) -> pd.Series | None:
    """Bin picker wired to ``SESSION_KEY_SELECTED_BIN`` (one rerun per change)."""
    bins = list_bins_from_frame(frame)
    if not bins:
        return None

    current = _normalize_bin_xy(st.session_state.get(SESSION_KEY_SELECTED_BIN))
    if current is None or current not in bins:
        st.session_state[SESSION_KEY_SELECTED_BIN] = bins[0]

    chosen = st.selectbox(
        label,
        bins,
        format_func=lambda bin_xy: format_elite_bin_label(frame, bin_xy),
        key=SESSION_KEY_SELECTED_BIN,
    )
    bin_xy = _normalize_bin_xy(chosen) or bins[0]
    return elite_row_for_bin(frame, bin_xy[0], bin_xy[1])


def run_cached_simulation_with_ui(world_spec: dict[str, Any]) -> SimulationResult:
    """Run a cached simulation; show a spinner only on the first visit per spec hash."""
    spec_hash = canonical_world_spec_hash(world_spec)
    simulated = st.session_state.get(SESSION_KEY_SIMULATED_HASHES)
    if not isinstance(simulated, set):
        simulated = set()
        st.session_state[SESSION_KEY_SIMULATED_HASHES] = simulated

    if spec_hash in simulated:
        return run_and_cache_world_from_dict(world_spec)

    with st.spinner("Running world simulation…"):
        result = run_and_cache_world_from_dict(world_spec)
    simulated.add(spec_hash)
    return result


def render_diagnostic_panel(
    elite_row: Mapping[str, Any] | pd.Series,
    *,
    surrogate_pred: dict[str, float] | None = None,
) -> None:
    """Render the Plotly diagnostic dashboard for one archive elite."""
    row_dict = (
        elite_row.to_dict() if isinstance(elite_row, pd.Series) else dict(elite_row)
    )
    world_spec = row_dict.get("world_spec")
    if not isinstance(world_spec, dict):
        st.error("Selected elite has no valid world_spec.")
        return

    bin_x = int(row_dict["bin_x"]) if "bin_x" in row_dict else None
    bin_y = int(row_dict["bin_y"]) if "bin_y" in row_dict else None
    fitness = float(row_dict["fitness"]) if "fitness" in row_dict else float("nan")
    title_parts = ["Diagnostic"]
    if bin_x is not None and bin_y is not None:
        title_parts.append(f"bin ({bin_x}, {bin_y})")
    title_parts.append(f"fitness={fitness:.4f}")
    title = " — ".join(title_parts)

    spec_hash = canonical_world_spec_hash(world_spec)
    sim_result = run_cached_simulation_with_ui(world_spec)
    resolved_surrogate = surrogate_pred
    if resolved_surrogate is None:
        from dashboard.components.surrogate_widget import (
            predict_world_spec_dict,
            surrogate_status,
        )

        if not surrogate_status().is_stub:
            resolved_surrogate = predict_world_spec_dict(world_spec)
    diagnostic_fig = create_diagnostic_dashboard(
        sim_result,
        title=title,
        surrogate_pred=resolved_surrogate,
    )

    archive_metrics = metrics_dict_from_row(row_dict)
    if archive_metrics:
        st.caption(
            "Archive metrics (precomputed): "
            + ", ".join(
                f"{key}={value:.3f}" for key, value in sorted(archive_metrics.items())
            )
        )
    chart_key = "explorer_diagnostic"
    if bin_x is not None and bin_y is not None:
        chart_key = f"{chart_key}_{bin_x}_{bin_y}_{spec_hash[:16]}"
    else:
        chart_key = f"{chart_key}_{spec_hash[:16]}"
    st.plotly_chart(diagnostic_fig, width="stretch", key=chart_key)
    with st.expander("What do these panels mean?", expanded=False):
        for panel_key, blurb in DIAGNOSTIC_PANEL_HELP.items():
            st.markdown(f"**{panel_key.replace('_', ' ').title()}** — {blurb}")
    if sim_result.metrics is not None:
        st.markdown("##### Interpretation")
        for paragraph in format_diagnostic_interpretation(sim_result.metrics).split(
            "\n\n"
        ):
            block = paragraph.strip()
            if block:
                st.markdown(block)


def _row_int(row: pd.Series, key: str) -> int:
    return int(row.at[key])


def _row_float(row: pd.Series, key: str) -> float:
    return float(row.at[key])


def _optional_row_int(row: pd.Series, key: str) -> int | None:
    if key not in row.index:
        return None
    value: Any = row.at[key]
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return int(value)

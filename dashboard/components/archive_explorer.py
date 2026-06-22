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
SESSION_KEY_SELECTED_CELL = "explorer_selected_cell"
SESSION_KEY_SIMULATED_HASHES = "explorer_simulated_hashes"
SESSION_KEY_ARCHIVE_PATH = "explorer_archive_path"

__all__ = [
    "SESSION_KEY_ARCHIVE_PATH",
    "SESSION_KEY_SELECTED_BIN",
    "SESSION_KEY_SELECTED_CELL",
    "SESSION_KEY_SIMULATED_HASHES",
    "elite_row_for_bin",
    "elite_row_for_cell",
    "format_elite_bin_label",
    "format_elite_cell_label",
    "diagnostic_chart_key",
    "get_selected_elite_row",
    "list_bins_from_frame",
    "list_cells_from_frame",
    "list_niches_from_frame",
    "render_diagnostic_panel",
    "reset_explorer_session_for_archive",
    "run_cached_simulation_with_ui",
    "sync_selected_bin_selectbox",
    "sync_selected_cell_selectbox",
    "sync_selected_niche_selectbox",
]


def reset_explorer_session_for_archive(archive_path: str) -> None:
    """Clear bin/simulation session keys when the user switches archive JSONL."""
    previous = st.session_state.get(SESSION_KEY_ARCHIVE_PATH)
    if previous == archive_path:
        return
    st.session_state[SESSION_KEY_ARCHIVE_PATH] = archive_path
    st.session_state.pop(SESSION_KEY_SELECTED_BIN, None)
    st.session_state.pop(SESSION_KEY_SELECTED_CELL, None)
    st.session_state.pop(SESSION_KEY_SIMULATED_HASHES, None)


def list_cells_from_frame(frame: pd.DataFrame) -> list[int]:
    """Return ``cell_id`` values present in a collapsed CVT archive frame."""
    if frame.empty or "cell_id" not in frame.columns:
        return []
    return sorted({int(value) for value in frame["cell_id"].dropna().unique()})


def elite_row_for_cell(frame: pd.DataFrame, cell_id: int) -> pd.Series | None:
    """Lookup the collapsed elite row for a CVT ``cell_id``."""
    if frame.empty or "cell_id" not in frame.columns:
        return None
    match = frame[frame["cell_id"] == cell_id]
    if match.empty:
        return None
    return match.iloc[0]


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


def list_niches_from_frame(
    frame: pd.DataFrame, archive_type: str
) -> list[int] | list[tuple[int, int]]:
    """Return occupied niche keys for grid bins or CVT cell ids."""
    if archive_type == "cvt":
        return list_cells_from_frame(frame)
    return list_bins_from_frame(frame)


def get_selected_elite_row(
    frame: pd.DataFrame,
    *,
    archive_type: str = "grid",
) -> pd.Series | None:
    """Return the elite for the active niche session key, if set and present."""
    if archive_type == "cvt":
        selected_cell = st.session_state.get(SESSION_KEY_SELECTED_CELL)
        if isinstance(selected_cell, int):
            return elite_row_for_cell(frame, selected_cell)
        return None
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


def format_elite_cell_label(frame: pd.DataFrame, cell_id: int) -> str:
    """Short label for CVT cell selectors (cell id + BC niche center + fitness)."""
    row = elite_row_for_cell(frame, cell_id)
    if row is None:
        return f"cell {cell_id}"
    fitness = _row_float(row, "fitness") if "fitness" in row.index else float("nan")
    label = f"cell {cell_id}"
    if "centroid_s" in row.index and "centroid_d" in row.index:
        s = _row_float(row, "centroid_s")
        d = _row_float(row, "centroid_d")
        if not (pd.isna(s) or pd.isna(d)):
            label = f"cell {cell_id} (s={s:.2f}, d={d:.2f})"
    parts = [label, f"fitness={fitness:.4f}"]
    seed = _optional_row_int(row, "seed")
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


def sync_selected_cell_selectbox(
    frame: pd.DataFrame,
    *,
    label: str = "Selected cell",
) -> pd.Series | None:
    """CVT cell picker wired to ``SESSION_KEY_SELECTED_CELL``."""
    cells = list_cells_from_frame(frame)
    if not cells:
        return None

    current = st.session_state.get(SESSION_KEY_SELECTED_CELL)
    if not isinstance(current, int) or current not in cells:
        st.session_state[SESSION_KEY_SELECTED_CELL] = cells[0]

    chosen = st.selectbox(
        label,
        cells,
        format_func=lambda cell_id: format_elite_cell_label(frame, cell_id),
        key=SESSION_KEY_SELECTED_CELL,
    )
    cell_id = int(chosen) if isinstance(chosen, int) else cells[0]
    return elite_row_for_cell(frame, cell_id)


def sync_selected_niche_selectbox(
    frame: pd.DataFrame,
    archive_type: str,
    *,
    label: str = "Selected niche",
) -> pd.Series | None:
    """Grid bin or CVT cell picker depending on ``archive_type``."""
    if archive_type == "cvt":
        return sync_selected_cell_selectbox(frame, label=label)
    return sync_selected_bin_selectbox(frame, label=label)


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

    fitness = float(row_dict["fitness"]) if "fitness" in row_dict else float("nan")
    title_parts = ["Diagnostic"]
    niche_label = _diagnostic_niche_label(row_dict)
    if niche_label is not None:
        title_parts.append(niche_label)
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
    chart_key = diagnostic_chart_key(row_dict, spec_hash)
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


def _is_cvt_row(row_dict: Mapping[str, Any]) -> bool:
    return str(row_dict.get("archive_type", "grid")) == "cvt"


def _diagnostic_niche_label(row_dict: Mapping[str, Any]) -> str | None:
    """Human-readable niche label aligned with chart key identity."""
    if _is_cvt_row(row_dict) and "cell_id" in row_dict:
        cell_id = int(row_dict["cell_id"])
        if "centroid_s" in row_dict and "centroid_d" in row_dict:
            try:
                s = float(row_dict["centroid_s"])
                d = float(row_dict["centroid_d"])
                if not (pd.isna(s) or pd.isna(d)):
                    return f"cell {cell_id} (s={s:.2f}, d={d:.2f})"
            except (TypeError, ValueError):
                pass
        return f"cell {cell_id}"
    if "bin_x" in row_dict and "bin_y" in row_dict:
        return f"bin ({int(row_dict['bin_x'])}, {int(row_dict['bin_y'])})"
    if "cell_id" in row_dict:
        return f"cell {int(row_dict['cell_id'])}"
    return None


def diagnostic_chart_key(
    row_dict: Mapping[str, Any],
    spec_hash: str,
    *,
    prefix: str = "explorer_diagnostic",
) -> str:
    """Build a stable Streamlit chart key using the same niche identity as the title."""
    if _is_cvt_row(row_dict) and "cell_id" in row_dict:
        niche = f"cell_{int(row_dict['cell_id'])}"
    elif "bin_x" in row_dict and "bin_y" in row_dict:
        niche = f"bin_{int(row_dict['bin_x'])}_{int(row_dict['bin_y'])}"
    elif "cell_id" in row_dict:
        niche = f"cell_{int(row_dict['cell_id'])}"
    else:
        niche = "unknown"
    return f"{prefix}_{niche}_{spec_hash[:16]}"


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

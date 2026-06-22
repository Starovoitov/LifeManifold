"""LLM prompt dry-run preview with surrogate placeholders."""

from __future__ import annotations

import json
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

from dashboard.components.archive_loader import get_archive_bundle
from dashboard.components.llm_prompt_tester import (
    build_user_prompt_like_emitter,
    format_cell_label,
    list_format_placeholders,
    load_grid_archive,
    load_user_prompt_from_config,
    minimal_user_prompt_kwargs,
    occupied_cell_ids,
    parent_world_spec_dict,
    preview_rng,
    render_system_prompt_preview,
    render_user_prompt_preview,
    target_for_cell_id,
    user_prompt_format_kwargs,
    PREVIEW_RNG_SEED,
)
from dashboard.components.surrogate_widget import (
    predict_world_spec_dict,
    render_surrogate_status_banner,
)
from dashboard.utils.config import (
    DASHBOARD_ARCHIVE_SESSION_KEY,
    existing_archive_paths,
    load_config,
    repo_root,
)
from worldspace.illuminators.archive_protocol import ArchiveProtocol

_TEMPLATE_SESSION_KEY = "llm_user_prompt_template"


@st.cache_data(show_spinner=False)
def _cached_archive(
    path_str: str,
    mtime: float,
    resolution: int,
    archive_type: str,
) -> ArchiveProtocol:
    del mtime
    return load_grid_archive(
        Path(path_str),
        resolution=resolution,
        archive_type=archive_type,
    )


st.set_page_config(page_title="LLM Prompt Tester", layout="wide")
st.title("LLM Prompt Tester")
st.caption("Dry-run MAP-Elites LLM prompts with surrogate placeholders (no API call).")

cfg = load_config()

defaults = cfg.get("defaults")
grid_resolution = int(defaults["grid_resolution"]) if isinstance(defaults, dict) else 50
archive_type = "grid"
n_centroids: int | None = None

file_template = load_user_prompt_from_config(cfg)
if _TEMPLATE_SESSION_KEY not in st.session_state:
    st.session_state[_TEMPLATE_SESSION_KEY] = file_template

with st.sidebar:
    if st.button("Reset template to file"):
        st.session_state[_TEMPLATE_SESSION_KEY] = file_template
        st.rerun()

archives = existing_archive_paths(cfg)
loaded_archive: ArchiveProtocol | None = None
selected_path: Path | None = None
chosen_cell_id: int | None = None

if archives:

    def _archive_label(path: Path) -> str:
        return str(path.relative_to(repo_root()))

    selected_path = st.sidebar.selectbox(
        "Archive JSONL",
        archives,
        format_func=_archive_label,
        key=DASHBOARD_ARCHIVE_SESSION_KEY,
    )
    render_surrogate_status_banner(cfg, archive_path=selected_path)
    bundle = get_archive_bundle(selected_path)
    archive_type = bundle.archive_type
    n_centroids = bundle.n_cells if archive_type == "cvt" else None
    loaded_archive = _cached_archive(
        str(selected_path.resolve()),
        float(selected_path.stat().st_mtime),
        bundle.resolution,
        archive_type,
    )
    grid_resolution = bundle.resolution
    st.sidebar.metric("Archive type", bundle.archive_type)
    st.sidebar.metric("Niches", bundle.n_cells)
else:
    render_surrogate_status_banner(cfg)
    st.warning(
        "No archive JSONL found. Few-shot and current-elite blocks use placeholders."
    )

st.subheader("Prompt template")
placeholders = list_format_placeholders(file_template)
st.markdown(
    "Placeholders: " + ", ".join(f"`{{{name}}}`" for name in placeholders)
    if placeholders
    else "_none detected_"
)

user_template = st.text_area(
    "User prompt template",
    value=st.session_state[_TEMPLATE_SESSION_KEY],
    height=280,
)
st.session_state[_TEMPLATE_SESSION_KEY] = user_template
prompt_template = user_template or ""

surrogate_mean = 0.5
surrogate_uncertainty = 0.85
target_stability = 0.5
target_diversity = 0.5

col_s, col_d = st.columns(2)
with col_s:
    target_stability = st.number_input(
        "Target stability",
        min_value=0.0,
        max_value=1.0,
        value=float(target_stability),
        step=0.01,
    )
with col_d:
    target_diversity = st.number_input(
        "Target diversity",
        min_value=0.0,
        max_value=1.0,
        value=float(target_diversity),
        step=0.01,
    )

if loaded_archive is not None:
    archive = loaded_archive
    cells = occupied_cell_ids(archive)
    if not cells:
        st.warning("Archive has no occupied cells.")
    else:
        chosen_cell_id = st.selectbox(
            "Target cell",
            cells,
            format_func=lambda cell_id: format_cell_label(archive, cell_id),
        )
    elite = archive.get_cell(chosen_cell_id) if chosen_cell_id is not None else None
    parent_spec = parent_world_spec_dict(elite)
    if parent_spec is not None:
        prediction = predict_world_spec_dict(parent_spec)
        if prediction is not None:
            surrogate_mean = float(prediction["fitness"])
            surrogate_uncertainty = float(prediction["uncertainty"])
else:
    st.subheader("Manual parent world_spec")
    manual_json = st.text_area(
        "Parent world_spec JSON",
        value="{}",
        height=140,
    )
    if manual_json.strip():
        try:
            parsed = json.loads(manual_json)
        except json.JSONDecodeError:
            st.error("Invalid JSON for world_spec.")
        else:
            if isinstance(parsed, dict):
                prediction = predict_world_spec_dict(parsed)
                if prediction is not None:
                    surrogate_mean = float(prediction["fitness"])
                    surrogate_uncertainty = float(prediction["uncertainty"])

st.metric("Surrogate mean (preview)", f"{surrogate_mean:.3f}")
st.metric("Surrogate uncertainty", f"{surrogate_uncertainty:.3f}")

st.subheader("Rendered preview")
rendered_user = ""
format_error: str | None = None
reference: str | None = None

if loaded_archive is not None and chosen_cell_id is not None:
    target = target_for_cell_id(
        loaded_archive,
        chosen_cell_id,
        target_stability=float(target_stability),
        target_diversity=float(target_diversity),
    )
    kwargs = user_prompt_format_kwargs(
        loaded_archive,
        target,
        surrogate_mean,
        surrogate_uncertainty,
        rng=preview_rng(PREVIEW_RNG_SEED),
    )
    rendered_user, format_error = render_user_prompt_preview(prompt_template, kwargs)
    reference = build_user_prompt_like_emitter(
        loaded_archive,
        target,
        surrogate_mean,
        surrogate_uncertainty,
        rng=preview_rng(PREVIEW_RNG_SEED),
    )
    if (
        format_error is None
        and prompt_template == file_template
        and rendered_user != reference
    ):
        st.warning("Rendered prompt differs from stock build_user_prompt (unexpected).")
    elif format_error is None and prompt_template != file_template:
        with st.expander("Stock build_user_prompt reference"):
            st.code(reference, language="text")
else:
    kwargs = minimal_user_prompt_kwargs(
        surrogate_mean,
        surrogate_uncertainty,
        target_stability=float(target_stability),
        target_diversity=float(target_diversity),
    )
    rendered_user, format_error = render_user_prompt_preview(prompt_template, kwargs)

if format_error is not None:
    st.error(format_error)
st.text_area("Rendered user prompt", value=rendered_user, height=360, disabled=True)

system_text = render_system_prompt_preview(
    grid_resolution,
    cfg,
    archive_type=archive_type,
    n_centroids=n_centroids,
)
st.subheader("System prompt preview")
st.text_area("Rendered system prompt", value=system_text, height=220, disabled=True)

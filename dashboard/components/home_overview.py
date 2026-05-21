"""Home page: run cards, reproducibility block, and navigation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.components.archive_loader import get_archive_bundle
from dashboard.utils.config import repo_root
from dashboard.utils.run_discovery import RunInfo, summary_get

_MAX_RUNS_ON_HOME = 12
_MAX_ARCHIVE_STATS = 5

__all__ = [
    "archive_run_stats",
    "render_page_links",
    "render_reproducibility_block",
    "render_run_card",
    "render_runs_overview",
]


def render_runs_overview(runs: list[RunInfo]) -> None:
    """Render run cards and reproducibility for the newest runs."""
    if not runs:
        st.warning(
            "No MAP-Elites runs found. Run smoke or nightly MAP-Elites, "
            "or update scan paths in dashboard config."
        )
        return

    st.subheader("Recent MAP-Elites runs")
    for index, run in enumerate(runs[:_MAX_RUNS_ON_HOME]):
        with st.expander(_run_expander_title(run), expanded=index == 0):
            render_run_card(run, load_stats=index < _MAX_ARCHIVE_STATS)

    summary_for_repro = _first_summary(runs)
    if summary_for_repro is not None:
        st.subheader("Reproducibility")
        render_reproducibility_block(summary_for_repro)


def render_run_card(run: RunInfo, *, load_stats: bool = True) -> None:
    """Show summary metrics and optional archive-derived stats for one run."""
    rel_archive = run.archive_path.relative_to(repo_root())
    st.markdown(f"**Archive:** `{rel_archive}`")
    if run.summary_path is not None:
        rel_summary = run.summary_path.relative_to(repo_root())
        st.markdown(f"**Summary:** `{rel_summary}`")
    else:
        st.caption("No nightly/smoke summary JSON next to this archive.")

    summary = run.summary
    col1, col2, col3, col4 = st.columns(4)
    filled = summary_get(summary, "filled_cells", default=None)
    coverage = summary_get(summary, "coverage", default=None)
    evaluations = summary_get(summary, "evaluations", default=None)
    elapsed = summary_get(summary, "elapsed_seconds", default=None)

    col1.metric("Filled cells", _format_int(filled))
    col2.metric("Coverage", _format_percent(coverage))
    col3.metric("Evaluations", _format_int(evaluations))
    col4.metric("Elapsed (s)", _format_float(elapsed))

    if load_stats:
        stats = archive_run_stats(
            str(run.archive_path.resolve()),
            run.archive_mtime,
        )
        if stats.get("mean_fitness") is not None:
            st.metric("Mean fitness (archive)", f"{stats['mean_fitness']:.4f}")
        breakdown = stats.get("emitter_breakdown")
        if isinstance(breakdown, dict) and breakdown:
            st.markdown("**Emitter breakdown**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"emitter_type": key, "count": value}
                        for key, value in breakdown.items()
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )


def render_reproducibility_block(summary: dict[str, Any]) -> None:
    """Display scheduler, seed, grid, schema, and feature flags from summary JSON."""
    scheduler = summary_get(summary, "scheduler", default="—")
    seed = summary_get(summary, "seed", default="—")
    resolution = summary_get(summary, "grid_resolution", default="—")
    schema = summary_get(summary, "schema_version", default="—")
    llm = summary_get(summary, "llm_enabled", default="—")
    surrogate = summary_get(summary, "surrogate_enabled", default="—")
    jsonl_lines = summary_get(
        summary,
        "jsonl_raw_lines",
        "jsonl_lines",
        default="—",
    )

    st.markdown(
        f"- **Scheduler:** `{scheduler}`\n"
        f"- **Seed:** `{seed}`\n"
        f"- **Grid resolution:** `{resolution}`\n"
        f"- **Schema version:** `{schema}`\n"
        f"- **LLM enabled:** `{llm}`\n"
        f"- **Surrogate enabled:** `{surrogate}`\n"
        f"- **JSONL lines:** `{jsonl_lines}`"
    )


def render_page_links() -> None:
    """Quick navigation to dashboard pages."""
    st.subheader("Pages")
    st.page_link("pages/1_Archive_Explorer.py", label="Archive Explorer")
    st.page_link("pages/2_Surrogate_Analysis.py", label="Surrogate Analysis")
    st.page_link("pages/3_Metrics_Dashboard.py", label="Metrics Dashboard")
    st.page_link("pages/4_LLM_Prompt_Tester.py", label="LLM Prompt Tester")
    st.page_link("pages/5_Training_Buffer.py", label="Training Buffer")


@st.cache_data(show_spinner=False)
def archive_run_stats(archive_path_str: str, mtime: float) -> dict[str, Any]:
    """Mean fitness and emitter counts from a collapsed archive (cached)."""
    del mtime
    bundle = get_archive_bundle(Path(archive_path_str))
    frame = bundle.collapsed
    stats: dict[str, Any] = {}
    if frame.empty or "fitness" not in frame.columns:
        return stats
    stats["mean_fitness"] = float(np.mean(frame["fitness"].to_numpy(dtype=np.float64)))
    if "emitter_type" in frame.columns:
        counts = frame["emitter_type"].value_counts(dropna=True)
        stats["emitter_breakdown"] = {
            str(key): int(value) for key, value in counts.items()
        }
    return stats


def _run_expander_title(run: RunInfo) -> str:
    rel = run.run_dir.relative_to(repo_root())
    filled = summary_get(run.summary, "filled_cells", default="?")
    coverage = summary_get(run.summary, "coverage", default=None)
    cov_text = f"{float(coverage) * 100:.1f}%" if coverage is not None else "n/a"
    return f"{rel} — filled={filled}, coverage={cov_text}"


def _first_summary(runs: list[RunInfo]) -> dict[str, Any] | None:
    for run in runs:
        if run.summary is not None:
            return run.summary
    return None


def _format_int(value: Any) -> str:
    if value is None:
        return "—"
    return str(int(value))


def _format_float(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.1f}"


def _format_percent(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.2f}%"

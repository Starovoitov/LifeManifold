"""Metric column helpers for dashboard tables (no fitness formula duplication)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dashboard.utils.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from worldspace.metrics import METRIC_KEYS

# Short hover / glossary text (aligned with docs/FORMULAS.md §4, WORLD_SPEC_AUDIT §7).
METRIC_HELP: dict[str, str] = {
    "entropy": (
        "Binary Shannon H on the time-mean live density (not spatial pattern entropy). "
        "Mid occupancy → higher; all-dead or all-live → low."
    ),
    "stability": (
        "1 − σ(ρ)/(μ(ρ)+ε) over per-step mean density. "
        "Steady density trace → high; wild swings → low. MAP-Elites behavior axis."
    ),
    "average_lifespan": (
        "Mean age at death when a live cell dies (1→0). "
        "Long-lived cells → high; bar display divides raw value by 10."
    ),
    "density_mean": (
        "Time-mean fraction of live cells. Feeds entropy and extinction-related terms."
    ),
    "oscillation_score": (
        "Peak normalized autocorrelation of recent density (512-step window). "
        "Rhythmic activity → high."
    ),
    "diversity": (
        "Fraction of unique random 3×3 life patches on the final grid (128 samples). "
        "MAP-Elites behavior axis."
    ),
    "mo_eoc_indicator": (
        "Multi-objective + edge-of-chaos scalar for GA/LLM search (not MAP-Elites fitness). "
        "Bar display divides raw value by 3."
    ),
    "topology_interface_index": (
        "Mean Moore share of neighbors where life differs. "
        "Fragmented boundaries → high (proxy, not Betti numbers)."
    ),
    "topology_window_heterogeneity": (
        "Share of toroidal 2×2 windows with mixed corners. "
        "Local mesoscale mixing → high."
    ),
    "compressibility_score": (
        "1 − zlib(life‖food)/raw length. Ordered configurations → high; noisy → low."
    ),
    "ecology_state_entropy_norm": (
        "Shannon entropy of joint (life, food) per-cell classes, normalized by log₂(k)."
    ),
    "ecology_resource_adjacency": (
        "On live cells only: mean Moore fraction of food neighbors. "
        "Consumers near resources → high."
    ),
    "fitness": (
        "MAP-Elites archive fitness (weighted stability, diversity, oscillation, topology)."
    ),
}

__all__ = [
    "METRIC_HELP",
    "METRIC_KEYS",
    "add_metrics_columns",
    "correlation_matrix",
    "metric_help_text",
    "metrics_dict_from_row",
    "metrics_series_from_dataframe",
]


def metric_help_text(key: str) -> str:
    """Return glossary text for a metric key (falls back to the key name)."""
    return METRIC_HELP.get(key, key.replace("_", " "))


def metrics_dict_from_row(row: dict[str, Any]) -> dict[str, float]:
    """Extract the standard metric vector from a flat or nested archive row."""
    nested = row.get("metrics")
    if isinstance(nested, dict):
        source: dict[str, Any] = nested
    else:
        source = row
    return {key: float(source[key]) for key in METRIC_KEYS if key in source}


def add_metrics_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``METRIC_KEYS`` appear as top-level float columns when present."""
    out = frame.copy()
    for key in METRIC_KEYS:
        if key in out.columns:
            continue
        if "metrics" in out.columns:
            out[key] = out["metrics"].map(
                lambda value: (
                    float(value[key])
                    if isinstance(value, dict) and key in value
                    else np.nan
                )
            )
    return out


def metrics_series_from_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with only metric columns that exist."""
    enriched = add_metrics_columns(frame)
    cols = [key for key in METRIC_KEYS if key in enriched.columns]
    if not cols:
        return pd.DataFrame()
    return enriched[cols].astype(float)


def correlation_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation over available ``METRIC_KEYS`` columns."""
    metrics = metrics_series_from_dataframe(frame)
    if metrics.empty or metrics.shape[1] < 2:
        return pd.DataFrame()
    return metrics.corr()

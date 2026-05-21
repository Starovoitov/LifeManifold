"""Metric column helpers for dashboard tables (no fitness formula duplication)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dashboard.utils.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from worldspace.metrics import METRIC_KEYS

__all__ = [
    "METRIC_KEYS",
    "add_metrics_columns",
    "correlation_matrix",
    "metrics_dict_from_row",
    "metrics_series_from_dataframe",
]


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

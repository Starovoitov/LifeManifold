"""Pure helpers for surrogate evaluation joins and metrics (no Streamlit)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "build_prediction_frame",
    "calibration_table",
    "regression_metrics",
    "sample_collapsed_rows",
]


def sample_collapsed_rows(
    frame: pd.DataFrame,
    *,
    max_rows: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Return up to ``max_rows`` elites with a ``world_spec`` column."""
    if frame.empty or "world_spec" not in frame.columns:
        return frame.iloc[0:0].copy()
    valid_mask = frame["world_spec"].map(lambda value: isinstance(value, dict))
    valid = frame.loc[valid_mask].copy()
    if valid.empty:
        return pd.DataFrame(columns=frame.columns)
    if len(valid) <= max_rows:
        return valid.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(valid), size=max_rows, replace=False)
    return valid.iloc[sorted(indices)].reset_index(drop=True)


def build_prediction_frame(
    frame: pd.DataFrame,
    predict_fn: Callable[[dict[str, Any]], dict[str, float] | None],
) -> pd.DataFrame:
    """Attach ``pred_fitness`` and ``pred_uncertainty`` for each elite row."""
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        world_spec = row.get("world_spec")
        if not isinstance(world_spec, dict):
            continue
        prediction = predict_fn(world_spec)
        if prediction is None:
            continue
        fitness_value = row.at["fitness"] if "fitness" in row.index else np.nan
        rows.append(
            {
                "fitness": float(fitness_value),
                "pred_fitness": float(prediction["fitness"]),
                "pred_uncertainty": float(prediction["uncertainty"]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["fitness", "pred_fitness", "pred_uncertainty"])
    return pd.DataFrame(rows)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Return MAE and R² for aligned prediction vectors."""
    true_arr = np.asarray(y_true, dtype=np.float64)
    pred_arr = np.asarray(y_pred, dtype=np.float64)
    if true_arr.size == 0:
        return float("nan"), float("nan")
    errors = true_arr - pred_arr
    mae = float(np.mean(np.abs(errors)))
    if true_arr.size < 2:
        return mae, float("nan")
    ss_res = float(np.sum(errors**2))
    true_mean = float(np.mean(true_arr))
    ss_tot = float(np.sum((true_arr - true_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return mae, r2


def calibration_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    *,
    n_bins: int = 8,
) -> pd.DataFrame:
    """Bin by uncertainty quantiles and compute mean absolute error per bin."""
    true_arr = np.asarray(y_true, dtype=np.float64)
    pred_arr = np.asarray(y_pred, dtype=np.float64)
    unc_arr = np.asarray(uncertainty, dtype=np.float64)
    if true_arr.size < n_bins:
        return pd.DataFrame(
            columns=["bin", "uncertainty_lo", "uncertainty_hi", "mae", "count"]
        )

    order = np.argsort(unc_arr)
    sorted_true = true_arr[order]
    sorted_pred = pred_arr[order]
    sorted_unc = unc_arr[order]
    chunks = np.array_split(np.arange(sorted_true.size), n_bins)

    records: list[dict[str, Any]] = []
    for index, indices in enumerate(chunks):
        if indices.size == 0:
            continue
        chunk_true = sorted_true[indices]
        chunk_pred = sorted_pred[indices]
        chunk_unc = sorted_unc[indices]
        records.append(
            {
                "bin": index,
                "uncertainty_lo": float(np.min(chunk_unc)),
                "uncertainty_hi": float(np.max(chunk_unc)),
                "mae": float(np.mean(np.abs(chunk_true - chunk_pred))),
                "count": int(indices.size),
            }
        )
    return pd.DataFrame(records)

"""Distribution and hold-out diagnostics for surrogate training buffers."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from worldspace.surrogate.evaluation import (
    QUALITY_MAE_FITNESS_MAX,
    QUALITY_MAE_STABILITY_MAX,
    QUALITY_R2_FITNESS_MIN,
    evaluate_holdout,
)
from worldspace.surrogate.feature_extractor import FEATURE_NAMES
from worldspace.surrogate.model import TARGET_KEYS, SurrogateModel
from worldspace.surrogate.training import holdout_split, load_buffer

__all__ = [
    "STABILITY_HISTOGRAM_BINS",
    "STABILITY_MAE_BANDS",
    "analyze_buffer_path",
    "format_analysis_report",
    "scan_buffer_metadata",
    "summarize_target_array",
]

STABILITY_HISTOGRAM_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.01),
)

STABILITY_MAE_BANDS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.4, "low"),
    (0.4, 0.7, "mid"),
    (0.7, 0.9, "high"),
    (0.9, 1.01, "very_high"),
)


@dataclass(frozen=True)
class BufferMetadataScan:
    emitter_types: dict[str, int]
    metadata_sources: dict[str, int]


def scan_buffer_metadata(path: Path) -> BufferMetadataScan:
    """Count emitter types and metadata.source values without loading matrices."""
    emitters: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            emitters[str(row.get("emitter_type") or "unknown")] += 1
            metadata = row.get("metadata") or {}
            sources[str(metadata.get("source") or "unknown")] += 1
    return BufferMetadataScan(
        emitter_types=dict(emitters),
        metadata_sources=dict(sources),
    )


def summarize_target_array(values: np.ndarray) -> dict[str, float | int]:
    """Return basic distribution stats for one target vector."""
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "count": int(arr.shape[0]),
    }


def _histogram(values: np.ndarray, bins: tuple[tuple[float, float], ...]) -> list[dict[str, float | int]]:
    arr = np.asarray(values, dtype=float)
    n = int(arr.shape[0])
    rows: list[dict[str, float | int]] = []
    for lo, hi in bins:
        count = int(np.sum((arr >= lo) & (arr < hi)))
        rows.append(
            {
                "lo": lo,
                "hi": hi,
                "count": count,
                "fraction": float(count / n) if n else 0.0,
            }
        )
    return rows


def _top_feature_correlations(
    feature_matrix: np.ndarray,
    target_values: np.ndarray,
    *,
    top_n: int = 8,
) -> list[dict[str, float | str]]:
    arr = np.asarray(target_values, dtype=float)
    rows: list[tuple[float, float, str]] = []
    for index, name in enumerate(FEATURE_NAMES):
        corr = float(np.corrcoef(feature_matrix[:, index], arr)[0, 1])
        if np.isfinite(corr):
            rows.append((abs(corr), corr, name))
    rows.sort(reverse=True)
    return [
        {"feature": name, "corr": corr}
        for _, corr, name in rows[:top_n]
    ]


def _emitter_breakdown(path: Path) -> list[dict[str, float | int | str]]:
    by_emitter: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            emitter = str(row.get("emitter_type") or "unknown")
            stability = float(row["targets"]["stability"])
            by_emitter.setdefault(emitter, []).append(stability)
    rows: list[dict[str, float | int | str]] = []
    for emitter, values in sorted(
        by_emitter.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    ):
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "emitter_type": emitter,
                "count": int(arr.shape[0]),
                "stability_mean": float(arr.mean()),
                "stability_std": float(arr.std()),
                "stability_low_fraction": float(np.mean(arr < 0.4)),
                "stability_high_fraction": float(np.mean(arr >= 0.8)),
            }
        )
    return rows


def _stability_baselines(
    y_train: dict[str, np.ndarray],
    y_hold: dict[str, np.ndarray],
) -> dict[str, float]:
    hold = np.asarray(y_hold["stability"], dtype=float)
    train = np.asarray(y_train["stability"], dtype=float)
    train_mean = float(train.mean())
    train_median = float(np.median(train))
    return {
        "train_mean_mae": float(np.mean(np.abs(hold - train_mean))),
        "train_median_mae": float(np.mean(np.abs(hold - train_median))),
        "constant_0_5_mae": float(np.mean(np.abs(hold - 0.5))),
    }


def _stability_mae_bands(
    model: SurrogateModel,
    x_hold: np.ndarray,
    y_hold: dict[str, np.ndarray],
) -> list[dict[str, float | int | str]]:
    hold = np.asarray(y_hold["stability"], dtype=float)
    preds = np.asarray(
        [model.predict_components(x_hold[index])["stability"] for index in range(len(x_hold))],
        dtype=float,
    )
    rows: list[dict[str, float | int | str]] = []
    for lo, hi, label in STABILITY_MAE_BANDS:
        mask = (hold >= lo) & (hold < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        mae = float(np.mean(np.abs(preds[mask] - hold[mask])))
        rows.append(
            {
                "label": label,
                "lo": lo,
                "hi": hi,
                "count": count,
                "mae": mae,
                "mean_true": float(hold[mask].mean()),
            }
        )
    return rows


def _per_target_holdout(
    model: SurrogateModel,
    x_hold: np.ndarray,
    y_hold: dict[str, np.ndarray],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for key in TARGET_KEYS:
        true = np.asarray(y_hold[key], dtype=float)
        preds = np.asarray(
            [model.predict_components(x_hold[index])[key] for index in range(len(x_hold))],
            dtype=float,
        )
        ss_res = float(np.sum((preds - true) ** 2))
        ss_tot = float(np.sum((true - true.mean()) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
        rows.append(
            {
                "target": key,
                "mae": float(np.mean(np.abs(preds - true))),
                "r2": r2,
            }
        )
    return rows


def analyze_buffer_path(
    path: Path | str,
    *,
    fit_model: bool = False,
    random_state: int = 42,
    test_fraction: float = 0.2,
    ensemble_size: int = 1,
) -> dict[str, Any]:
    """Analyze one schema 2.0 buffer JSONL and return a JSON-serializable report."""
    buffer_path = Path(path)
    feature_matrix, targets = load_buffer(buffer_path)
    metadata = scan_buffer_metadata(buffer_path)
    sample_count = int(feature_matrix.shape[0])

    target_stats = {
        key: summarize_target_array(targets[key]) for key in TARGET_KEYS
    }
    stability_histogram = _histogram(targets["stability"], STABILITY_HISTOGRAM_BINS)

    x_train, y_train, x_hold, y_hold = holdout_split(
        feature_matrix,
        targets,
        test_fraction=test_fraction,
        random_state=random_state,
    )
    holdout_split_stats = {
        "train_count": int(x_train.shape[0]),
        "holdout_count": int(x_hold.shape[0]),
        "random_state": random_state,
        "test_fraction": test_fraction,
        "train_mean": {
            key: float(y_train[key].mean()) for key in TARGET_KEYS
        },
        "holdout_mean": {
            key: float(y_hold[key].mean()) for key in TARGET_KEYS
        },
        "train_std": {
            key: float(y_train[key].std()) for key in TARGET_KEYS
        },
        "holdout_std": {
            key: float(y_hold[key].std()) for key in TARGET_KEYS
        },
    }

    stability_baselines = _stability_baselines(y_train, y_hold)
    stability_diversity_corr = float(
        np.corrcoef(targets["stability"], targets["diversity"])[0, 1]
    )
    feature_correlations = _top_feature_correlations(
        feature_matrix,
        targets["stability"],
    )

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "buffer_path": str(buffer_path.resolve()),
        "sample_count": sample_count,
        "targets": target_stats,
        "stability_histogram": stability_histogram,
        "stability_diversity_corr": stability_diversity_corr,
        "holdout_split": holdout_split_stats,
        "stability_baselines": stability_baselines,
        "feature_correlations_stability": feature_correlations,
        "emitter_breakdown": _emitter_breakdown(buffer_path),
        "metadata_sources": metadata.metadata_sources,
        "emitter_types": metadata.emitter_types,
        "quality_gate": {
            "r2_fitness_min": QUALITY_R2_FITNESS_MIN,
            "mae_fitness_max": QUALITY_MAE_FITNESS_MAX,
            "mae_stability_max": QUALITY_MAE_STABILITY_MAX,
            "naive_train_mean_mae": stability_baselines["train_mean_mae"],
            "mae_stability_gap_to_gate": float(
                stability_baselines["train_mean_mae"] - QUALITY_MAE_STABILITY_MAX
            ),
        },
        "model_fit": fit_model,
    }

    if fit_model:
        model = SurrogateModel(
            model_type="lightgbm",
            random_state=random_state,
            ensemble_size=ensemble_size,
        )
        model.fit(x_train, y_train)
        holdout_metrics = evaluate_holdout(model, x_hold, y_hold)
        report["model_holdout"] = holdout_metrics
        report["per_target_holdout"] = _per_target_holdout(model, x_hold, y_hold)
        report["stability_mae_bands"] = _stability_mae_bands(model, x_hold, y_hold)
        report["quality_gate"]["model_mae_stability"] = holdout_metrics["mae_stability"]
        report["quality_gate"]["model_mae_stability_gap_to_gate"] = float(
            holdout_metrics["mae_stability"] - QUALITY_MAE_STABILITY_MAX
        )

    return report


def format_analysis_report(report: dict[str, Any]) -> str:
    """Render a human-readable multi-line summary."""
    lines = [
        f"Buffer: {report['buffer_path']}",
        f"Samples: {report['sample_count']}",
        f"Sources: {report.get('metadata_sources', {})}",
        "",
        "Target stability:",
    ]
    stability = report["targets"]["stability"]
    lines.append(
        "  "
        f"mean={stability['mean']:.4f} std={stability['std']:.4f} "
        f"p50={stability['p50']:.4f} p10={stability['p10']:.4f} p90={stability['p90']:.4f}"
    )
    lines.append("  histogram:")
    for band in report["stability_histogram"]:
        lines.append(
            "    "
            f"[{band['lo']:.1f}, {band['hi']:.1f}): "
            f"{band['count']} ({band['fraction'] * 100:.1f}%)"
        )

    split = report["holdout_split"]
    lines.extend(
        [
            "",
            "Hold-out split:",
            f"  train={split['train_count']} hold={split['holdout_count']} "
            f"(seed={split['random_state']})",
            (
                "  stability train mean="
                f"{split['train_mean']['stability']:.4f} hold="
                f"{split['holdout_mean']['stability']:.4f}"
            ),
            "",
            "Stability MAE baselines (hold-out):",
            f"  train-mean: {report['stability_baselines']['train_mean_mae']:.4f}",
            f"  train-median: {report['stability_baselines']['train_median_mae']:.4f}",
            f"  constant 0.5: {report['stability_baselines']['constant_0_5_mae']:.4f}",
            "",
            "Top feature |corr(stability)|:",
        ]
    )
    for row in report["feature_correlations_stability"]:
        lines.append(f"  {row['feature']:22s} {row['corr']:+.4f}")

    lines.append("")
    lines.append("Emitter breakdown:")
    for row in report["emitter_breakdown"]:
        lines.append(
            "  "
            f"{row['emitter_type']:8s} n={row['count']:4d} "
            f"stab μ={row['stability_mean']:.3f} "
            f"low<0.4={row['stability_low_fraction'] * 100:.1f}% "
            f"high≥0.8={row['stability_high_fraction'] * 100:.1f}%"
        )

    gate = report["quality_gate"]
    lines.extend(
        [
            "",
            "Quality gate reference:",
            f"  MAE(stability) < {gate['mae_stability_max']:.3f}",
            f"  naive train-mean MAE = {gate['naive_train_mean_mae']:.4f} "
            f"(gap to gate: {gate['mae_stability_gap_to_gate']:+.4f})",
        ]
    )

    if report.get("model_holdout"):
        metrics = report["model_holdout"]
        lines.extend(
            [
                "",
                "Model hold-out:",
                f"  r2_fitness={metrics['r2_fitness']:.4f}",
                f"  mae_fitness={metrics['mae_fitness']:.4f}",
                f"  mae_stability={metrics['mae_stability']:.4f} "
                f"(gap to gate: {gate.get('model_mae_stability_gap_to_gate', float('nan')):+.4f})",
                "",
                "Stability MAE by band:",
            ]
        )
        for band in report.get("stability_mae_bands", []):
            lines.append(
                "  "
                f"{band['label']:10s} n={band['count']:3d} "
                f"MAE={band['mae']:.4f} mean_true={band['mean_true']:.3f}"
            )

    return "\n".join(lines)

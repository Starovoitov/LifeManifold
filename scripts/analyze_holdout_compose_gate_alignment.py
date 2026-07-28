#!/usr/bin/env python3
"""Full hold-out compose-gate alignment: gate 0.5 vs 0.95 on nightly checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.illuminators.evaluation import compute_fitness
from worldspace.metrics import WorldMetrics
from worldspace.surrogate.checkpoint_io import load_surrogate_checkpoint
from worldspace.surrogate.evaluation import (
    HINTS_R2_FITNESS_MIN,
    QUALITY_MAE_FITNESS_MAX,
    QUALITY_MAE_STABILITY_MAX,
    QUALITY_R2_FITNESS_MIN,
)
from worldspace.surrogate.model import TARGET_KEYS
from worldspace.surrogate.training import holdout_split, load_buffer
from worldspace.surrogate.types import SurrogatePrediction
from worldspace.surrogate.utils import compute_fitness_from_prediction

BUFFER = ROOT / "artifacts/surrogate/buffer_nightly.jsonl"
CKPT = ROOT / "artifacts/surrogate/checkpoints/nightly_v3_mc_d005.pkl"
SUMMARY = ROOT / "artifacts/surrogate/checkpoints/nightly_v3_mc_d005.summary.json"
OUT_JSON = ROOT / "artifacts/surrogate/holdout_compose_gate_alignment.json"
OUT_MD = ROOT / "artifacts/experiments/q1-full/HOLDOUT_COMPOSE_GATE_ALIGNMENT.md"

BOOTSTRAP_B = 10_000
RNG = 42
MIN_FIT = 0.45
GATES = (0.5, 0.95)


def _target_distribution(y: np.ndarray) -> dict[str, float]:
    """Describe composed-fitness label distribution under one gate."""
    y = np.asarray(y, dtype=float)
    nonzero = y[y > 0]
    return {
        "mean": float(np.mean(y)),
        "std": float(np.std(y, ddof=0)),
        "var": float(np.var(y, ddof=0)),
        "median": float(np.median(y)),
        "p25": float(np.percentile(y, 25)),
        "p75": float(np.percentile(y, 75)),
        "p95": float(np.percentile(y, 95)),
        "max": float(np.max(y)),
        "frac_zero": float(np.mean(y <= 0)),
        "mean_nonzero": float(np.mean(nonzero)) if nonzero.size else float("nan"),
        "std_nonzero": (
            float(np.std(nonzero, ddof=0)) if nonzero.size else float("nan")
        ),
        "n_nonzero": int(nonzero.size),
    }


def _nmae(y: np.ndarray, pred: np.ndarray) -> float:
    """MAE normalized by population std of the target (comparable across gates)."""
    std = float(np.std(y, ddof=0))
    if std <= 0.0:
        return float("nan")
    return float(mean_absolute_error(y, pred) / std)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y, pred))
    std = float(np.std(y, ddof=0))
    return {
        "r2_fitness": float(r2_score(y, pred)),
        "mae_fitness": mae,
        "nmae_fitness": float(mae / std) if std > 0.0 else float("nan"),
        "target_std": std,
        "frac_true_zero": float(np.mean(y <= 0)),
        "frac_pred_zero": float(np.mean(pred <= 0)),
    }


def _bootstrap_ci(
    y: np.ndarray,
    pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    level: float = 0.95,
    b: int = BOOTSTRAP_B,
    random_state: int = RNG,
) -> tuple[float, float]:
    n = int(y.shape[0])
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(random_state)
    alpha = 1.0 - level
    samples = np.empty(b, dtype=float)
    for draw in range(b):
        idx = rng.integers(0, n, size=n)
        samples[draw] = metric_fn(y[idx], pred[idx])
    samples.sort()
    lo = float(samples[int((alpha / 2) * b)])
    hi = float(samples[int((1 - alpha / 2) * b) - 1])
    return (lo, hi)


def _true_fitness(y_hold: dict[str, np.ndarray], index: int, gate: float) -> float:
    comps = {key: float(y_hold[key][index]) for key in TARGET_KEYS}
    metrics = WorldMetrics(
        entropy=0.0,
        stability=comps["stability"],
        average_lifespan=0.0,
        density_mean=comps["final_density"],
        oscillation_score=comps["oscillation_score"],
        diversity=comps["diversity"],
        mo_eoc_indicator=0.0,
        topology_interface_index=comps["topology_interface_index"],
        topology_window_heterogeneity=comps["topology_window_heterogeneity"],
        compressibility_score=0.0,
        ecology_state_entropy_norm=0.0,
        ecology_resource_adjacency=0.0,
    )
    return compute_fitness(
        metrics,
        {"stability": comps["stability"], "diversity": comps["diversity"]},
        early_extinct=comps["early_extinction_prob"] >= gate,
        final_density=comps["final_density"],
    )


def _pred_fitness(comps: dict[str, float], gate: float) -> float:
    pred = SurrogatePrediction(
        components=comps,
        measures={
            "stability": float(comps["stability"]),
            "diversity": float(comps["diversity"]),
        },
        fitness=0.0,
        uncertainty=0.0,
    )
    return compute_fitness_from_prediction(pred, extinction_gate_threshold=gate)


def _quality_flags(
    metrics: dict[str, float], *, mae_stability: float | None
) -> dict[str, bool]:
    return {
        "hints_ok": metrics["r2_fitness"] >= HINTS_R2_FITNESS_MIN
        and metrics["mae_fitness"] < QUALITY_MAE_FITNESS_MAX,
        "quality_passed": metrics["r2_fitness"] > QUALITY_R2_FITNESS_MIN
        and metrics["mae_fitness"] < QUALITY_MAE_FITNESS_MAX
        and (mae_stability is None or mae_stability < QUALITY_MAE_STABILITY_MAX),
    }


def analyze_holdout_compose_gate_alignment(
    *,
    buffer_path: Path = BUFFER,
    checkpoint_path: Path = CKPT,
    summary_path: Path = SUMMARY,
    random_state: int = RNG,
    bootstrap_b: int = BOOTSTRAP_B,
) -> dict[str, Any]:
    features, targets = load_buffer(buffer_path)
    _x_tr, _y_tr, x_hold, y_hold = holdout_split(
        features, targets, random_state=random_state
    )
    model = load_surrogate_checkpoint(checkpoint_path)
    n_hold = int(x_hold.shape[0])

    pred_comps = model.predict_components_batch(x_hold)
    true_by_gate = {
        gate: np.asarray([_true_fitness(y_hold, i, gate) for i in range(n_hold)])
        for gate in GATES
    }
    pred_by_gate = {
        gate: np.asarray([_pred_fitness(comps, gate) for comps in pred_comps])
        for gate in GATES
    }
    true_stability = np.asarray(y_hold["stability"], dtype=float)
    pred_stability = np.asarray(
        [float(c["stability"]) for c in pred_comps], dtype=float
    )
    mae_stability = float(mean_absolute_error(true_stability, pred_stability))

    aligned: dict[str, Any] = {}
    for gate in GATES:
        key = f"gate_{gate:g}".replace(".", "p")
        y = true_by_gate[gate]
        p = pred_by_gate[gate]
        point = _metrics(y, p)
        target_dist = _target_distribution(y)
        r2_ci = _bootstrap_ci(
            y,
            p,
            lambda yy, pp: float(r2_score(yy, pp)),
            b=bootstrap_b,
            random_state=random_state,
        )
        mae_ci = _bootstrap_ci(
            y,
            p,
            lambda yy, pp: float(mean_absolute_error(yy, pp)),
            b=bootstrap_b,
            random_state=random_state + 1,
        )
        nmae_ci = _bootstrap_ci(
            y,
            p,
            _nmae,
            b=bootstrap_b,
            random_state=random_state + 2,
        )
        aligned[key] = {
            "extinction_gate_threshold": gate,
            "point_metrics": point,
            "target_distribution": target_dist,
            "mae_stability": mae_stability,
            "bootstrap": {
                "B": bootstrap_b,
                "random_state": random_state,
                "r2_fitness_ci_95": list(r2_ci),
                "mae_fitness_ci_95": list(mae_ci),
                "nmae_fitness_ci_95": list(nmae_ci),
            },
            "quality_flags": _quality_flags(point, mae_stability=mae_stability),
        }

    pext = np.asarray([float(c["early_extinction_prob"]) for c in pred_comps])
    y05, y95 = true_by_gate[0.5], true_by_gate[0.95]
    p05, p95 = pred_by_gate[0.5], pred_by_gate[0.95]
    in_band = (pext >= 0.5) & (pext < 0.95)

    baseline: dict[str, Any] | None = None
    legacy_r2: float | None = None
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        baseline = {
            "path": str(summary_path),
            "holdout_count": summary.get("holdout_count"),
            "holdout_metrics": summary.get("holdout_metrics"),
            "holdout_metrics_gate_0p5_legacy": summary.get(
                "holdout_metrics_gate_0p5_legacy"
            ),
            "holdout_extinction_gate_threshold": summary.get(
                "holdout_extinction_gate_threshold"
            ),
            "hints_ok": summary.get("hints_ok"),
            "quality_passed": summary.get("quality_passed"),
        }
        legacy_block = summary.get("holdout_metrics_gate_0p5_legacy")
        if isinstance(legacy_block, dict) and "r2_fitness" in legacy_block:
            legacy_r2 = float(legacy_block["r2_fitness"])
        elif float(
            summary.get("holdout_extinction_gate_threshold", 0.5)
        ) < 0.9 and isinstance(summary.get("holdout_metrics"), dict):
            legacy_r2 = float(summary["holdout_metrics"]["r2_fitness"])

    gate05 = aligned["gate_0p5"]["point_metrics"]
    gate95 = aligned["gate_0p95"]["point_metrics"]
    repro_ok = legacy_r2 is not None and abs(gate05["r2_fitness"] - legacy_r2) < 1e-3

    dist05 = aligned["gate_0p5"]["target_distribution"]
    dist95 = aligned["gate_0p95"]["target_distribution"]
    std_ratio = dist95["std"] / dist05["std"] if dist05["std"] > 0 else float("nan")
    mae_ratio = (
        gate95["mae_fitness"] / gate05["mae_fitness"]
        if gate05["mae_fitness"] > 0
        else float("nan")
    )
    r2_mae_divergence = {
        "target_std_gate_0p5": dist05["std"],
        "target_std_gate_0p95": dist95["std"],
        "target_std_ratio_0p95_over_0p5": float(std_ratio),
        "target_var_ratio_0p95_over_0p5": float(
            dist95["var"] / dist05["var"] if dist05["var"] > 0 else float("nan")
        ),
        "frac_zero_gate_0p5": dist05["frac_zero"],
        "frac_zero_gate_0p95": dist95["frac_zero"],
        "mae_ratio_0p95_over_0p5": float(mae_ratio),
        "nmae_gate_0p5": gate05["nmae_fitness"],
        "nmae_gate_0p95": gate95["nmae_fitness"],
        "interpretation": (
            "Gate 0.95 zeros far fewer rows (proxy p_ext≥0.95 vs ≥0.5), so composed "
            "fitness has much larger variance. R² rises because SS_tot grows faster "
            "than residual error; absolute MAE also rises because nonzero fitness "
            "errors are no longer collapsed to the near-zero mass. NMAE=MAE/std(y) "
            "is the cross-gate comparable absolute-error metric. Do not read "
            "R² 0.76→0.94 as a model improvement — it is a target-definition change."
        ),
    }

    validity = {
        "headline_gate_0p5_reproduces_checkpoint_summary": repro_ok,
        "aligned_gate_0p95_passes_hints_ok": aligned["gate_0p95"]["quality_flags"][
            "hints_ok"
        ],
        "aligned_gate_0p95_passes_quality_passed": aligned["gate_0p95"][
            "quality_flags"
        ]["quality_passed"],
        "protocol_fix_align_holdout_to_runtime_0p95_supports_validity_headline": bool(
            aligned["gate_0p95"]["quality_flags"]["quality_passed"]
        ),
        "r2_up_mae_up_is_target_rescaling_not_model_gain": True,
        "note": (
            "Aligning hold-out to runtime gate 0.95 makes labels match live compose. "
            "R²@0.95 is higher and MAE@0.95 is larger because the target distribution "
            "changes (see r2_mae_divergence); report NMAE for cross-gate comparison. "
            "This does not unlock confirmatory H3 by itself."
        ),
    }

    return {
        "n_holdout": n_hold,
        "buffer": str(buffer_path),
        "checkpoint": str(checkpoint_path),
        "random_state": random_state,
        "filter_min_predicted_fitness": MIN_FIT,
        "baseline_checkpoint_summary": baseline,
        "aligned_gates": aligned,
        "r2_mae_divergence": r2_mae_divergence,
        "cross_gate": {
            "pred0.5_vs_true0.5": _metrics(y05, p05),
            "pred0.95_vs_true0.95": _metrics(y95, p95),
            "pred0.95_vs_true0.5": _metrics(y05, p95),
            "pred0.5_vs_true0.95": _metrics(y95, p05),
        },
        "gate_sensitivity": {
            "frac_true_zero_flips": float(np.mean((y05 <= 0) != (y95 <= 0))),
            "frac_pred_zero_flips": float(np.mean((p05 <= 0) != (p95 <= 0))),
            "mean_abs_pred_diff": float(np.mean(np.abs(p95 - p05))),
            "median_abs_pred_diff": float(np.median(np.abs(p95 - p05))),
            "frac_pred_ext_p_in_[0.5,0.95)": float(np.mean(in_band)),
            "divergent_skip_fraction_at_min_fit_0.45": float(
                np.mean((p05 < MIN_FIT) != (p95 < MIN_FIT))
            ),
        },
        "validity_decision": validity,
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    g05 = payload["aligned_gates"]["gate_0p5"]
    g95 = payload["aligned_gates"]["gate_0p95"]
    m05 = g05["point_metrics"]
    m95 = g95["point_metrics"]
    d05 = g05["target_distribution"]
    d95 = g95["target_distribution"]
    r2_ci_05 = g05["bootstrap"]["r2_fitness_ci_95"]
    r2_ci_95 = g95["bootstrap"]["r2_fitness_ci_95"]
    mae_ci_05 = g05["bootstrap"]["mae_fitness_ci_95"]
    mae_ci_95 = g95["bootstrap"]["mae_fitness_ci_95"]
    nmae_ci_05 = g05["bootstrap"]["nmae_fitness_ci_95"]
    nmae_ci_95 = g95["bootstrap"]["nmae_fitness_ci_95"]
    sens = payload["gate_sensitivity"]
    div = payload["r2_mae_divergence"]
    dec = payload["validity_decision"]

    lines = [
        "# Hold-out compose-gate alignment (full n=7,023)",
        "",
        f"Checkpoint `{Path(payload['checkpoint']).name}` on `{Path(payload['buffer']).name}` "
        f"(hold-out split `random_state={payload['random_state']}`).",
        "",
        "## Why R²↑ and MAE↑ together",
        "",
        div["interpretation"],
        "",
        f"- Target std: {d05['std']:.4f} @0.5 → {d95['std']:.4f} @0.95 "
        f"(×{div['target_std_ratio_0p95_over_0p5']:.2f})",
        f"- Frac true zero: {100*d05['frac_zero']:.1f}% @0.5 → "
        f"{100*d95['frac_zero']:.1f}% @0.95",
        f"- MAE ratio: ×{div['mae_ratio_0p95_over_0p5']:.2f}; "
        f"NMAE: {m05['nmae_fitness']:.3f} @0.5 → {m95['nmae_fitness']:.3f} @0.95",
        "",
        "## Target distribution (composed fitness labels)",
        "",
        "| Gate | mean | std | frac zero | p25 | median | p75 | p95 |",
        "|------|------|-----|-----------|-----|--------|-----|-----|",
        (
            f"| 0.5 | {d05['mean']:.4f} | {d05['std']:.4f} | "
            f"{100*d05['frac_zero']:.1f}% | {d05['p25']:.4f} | "
            f"{d05['median']:.4f} | {d05['p75']:.4f} | {d05['p95']:.4f} |"
        ),
        (
            f"| 0.95 | {d95['mean']:.4f} | {d95['std']:.4f} | "
            f"{100*d95['frac_zero']:.1f}% | {d95['p25']:.4f} | "
            f"{d95['median']:.4f} | {d95['p75']:.4f} | {d95['p95']:.4f} |"
        ),
        "",
        "## Aligned metrics (labels and preds share gate)",
        "",
        "| Gate | R² | R² 95% CI | MAE | MAE 95% CI | NMAE | NMAE 95% CI | quality |",
        "|------|-----|-----------|-----|------------|------|-------------|---------|",
        (
            f"| 0.5 | {m05['r2_fitness']:.3f} | "
            f"[{r2_ci_05[0]:.2f}, {r2_ci_05[1]:.2f}] | {m05['mae_fitness']:.4f} | "
            f"[{mae_ci_05[0]:.4f}, {mae_ci_05[1]:.4f}] | {m05['nmae_fitness']:.3f} | "
            f"[{nmae_ci_05[0]:.3f}, {nmae_ci_05[1]:.3f}] | "
            f"{g05['quality_flags']['quality_passed']} |"
        ),
        (
            f"| 0.95 | {m95['r2_fitness']:.3f} | "
            f"[{r2_ci_95[0]:.2f}, {r2_ci_95[1]:.2f}] | {m95['mae_fitness']:.4f} | "
            f"[{mae_ci_95[0]:.4f}, {mae_ci_95[1]:.4f}] | {m95['nmae_fitness']:.3f} | "
            f"[{nmae_ci_95[0]:.3f}, {nmae_ci_95[1]:.3f}] | "
            f"{g95['quality_flags']['quality_passed']} |"
        ),
        "",
        f"MAE_stability (gate-invariant components): {g05['mae_stability']:.4f}.",
        "",
        "## Cross-gate mismatch (misaligned label/pred gates)",
        "",
        f"- pred@0.95 vs true@0.5: R²={payload['cross_gate']['pred0.95_vs_true0.5']['r2_fitness']:.3f}",
        f"- divergent skip @ min_fit={MIN_FIT}: {sens['divergent_skip_fraction_at_min_fit_0.45']:.3f}",
        f"- frac pred p_ext in [0.5, 0.95): {sens['frac_pred_ext_p_in_[0.5,0.95)']:.3f}",
        "",
        "## Validity decision",
        "",
        f"- Reproduces legacy @0.5 summary: **{dec['headline_gate_0p5_reproduces_checkpoint_summary']}**",
        f"- Aligned @0.95 passes production quality gate: **{dec['aligned_gate_0p95_passes_quality_passed']}**",
        f"- R²↑/MAE↑ is target rescaling, not model gain: **{dec['r2_up_mae_up_is_target_rescaling_not_model_gain']}**",
        "",
        dec["note"],
        "",
        f"*Generated by `scripts/analyze_holdout_compose_gate_alignment.py` (bootstrap B={BOOTSTRAP_B}).*",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer", type=Path, default=BUFFER)
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--bootstrap-b", type=int, default=BOOTSTRAP_B)
    parser.add_argument("--random-state", type=int, default=RNG)
    args = parser.parse_args()

    print("Loading buffer + checkpoint...", flush=True)
    payload = analyze_holdout_compose_gate_alignment(
        buffer_path=args.buffer,
        checkpoint_path=args.checkpoint,
        summary_path=args.summary,
        random_state=args.random_state,
        bootstrap_b=args.bootstrap_b,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(payload, args.out_md)

    g05 = payload["aligned_gates"]["gate_0p5"]["point_metrics"]
    g95 = payload["aligned_gates"]["gate_0p95"]["point_metrics"]
    div = payload["r2_mae_divergence"]
    print(
        f"gate@0.5: R²={g05['r2_fitness']:.4f}, MAE={g05['mae_fitness']:.4f}, "
        f"NMAE={g05['nmae_fitness']:.4f}, std(y)={g05['target_std']:.4f}",
        flush=True,
    )
    print(
        f"gate@0.95: R²={g95['r2_fitness']:.4f}, MAE={g95['mae_fitness']:.4f}, "
        f"NMAE={g95['nmae_fitness']:.4f}, std(y)={g95['target_std']:.4f}",
        flush=True,
    )
    print(
        f"std ratio={div['target_std_ratio_0p95_over_0p5']:.2f}, "
        f"mae ratio={div['mae_ratio_0p95_over_0p5']:.2f}",
        flush=True,
    )
    print(json.dumps(payload["validity_decision"], indent=2), flush=True)
    print(f"Wrote {args.out_json} and {args.out_md}", flush=True)


if __name__ == "__main__":
    main()

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


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "r2_fitness": float(r2_score(y, pred)),
        "mae_fitness": float(mean_absolute_error(y, pred)),
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
        aligned[key] = {
            "extinction_gate_threshold": gate,
            "point_metrics": point,
            "mae_stability": mae_stability,
            "bootstrap": {
                "B": bootstrap_b,
                "random_state": random_state,
                "r2_fitness_ci_95": list(r2_ci),
                "mae_fitness_ci_95": list(mae_ci),
            },
            "quality_flags": _quality_flags(point, mae_stability=mae_stability),
        }

    pext = np.asarray([float(c["early_extinction_prob"]) for c in pred_comps])
    y05, y95 = true_by_gate[0.5], true_by_gate[0.95]
    p05, p95 = pred_by_gate[0.5], pred_by_gate[0.95]
    in_band = (pext >= 0.5) & (pext < 0.95)

    baseline: dict[str, Any] | None = None
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        baseline = {
            "path": str(summary_path),
            "holdout_count": summary.get("holdout_count"),
            "holdout_metrics": summary.get("holdout_metrics"),
            "hints_ok": summary.get("hints_ok"),
            "quality_passed": summary.get("quality_passed"),
        }

    gate05 = aligned["gate_0p5"]["point_metrics"]
    repro_ok = (
        baseline is not None
        and abs(gate05["r2_fitness"] - float(baseline["holdout_metrics"]["r2_fitness"]))
        < 1e-3
    )

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
        "note": (
            "Aligning hold-out to runtime gate 0.95 can restore a strong validity headline "
            "when labels and preds share the gate; it does not by itself unlock confirmatory H3 "
            "or retire the need for a new live criterion."
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
    r2_ci_05 = g05["bootstrap"]["r2_fitness_ci_95"]
    r2_ci_95 = g95["bootstrap"]["r2_fitness_ci_95"]
    mae_ci_05 = g05["bootstrap"]["mae_fitness_ci_95"]
    mae_ci_95 = g95["bootstrap"]["mae_fitness_ci_95"]
    sens = payload["gate_sensitivity"]
    dec = payload["validity_decision"]

    lines = [
        "# Hold-out compose-gate alignment (full n=7,023)",
        "",
        f"Checkpoint `{Path(payload['checkpoint']).name}` on `{Path(payload['buffer']).name}` "
        f"(hold-out split `random_state={payload['random_state']}`).",
        "",
        "## Aligned metrics (labels and preds share gate)",
        "",
        "| Gate | R² | R² 95% CI | MAE | MAE 95% CI | hints_ok | quality_passed |",
        "|------|-----|-----------|-----|------------|----------|----------------|",
        (
            f"| 0.5 (code default / manuscript headline) | {m05['r2_fitness']:.3f} | "
            f"[{r2_ci_05[0]:.2f}, {r2_ci_05[1]:.2f}] | {m05['mae_fitness']:.4f} | "
            f"[{mae_ci_05[0]:.4f}, {mae_ci_05[1]:.4f}] | "
            f"{g05['quality_flags']['hints_ok']} | {g05['quality_flags']['quality_passed']} |"
        ),
        (
            f"| 0.95 (runtime YAML) | {m95['r2_fitness']:.3f} | "
            f"[{r2_ci_95[0]:.2f}, {r2_ci_95[1]:.2f}] | {m95['mae_fitness']:.4f} | "
            f"[{mae_ci_95[0]:.4f}, {mae_ci_95[1]:.4f}] | "
            f"{g95['quality_flags']['hints_ok']} | {g95['quality_flags']['quality_passed']} |"
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
        f"- Reproduces checkpoint summary @0.5: **{dec['headline_gate_0p5_reproduces_checkpoint_summary']}**",
        f"- Aligned @0.95 passes production quality gate: **{dec['aligned_gate_0p95_passes_quality_passed']}**",
        f"- Protocol fix (align hold-out → 0.95) supports validity headline: **{dec['protocol_fix_align_holdout_to_runtime_0p95_supports_validity_headline']}**",
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
    print(
        f"gate@0.5: R²={g05['r2_fitness']:.4f}, MAE={g05['mae_fitness']:.4f}",
        flush=True,
    )
    print(
        f"gate@0.95: R²={g95['r2_fitness']:.4f}, MAE={g95['mae_fitness']:.4f}",
        flush=True,
    )
    print(json.dumps(payload["validity_decision"], indent=2), flush=True)
    print(f"Wrote {args.out_json} and {args.out_md}", flush=True)


if __name__ == "__main__":
    main()

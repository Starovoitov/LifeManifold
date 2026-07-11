#!/usr/bin/env python3
"""D1 quantitative checks: hard/soft compose A/B + gate 0.5 vs 0.95 (batched)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.illuminators.evaluation import compute_fitness
from worldspace.metrics import WorldMetrics
from worldspace.surrogate.checkpoint_io import load_surrogate_checkpoint
from worldspace.surrogate.model import TARGET_KEYS
from worldspace.surrogate.training import holdout_split, load_buffer
from worldspace.surrogate.types import SurrogatePrediction
from worldspace.surrogate.utils import (
    compute_fitness_from_prediction,
    compute_soft_fitness_from_prediction,
)

BUFFER = ROOT / "artifacts/surrogate/buffer_nightly.jsonl"
CKPT = ROOT / "artifacts/surrogate/checkpoints/nightly_v3_mc_d005.pkl"
OUT_AB = ROOT / "artifacts/surrogate/compose_ab_check.json"
OUT_GATE = ROOT / "artifacts/surrogate/compose_gate_0p5_vs_0p95.json"
MIN_FIT = 0.45
RNG = 42
MAX_ROWS = 1500


def _mets(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y, p)),
        "mae": float(mean_absolute_error(y, p)),
        "frac_pred_zero": float(np.mean(p <= 0)),
        "frac_true_zero": float(np.mean(y <= 0)),
    }


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


def main() -> None:
    print("Loading buffer + checkpoint...", flush=True)
    features, targets = load_buffer(BUFFER)
    _x_tr, _y_tr, x_hold, y_hold = holdout_split(features, targets, random_state=RNG)
    if int(x_hold.shape[0]) > MAX_ROWS:
        rng = np.random.default_rng(RNG)
        idx = np.sort(rng.choice(x_hold.shape[0], size=MAX_ROWS, replace=False))
        x_hold = x_hold[idx]
        y_hold = {key: arr[idx] for key, arr in y_hold.items()}
    model = load_surrogate_checkpoint(CKPT)
    n_hold = int(x_hold.shape[0])
    print(f"holdout n={n_hold}; batch predicting components...", flush=True)

    pred_comps = model.predict_components_batch(x_hold)
    print("composing fitness...", flush=True)

    y05 = np.asarray([_true_fitness(y_hold, i, 0.5) for i in range(n_hold)])
    y95 = np.asarray([_true_fitness(y_hold, i, 0.95) for i in range(n_hold)])
    # true fitness for soft/hard AB uses default target compose (gate 0.5 labels)
    y_ab = y05

    p_hard_05 = np.empty(n_hold)
    p_hard_95 = np.empty(n_hold)
    p_soft = np.empty(n_hold)
    pext = np.empty(n_hold)
    for i, comps in enumerate(pred_comps):
        pred = SurrogatePrediction(
            components=comps,
            measures={
                "stability": float(comps["stability"]),
                "diversity": float(comps["diversity"]),
            },
            fitness=0.0,
            uncertainty=0.0,
        )
        p_hard_05[i] = compute_fitness_from_prediction(
            pred, extinction_gate_threshold=0.5
        )
        p_hard_95[i] = compute_fitness_from_prediction(
            pred, extinction_gate_threshold=0.95
        )
        p_soft[i] = compute_soft_fitness_from_prediction(pred)
        pext[i] = float(comps["early_extinction_prob"])

    ab = {
        "hard": _mets(y_ab, p_hard_05),
        "soft": _mets(y_ab, p_soft),
    }
    # rename keys to match evaluate_fitness_compose_ab
    ab = {
        "hard": {"r2_fitness": ab["hard"]["r2"], "mae_fitness": ab["hard"]["mae"]},
        "soft": {"r2_fitness": ab["soft"]["r2"], "mae_fitness": ab["soft"]["mae"]},
    }
    OUT_AB.parent.mkdir(parents=True, exist_ok=True)
    OUT_AB.write_text(
        json.dumps(
            {
                "fitness_compose_ab": ab,
                "n_holdout": n_hold,
                "checkpoint": str(CKPT),
                "subsample_max_rows": MAX_ROWS,
                "random_state": RNG,
                "note": (
                    "hard vs soft compose on production checkpoint; "
                    "hard uses gate 0.5; labels composed from buffer targets at gate 0.5"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("compose_ab:", json.dumps(ab, indent=2), flush=True)

    in_band = (pext >= 0.5) & (pext < 0.95)
    divergent_skip = float(np.mean((p_hard_05 < MIN_FIT) != (p_hard_95 < MIN_FIT)))
    gate = {
        "n_holdout": n_hold,
        "checkpoint": str(CKPT),
        "random_state": RNG,
        "subsample_max_rows": MAX_ROWS,
        "filter_min_predicted_fitness": MIN_FIT,
        "paired_metrics": {
            "pred0.5_vs_true0.5": _mets(y05, p_hard_05),
            "pred0.95_vs_true0.95": _mets(y95, p_hard_95),
            "pred0.95_vs_true0.5": _mets(y05, p_hard_95),
            "pred0.5_vs_true0.95": _mets(y95, p_hard_05),
        },
        "gate_sensitivity": {
            "frac_true_zero_flips": float(np.mean((y05 <= 0) != (y95 <= 0))),
            "frac_pred_zero_flips": float(
                np.mean((p_hard_05 <= 0) != (p_hard_95 <= 0))
            ),
            "mean_abs_pred_diff": float(np.mean(np.abs(p_hard_95 - p_hard_05))),
            "median_abs_pred_diff": float(np.median(np.abs(p_hard_95 - p_hard_05))),
            "frac_pred_ext_p_in_[0.5,0.95)": float(np.mean(in_band)),
            "divergent_skip_fraction_at_min_fit_0.45": divergent_skip,
            "RQ3_confirmatory_rule_div_le_0.05": divergent_skip <= 0.05,
        },
    }
    OUT_GATE.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate["gate_sensitivity"], indent=2), flush=True)
    print(f"Wrote {OUT_AB} and {OUT_GATE}", flush=True)


if __name__ == "__main__":
    main()

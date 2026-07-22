#!/usr/bin/env python3
"""M1 Phase 2: offline acquisition replay — threshold_gate vs ucb_promote.

Merges ``acquisition_replay`` into ``artifacts/surrogate/gp_ucb_ablation.json``.
Uses the same hold-out split / GP subsample locks as Phase 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.surrogate.acquisition_config import AcquisitionConfig  # noqa: E402
from worldspace.surrogate.calibration import (  # noqa: E402
    apply_calibrated_uncertainty,
    load_uncertainty_calibration,
)
from worldspace.surrogate.checkpoint_io import load_surrogate_checkpoint  # noqa: E402
from worldspace.surrogate.evaluation import fitness_from_target_row  # noqa: E402
from worldspace.surrogate.model import TARGET_KEYS, SurrogateModel  # noqa: E402
from worldspace.surrogate.reporting import (  # noqa: E402
    evaluate_acquisition_replay_from_scores,
)
from worldspace.surrogate.training import holdout_split, load_buffer  # noqa: E402
from worldspace.surrogate.types import SurrogatePrediction  # noqa: E402
from worldspace.surrogate.utils import resolve_surrogate_fitness  # noqa: E402

DEFAULT_BUFFER = ROOT / "artifacts/surrogate/buffer_nightly.jsonl"
DEFAULT_CKPT = ROOT / "artifacts/surrogate/checkpoints/nightly_v3_mc_d005.pkl"
DEFAULT_CAL = ROOT / "artifacts/surrogate/checkpoints/calibration_v3_mc_d005.pkl"
DEFAULT_OUT = ROOT / "artifacts/surrogate/gp_ucb_ablation.json"
UCB_BETAS = (0.15, 0.5, 1.0)
SHADOW_SKIP_BAND = (0.25, 0.45)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M1 Phase 2: threshold_gate vs UCB acquisition replay",
    )
    parser.add_argument("--buffer-path", type=Path, default=DEFAULT_BUFFER)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CAL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--gp-max-train", type=int, default=5000)
    parser.add_argument("--min-predicted-fitness", type=float, default=0.45)
    parser.add_argument("--max-uncertainty-to-skip", type=float, default=1.0)
    parser.add_argument(
        "--uncertainty-sample",
        type=int,
        default=500,
        help="Hold-out rows for MLP+UCB (MC-dropout cost); default 500",
    )
    parser.add_argument(
        "--ucb-betas",
        type=str,
        default=",".join(str(b) for b in UCB_BETAS),
        help="Comma-separated exploration_weight values for ucb_promote",
    )
    return parser.parse_args()


def _composed_fitness_labels(targets: dict[str, np.ndarray]) -> np.ndarray:
    n_rows = int(next(iter(targets.values())).shape[0])
    return np.asarray(
        [
            fitness_from_target_row({k: float(targets[k][i]) for k in TARGET_KEYS})
            for i in range(n_rows)
        ],
        dtype=float,
    )


def _mlp_fitness_batch(model: SurrogateModel, feature_matrix: np.ndarray) -> np.ndarray:
    n_rows = int(feature_matrix.shape[0])
    fitness = np.empty(n_rows, dtype=float)
    components_batch = model.predict_components_batch(feature_matrix)
    for i in range(n_rows):
        components = components_batch[i]
        prediction = SurrogatePrediction(
            components=components,
            measures={
                "stability": float(components["stability"]),
                "diversity": float(components["diversity"]),
            },
            fitness=0.0,
            uncertainty=0.0,
        )
        fitness[i] = resolve_surrogate_fitness(model, feature_matrix[i], prediction)
    return fitness


def _subsample_train(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    max_n: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    n = int(train_x.shape[0])
    if max_n <= 0 or n <= max_n:
        return train_x, train_y, n
    rng = np.random.default_rng(random_state)
    idx = np.sort(rng.choice(n, size=max_n, replace=False))
    return train_x[idx], train_y[idx], int(max_n)


def _fit_gp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    random_state: int,
) -> tuple[GaussianProcessRegressor, StandardScaler, float]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(train_x)
    kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=random_state,
    )
    t0 = time.perf_counter()
    gp.fit(x_scaled, train_y)
    return gp, scaler, float(time.perf_counter() - t0)


def _gp_predict(
    gp: GaussianProcessRegressor,
    scaler: StandardScaler,
    feature_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mu, std = gp.predict(scaler.transform(feature_matrix), return_std=True)
    return np.asarray(mu, dtype=float), np.asarray(std, dtype=float)


def _in_shadow_band(skip_rate: float) -> bool:
    lo, hi = SHADOW_SKIP_BAND
    return lo <= float(skip_rate) <= hi


def _row(
    *,
    policy: str,
    regressor: str,
    metrics: Any,
    exploration_weight: float | None = None,
    note: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    block = {
        "policy": policy,
        "regressor": regressor,
        "exploration_weight": exploration_weight,
        "n": metrics.row_count,
        "recommended_skip_rate": metrics.recommended_skip_rate,
        "policy_skip_count": metrics.policy_skip_count,
        "false_skip_rate_estimate": metrics.false_skip_rate_estimate,
        "false_skip_count": metrics.false_skip_count,
        "consistency_mae": metrics.consistency_mae,
        "calibration_ece": metrics.calibration_ece,
        "in_shadow_skip_band_25_45": _in_shadow_band(metrics.recommended_skip_rate),
        "note": note,
    }
    if extra:
        block.update(extra)
    return block


def _decision_agreement(
    fitness_a: np.ndarray,
    unc_a: np.ndarray,
    fitness_b: np.ndarray,
    unc_b: np.ndarray,
    actual: np.ndarray,
    policy_a: AcquisitionConfig,
    policy_b: AcquisitionConfig,
) -> dict[str, float]:
    """Fraction of rows where both policies agree on skip vs eval."""
    from worldspace.illuminators.archive_factory import (
        ArchiveFactoryConfig,
        create_archive,
    )
    from worldspace.illuminators.scheduler import TargetBin
    from worldspace.surrogate.acquisition import decide
    from worldspace.surrogate.types import SurrogatePrediction

    archive = create_archive(ArchiveFactoryConfig(archive_type="grid", resolution=10))
    n_cells = archive.n_cells
    n = int(actual.shape[0])
    agree = 0
    both_skip = 0
    skip_a = 0
    skip_b = 0
    for i in range(n):
        cell = i % n_cells
        target = TargetBin(
            bin=archive.bin_from_cell_id(cell),
            target_stability=0.5,
            target_diversity=0.5,
        )
        pa = decide(
            policy_a,
            SurrogatePrediction(
                components={},
                measures={},
                fitness=float(fitness_a[i]),
                uncertainty=float(unc_a[i]),
            ),
            target,
            archive,
        )
        pb = decide(
            policy_b,
            SurrogatePrediction(
                components={},
                measures={},
                fitness=float(fitness_b[i]),
                uncertainty=float(unc_b[i]),
            ),
            target,
            archive,
        )
        if pa.action == "skip":
            skip_a += 1
        if pb.action == "skip":
            skip_b += 1
        if pa.action == pb.action:
            agree += 1
        if pa.action == "skip" and pb.action == "skip":
            both_skip += 1
    return {
        "agreement_rate": float(agree) / float(n) if n else float("nan"),
        "both_skip_rate": float(both_skip) / float(n) if n else float("nan"),
        "skip_rate_a": float(skip_a) / float(n) if n else float("nan"),
        "skip_rate_b": float(skip_b) / float(n) if n else float("nan"),
    }


def main() -> None:
    args = parse_args()
    betas = [float(x.strip()) for x in args.ucb_betas.split(",") if x.strip()]
    print(f"loading buffer {args.buffer_path}", file=sys.stderr)
    feature_matrix, targets = load_buffer(args.buffer_path)
    train_x, train_targets, hold_x, hold_targets = holdout_split(
        feature_matrix,
        targets,
        test_fraction=args.test_fraction,
        random_state=args.random_state,
    )
    train_y = _composed_fitness_labels(train_targets)
    hold_y = _composed_fitness_labels(hold_targets)
    print(
        f"holdout={hold_x.shape[0]} train={train_x.shape[0]}",
        file=sys.stderr,
    )

    calibrator = None
    if args.calibration_path.is_file():
        calibrator = load_uncertainty_calibration(args.calibration_path)

    model = load_surrogate_checkpoint(args.checkpoint_path)
    print("MLP fitness...", file=sys.stderr)
    mlp_fit = _mlp_fitness_batch(model, hold_x)
    # Production max_u=1.0 → uncertainty not load-bearing for threshold_gate.
    mlp_u_zero = np.zeros_like(mlp_fit)

    gp_x, gp_y, gp_n = _subsample_train(
        train_x, train_y, max_n=int(args.gp_max_train), random_state=args.random_state
    )
    print(f"fitting GP n={gp_n}...", file=sys.stderr)
    gp, scaler, gp_train_s = _fit_gp(gp_x, gp_y, random_state=args.random_state)
    gp_fit, gp_u = _gp_predict(gp, scaler, hold_x)
    print(f"GP train {gp_train_s:.1f}s", file=sys.stderr)

    base_tg = AcquisitionConfig(
        mode="filter",
        policy="threshold_gate",
        min_predicted_fitness=float(args.min_predicted_fitness),
        max_uncertainty_to_skip=float(args.max_uncertainty_to_skip),
        never_skip_empty_bin=False,
    )
    rows: list[dict[str, Any]] = []

    m_mlp_tg = evaluate_acquisition_replay_from_scores(
        mlp_fit, mlp_u_zero, hold_y, base_tg, never_skip_empty_bin=False
    )
    rows.append(
        _row(
            policy="threshold_gate",
            regressor="mlp",
            metrics=m_mlp_tg,
            note=(
                "Production knobs min_fit=0.45, max_u=1.0; u forced 0 "
                "(max_u=1.0 makes calibrated MC-u non-binding for skip)."
            ),
            extra={"uncertainty_mode": "zero_nonbinding"},
        )
    )

    m_gp_tg = evaluate_acquisition_replay_from_scores(
        gp_fit, gp_u, hold_y, base_tg, never_skip_empty_bin=False
    )
    rows.append(
        _row(
            policy="threshold_gate",
            regressor="gp",
            metrics=m_gp_tg,
            note="Same threshold_gate on GP μ/σ (σ unused when max_u=1.0 if σ≤1).",
            extra={"uncertainty_mode": "gp_posterior_std"},
        )
    )

    for beta in betas:
        policy = replace(
            base_tg,
            policy="ucb_promote",
            exploration_weight=float(beta),
        )
        m = evaluate_acquisition_replay_from_scores(
            gp_fit, gp_u, hold_y, policy, never_skip_empty_bin=False
        )
        rows.append(
            _row(
                policy="ucb_promote",
                regressor="gp",
                metrics=m,
                exploration_weight=float(beta),
                note="UCB = μ + β·σ; skip if UCB < min_predicted_fitness.",
                extra={"uncertainty_mode": "gp_posterior_std"},
            )
        )

    # MLP + UCB on MC-dropout subsample (expensive).
    rng = np.random.default_rng(args.random_state)
    take = min(int(args.uncertainty_sample), int(hold_x.shape[0]))
    idx = np.sort(rng.choice(hold_x.shape[0], size=take, replace=False))
    print(f"MLP MC-dropout u on n={take}...", file=sys.stderr)
    t0 = time.perf_counter()
    raw_u = np.asarray(model.predict_uncertainty_batch(hold_x[idx]), dtype=float)
    mlp_u_sample = np.asarray(
        [
            apply_calibrated_uncertainty(
                calibrator,
                max(0.0, float(u)),
                calibration_configured=calibrator is not None,
            )
            for u in raw_u
        ],
        dtype=float,
    )
    unc_s = time.perf_counter() - t0
    for beta in betas:
        policy = AcquisitionConfig(
            mode="filter",
            policy="ucb_promote",
            min_predicted_fitness=float(args.min_predicted_fitness),
            max_uncertainty_to_skip=float(args.max_uncertainty_to_skip),
            never_skip_empty_bin=False,
            exploration_weight=float(beta),
        )
        m = evaluate_acquisition_replay_from_scores(
            mlp_fit[idx],
            mlp_u_sample,
            hold_y[idx],
            policy,
            never_skip_empty_bin=False,
        )
        rows.append(
            _row(
                policy="ucb_promote",
                regressor="mlp",
                metrics=m,
                exploration_weight=float(beta),
                note="UCB on MLP μ + calibrated MC-dropout σ (subsample).",
                extra={
                    "uncertainty_mode": "mlp_mc_calibrated_subsample",
                    "subsample_n": take,
                    "uncertainty_seconds": float(unc_s),
                },
            )
        )

    # Agreement: production threshold_gate(MLP) vs UCB(GP, β=0.15)
    agree = _decision_agreement(
        mlp_fit,
        mlp_u_zero,
        gp_fit,
        gp_u,
        hold_y,
        base_tg,
        replace(base_tg, policy="ucb_promote", exploration_weight=0.15),
    )

    # Dual-report: never_skip_empty_bin=True on empty replay archive → skip≈0
    m_empty = evaluate_acquisition_replay_from_scores(
        mlp_fit,
        mlp_u_zero,
        hold_y,
        replace(base_tg, never_skip_empty_bin=True),
        never_skip_empty_bin=True,
    )

    payload_block: dict[str, Any] = {
        "phase": 2,
        "min_predicted_fitness": float(args.min_predicted_fitness),
        "max_uncertainty_to_skip": float(args.max_uncertainty_to_skip),
        "never_skip_empty_bin_replay": False,
        "shadow_skip_band": list(SHADOW_SKIP_BAND),
        "gp_train_n": gp_n,
        "gp_train_seconds": gp_train_s,
        "rows": rows,
        "agreement_threshold_mlp_vs_ucb_gp_beta_0_15": agree,
        "empty_bin_dual_report": {
            "never_skip_empty_bin": True,
            "recommended_skip_rate": m_empty.recommended_skip_rate,
            "note": (
                "Empty synthetic archive + never_skip_empty_bin=true forces eval "
                "on every row (skip≈0). Offline metrics use never_skip_empty_bin=false."
            ),
        },
        "verdict_notes": [],
    }

    prod = next(
        r for r in rows if r["policy"] == "threshold_gate" and r["regressor"] == "mlp"
    )
    ucb_gp = [
        r for r in rows if r["policy"] == "ucb_promote" and r["regressor"] == "gp"
    ]
    in_band = [r for r in ucb_gp if r["in_shadow_skip_band_25_45"]]
    payload_block["verdict_notes"] = [
        (
            f"Production threshold_gate+MLP skip_rate="
            f"{prod['recommended_skip_rate']:.3f} "
            f"(false_skip={prod['false_skip_rate_estimate']:.3f}); "
            f"shadow band 25–45%: {prod['in_shadow_skip_band_25_45']}."
        ),
        (
            "GP+UCB β sweep skip rates: "
            + ", ".join(
                f"β={r['exploration_weight']}→{r['recommended_skip_rate']:.3f}"
                for r in ucb_gp
            )
            + (f"; in-band: {len(in_band)}/{len(ucb_gp)}.")
        ),
        (
            "UCB raises eval tendency via β·σ (opposite of threshold_gate's "
            "max_u latch); with production max_u=1.0, threshold_gate ignores σ."
        ),
        (
            f"Decision agreement threshold_gate(MLP) vs ucb_promote(GP,β=0.15): "
            f"{agree['agreement_rate']:.3f}."
        ),
    ]

    out_path = args.output_json
    payload: dict[str, Any]
    if out_path.is_file():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        payload = {"family": "M1", "title": "MC-dropout MLP vs GP+UCB"}
    payload["phase"] = 2
    payload["acquisition_replay"] = payload_block
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows, "agreement": agree}, indent=2))
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

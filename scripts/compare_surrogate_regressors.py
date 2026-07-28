#!/usr/bin/env python3
"""M1 Phase 1: offline hold-out comparison of production MLP vs GP fitness regressor.

Writes ``artifacts/surrogate/gp_ucb_ablation.json`` (``holdout_regression`` block).
Does not run acquisition replay (Phase 2) or online MAP-Elites.

Note: production checkpoint uses ``ensemble_mc`` uncertainty (~tens of ms/row).
Hold-out R²/MAE use fast component fitness; uncertainty/ECE use a subsample.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.surrogate.calibration import (  # noqa: E402
    apply_calibrated_uncertainty,
    expected_calibration_error,
    load_uncertainty_calibration,
)
from worldspace.surrogate.checkpoint_io import load_surrogate_checkpoint  # noqa: E402
from worldspace.surrogate.evaluation import (  # noqa: E402
    PRODUCTION_EXTINCTION_GATE_THRESHOLD,
    fitness_from_target_row,
)
from worldspace.surrogate.model import (
    FITNESS_TARGET_KEY,
    TARGET_KEYS,
    SurrogateModel,
)  # noqa: E402
from worldspace.surrogate.training import holdout_split, load_buffer  # noqa: E402
from worldspace.surrogate.types import SurrogatePrediction  # noqa: E402
from worldspace.surrogate.utils import resolve_surrogate_fitness  # noqa: E402

DEFAULT_BUFFER = ROOT / "artifacts/surrogate/buffer_nightly.jsonl"
DEFAULT_CKPT = ROOT / "artifacts/surrogate/checkpoints/nightly_v3_mc_d005.pkl"
DEFAULT_CAL = ROOT / "artifacts/surrogate/checkpoints/calibration_v3_mc_d005.pkl"
DEFAULT_OUT = ROOT / "artifacts/surrogate/gp_ucb_ablation.json"
BATCH_PROXY_N = 32_500  # 650 iter × 50 batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MLP checkpoint vs GP on composed-fitness hold-out (M1 Phase 1)",
    )
    parser.add_argument("--buffer-path", type=Path, default=DEFAULT_BUFFER)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CAL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--gp-max-train",
        type=int,
        default=5000,
        help="Max GP training rows after hold-out split (O(n³); default 5000)",
    )
    parser.add_argument(
        "--timing-batch-n",
        type=int,
        default=BATCH_PROXY_N,
        help="Batch size for fitness predict wall-time proxy (default 32500)",
    )
    parser.add_argument(
        "--uncertainty-sample",
        type=int,
        default=1000,
        help="Hold-out rows for MLP MC-dropout uncertainty/ECE (default 1000)",
    )
    parser.add_argument(
        "--skip-gp",
        action="store_true",
        help="Only score MLP (debug / smoke)",
    )
    return parser.parse_args()


def _composed_fitness_labels(
    targets: dict[str, np.ndarray],
    *,
    extinction_gate_threshold: float = PRODUCTION_EXTINCTION_GATE_THRESHOLD,
) -> np.ndarray:
    n_rows = int(next(iter(targets.values())).shape[0])
    return np.asarray(
        [
            fitness_from_target_row(
                {k: float(targets[k][i]) for k in TARGET_KEYS},
                extinction_gate_threshold=extinction_gate_threshold,
            )
            for i in range(n_rows)
        ],
        dtype=float,
    )


def _mlp_fitness_batch(
    model: SurrogateModel,
    feature_matrix: np.ndarray,
    *,
    extinction_gate_threshold: float = PRODUCTION_EXTINCTION_GATE_THRESHOLD,
) -> np.ndarray:
    """Fast fitness predictions (components / direct head; no MC-dropout)."""
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
        fitness[i] = resolve_surrogate_fitness(
            model,
            feature_matrix[i],
            prediction,
            extinction_gate_threshold=extinction_gate_threshold,
        )
    return fitness


def _mlp_uncertainty_sample(
    model: SurrogateModel,
    feature_matrix: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    sample_n: int,
    random_state: int,
    calibrator: Any | None,
) -> dict[str, Any]:
    n = int(feature_matrix.shape[0])
    if sample_n <= 0 or n == 0:
        return {"n": 0}
    rng = np.random.default_rng(random_state)
    take = min(int(sample_n), n)
    idx = np.sort(rng.choice(n, size=take, replace=False))
    t0 = time.perf_counter()
    raw = np.asarray(
        model.predict_uncertainty_batch(feature_matrix[idx]),
        dtype=float,
    )
    elapsed = time.perf_counter() - t0
    cal = np.asarray(
        [
            apply_calibrated_uncertainty(
                calibrator,
                max(0.0, float(u)),
                calibration_configured=calibrator is not None,
            )
            for u in raw
        ],
        dtype=float,
    )
    abs_err = np.abs(y_true[idx] - y_pred[idx])
    return {
        "n": take,
        "seconds": float(elapsed),
        "seconds_per_row": float(elapsed / take) if take else None,
        "extrapolated_seconds_full_holdout": (
            float(elapsed / take * n) if take else None
        ),
        "extrapolated_seconds_batch_32500": (
            float(elapsed / take * 32500) if take else None
        ),
        "raw_uncertainty_mean": float(np.mean(raw)),
        "uncertainty_mean": float(np.mean(cal)),
        "uncertainty_std": float(np.std(cal, ddof=1)) if take > 1 else 0.0,
        "uncertainty_min": float(np.min(cal)),
        "uncertainty_max": float(np.max(cal)),
        "calibration_ece": float(expected_calibration_error(cal, abs_err)),
    }


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
    # Isotropic RBF: anisotropic length_scale=ones(D) is slow to optimize.
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
    train_s = time.perf_counter() - t0
    return gp, scaler, float(train_s)


def _gp_predict(
    gp: GaussianProcessRegressor,
    scaler: StandardScaler,
    feature_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_scaled = scaler.transform(feature_matrix)
    mu, std = gp.predict(x_scaled, return_std=True)
    return np.asarray(mu, dtype=float), np.asarray(std, dtype=float)


def _regression_block(
    *,
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray | None,
    train_seconds: float | None,
    predict_holdout_seconds: float,
    predict_batch_seconds: float | None,
    predict_batch_n: int | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    abs_err = np.abs(y_true - y_pred)
    target_std = float(np.std(y_true, ddof=0))
    mae = float(mean_absolute_error(y_true, y_pred))
    block: dict[str, Any] = {
        "regressor": name,
        "n_holdout": int(y_true.shape[0]),
        "r2_fitness": float(r2_score(y_true, y_pred)),
        "mae_fitness": mae,
        "nmae_fitness": float(mae / target_std) if target_std > 0 else float("nan"),
        "rmse_fitness": float(np.sqrt(np.mean(abs_err**2))),
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred, ddof=1)),
        "true_mean": float(np.mean(y_true)),
        "true_std": target_std,
        "frac_true_zero": float(np.mean(y_true <= 0)),
        "train_seconds": train_seconds,
        "predict_holdout_seconds": float(predict_holdout_seconds),
        "predict_batch_seconds": predict_batch_seconds,
        "predict_batch_n": predict_batch_n,
    }
    if uncertainty is not None and uncertainty.size:
        block["uncertainty_mean"] = float(np.mean(uncertainty))
        block["uncertainty_std"] = float(np.std(uncertainty, ddof=1))
        block["uncertainty_min"] = float(np.min(uncertainty))
        block["uncertainty_max"] = float(np.max(uncertainty))
        block["calibration_ece"] = float(
            expected_calibration_error(uncertainty, abs_err)
        )
    if extra:
        block.update(extra)
    return block


def _tile_rows(template_x: np.ndarray, n: int) -> np.ndarray:
    reps = int(np.ceil(n / template_x.shape[0]))
    return np.tile(template_x, (reps, 1))[:n]


def main() -> None:
    args = parse_args()
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
        f"split train={train_x.shape[0]} holdout={hold_x.shape[0]} dim={hold_x.shape[1]}",
        file=sys.stderr,
    )

    calibrator = None
    if args.calibration_path.is_file():
        calibrator = load_uncertainty_calibration(args.calibration_path)

    print(f"loading MLP {args.checkpoint_path}", file=sys.stderr)
    model = load_surrogate_checkpoint(args.checkpoint_path)
    t0 = time.perf_counter()
    mlp_pred = _mlp_fitness_batch(model, hold_x)
    mlp_hold_s = time.perf_counter() - t0
    print(f"MLP hold-out fitness {mlp_hold_s:.2f}s", file=sys.stderr)

    batch = _tile_rows(hold_x, int(args.timing_batch_n))
    t0 = time.perf_counter()
    _mlp_fitness_batch(model, batch)
    mlp_batch_s = time.perf_counter() - t0
    print(
        f"MLP fitness batch n={args.timing_batch_n} {mlp_batch_s:.2f}s",
        file=sys.stderr,
    )

    unc_block = _mlp_uncertainty_sample(
        model,
        hold_x,
        hold_y,
        mlp_pred,
        sample_n=int(args.uncertainty_sample),
        random_state=args.random_state,
        calibrator=calibrator,
    )
    print(
        f"MLP uncertainty sample n={unc_block.get('n')} "
        f"{unc_block.get('seconds', float('nan')):.2f}s",
        file=sys.stderr,
    )

    mlp_block = _regression_block(
        name="mlp_production",
        y_true=hold_y,
        y_pred=mlp_pred,
        uncertainty=None,
        train_seconds=None,
        predict_holdout_seconds=mlp_hold_s,
        predict_batch_seconds=mlp_batch_s,
        predict_batch_n=int(args.timing_batch_n),
        extra={
            "checkpoint": str(args.checkpoint_path),
            "calibration": str(args.calibration_path) if calibrator else None,
            "has_fitness_head": bool(getattr(model, "_has_fitness_head", False)),
            "mlp_uncertainty_method": str(
                getattr(model, "_mlp_uncertainty_method", "")
            ),
            "predict_path": "resolve_surrogate_fitness (components/composed; no MC per row)",
            "uncertainty_sample": unc_block,
            "note": (
                "Primary R²/MAE use fast fitness path. MC-dropout uncertainty/ECE "
                "reported on a hold-out subsample (see uncertainty_sample)."
            ),
        },
    )

    gp_block: dict[str, Any] | None = None
    if not args.skip_gp:
        gp_x, gp_y, gp_n = _subsample_train(
            train_x,
            train_y,
            max_n=int(args.gp_max_train),
            random_state=args.random_state,
        )
        print(f"fitting GP on n={gp_n} ...", file=sys.stderr)
        gp, scaler, gp_train_s = _fit_gp(gp_x, gp_y, random_state=args.random_state)
        print(f"GP train {gp_train_s:.2f}s kernel={gp.kernel_}", file=sys.stderr)
        t0 = time.perf_counter()
        gp_pred, gp_std = _gp_predict(gp, scaler, hold_x)
        gp_hold_s = time.perf_counter() - t0
        print(f"GP hold-out predict {gp_hold_s:.2f}s", file=sys.stderr)
        t0 = time.perf_counter()
        _gp_predict(gp, scaler, batch)
        gp_batch_s = time.perf_counter() - t0
        print(
            f"GP batch n={args.timing_batch_n} {gp_batch_s:.2f}s",
            file=sys.stderr,
        )
        gp_block = _regression_block(
            name="gp_sklearn",
            y_true=hold_y,
            y_pred=gp_pred,
            uncertainty=gp_std,
            train_seconds=gp_train_s,
            predict_holdout_seconds=gp_hold_s,
            predict_batch_seconds=gp_batch_s,
            predict_batch_n=int(args.timing_batch_n),
            extra={
                "gp_train_n": gp_n,
                "gp_train_pool_n": int(train_x.shape[0]),
                "gp_max_train": int(args.gp_max_train),
                "kernel": str(gp.kernel_),
                "note": (
                    "GaussianProcessRegressor on composed fitness; "
                    "features StandardScaler; isotropic RBF+WhiteKernel; normalize_y=True."
                ),
            },
        )

    payload: dict[str, Any] = {
        "family": "M1",
        "phase": 1,
        "title": "MC-dropout MLP vs GP hold-out regression",
        "locked": {
            "buffer": str(args.buffer_path),
            "test_fraction": float(args.test_fraction),
            "random_state": int(args.random_state),
            "target": "composed_illuminator_fitness",
            "extinction_gate_threshold": PRODUCTION_EXTINCTION_GATE_THRESHOLD,
            "gp_max_train": int(args.gp_max_train),
            "timing_batch_n": int(args.timing_batch_n),
            "uncertainty_sample": int(args.uncertainty_sample),
            "production_acquisition_deferred_to_phase_2": {
                "min_predicted_fitness": 0.45,
                "max_uncertainty_to_skip": 1.0,
                "never_skip_empty_bin": True,
            },
        },
        "data": {
            "n_buffer": int(feature_matrix.shape[0]),
            "feature_dim": int(feature_matrix.shape[1]),
            "n_train": int(train_x.shape[0]),
            "n_holdout": int(hold_x.shape[0]),
            "finite_stored_fitness_train": (
                int(
                    np.isfinite(
                        train_targets.get(FITNESS_TARGET_KEY, np.array([]))
                    ).sum()
                )
                if FITNESS_TARGET_KEY in train_targets
                else 0
            ),
        },
        "holdout_regression": {
            "mlp": mlp_block,
            "gp": gp_block,
        },
    }
    if gp_block is not None:
        payload["holdout_regression"]["delta_gp_minus_mlp"] = {
            "r2_fitness": float(gp_block["r2_fitness"] - mlp_block["r2_fitness"]),
            "mae_fitness": float(gp_block["mae_fitness"] - mlp_block["mae_fitness"]),
            "nmae_fitness": float(gp_block["nmae_fitness"] - mlp_block["nmae_fitness"]),
            "predict_batch_seconds": float(
                gp_block["predict_batch_seconds"] - mlp_block["predict_batch_seconds"]
            ),
            "predict_batch_speedup_mlp_over_gp": (
                float(
                    gp_block["predict_batch_seconds"]
                    / mlp_block["predict_batch_seconds"]
                )
                if mlp_block["predict_batch_seconds"]
                else None
            ),
            "note": (
                "Labels and composed MLP fitness use extinction_gate_threshold="
                f"{PRODUCTION_EXTINCTION_GATE_THRESHOLD} (runtime-aligned). "
                "Legacy M1 Phase 1 used gate 0.5 (R² MLP≈0.76 / GP≈0.22)."
            ),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.output_json.is_file():
        existing = json.loads(args.output_json.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            existing.update(
                {
                    "family": payload["family"],
                    "phase": payload["phase"],
                    "title": payload["title"],
                    "locked": payload["locked"],
                    "data": payload["data"],
                    "holdout_regression": payload["holdout_regression"],
                }
            )
            payload = existing
    args.output_json.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["holdout_regression"], indent=2))
    print(f"wrote {args.output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()

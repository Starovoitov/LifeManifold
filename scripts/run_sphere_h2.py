#!/usr/bin/env python3
"""CLI: Sphere H2 after-generation filter (supplementary; no LLM)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.benchmarks.sphere_h2 import (
    DEFAULT_PROPOSALS,
    DEFAULT_TAU,
    SphereH2Config,
    calibrate_tau,
    run_sphere_h2,
    train_sphere_surrogate,
)

DEFAULT_CKPT = ROOT / "artifacts/surrogate/sphere_h2_mlp.joblib"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train + calibrate sphere surrogate")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--n-train", type=int, default=50_000)
    t.add_argument("--target-skip", type=float, default=0.30)
    t.add_argument(
        "--calibrate-mode",
        choices=("me_like", "box"),
        default="me_like",
        help="me_like (default) matches live ME proposals; box is too lenient.",
    )
    t.add_argument("--out", type=Path, default=DEFAULT_CKPT)

    r = sub.add_parser("run", help="Run one Sphere H2 arm/seed")
    r.add_argument("--arm", choices=("me_uniform", "me_filter"), required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--proposals", type=int, default=DEFAULT_PROPOSALS)
    r.add_argument("--tau", type=float, default=None)
    r.add_argument("--surrogate", type=Path, default=DEFAULT_CKPT)
    r.add_argument("--output-dir", type=Path, required=True)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    if args.cmd == "train":
        sur = train_sphere_surrogate(seed=args.seed, n_train=args.n_train)
        tau = calibrate_tau(
            sur,
            target_skip=args.target_skip,
            seed=args.seed + 1,
            mode=args.calibrate_mode,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "surrogate": sur,
            "tau": tau,
            "target_skip": args.target_skip,
            "calibrate_mode": args.calibrate_mode,
            "train_mae": sur.train_mae,
            "train_r2": sur.train_r2,
            "n_train": sur.n_train,
        }
        joblib.dump(payload, args.out)
        logging.info(
            "Wrote %s  R2=%.4f MAE=%.4f tau=%.4f (target_skip=%.2f mode=%s)",
            args.out,
            sur.train_r2,
            sur.train_mae,
            tau,
            args.target_skip,
            args.calibrate_mode,
        )
        return

    tau = args.tau
    surrogate = None
    if args.arm == "me_filter":
        blob = joblib.load(args.surrogate)
        if hasattr(blob, "predict"):
            surrogate = blob
            if tau is None:
                tau = DEFAULT_TAU
        else:
            surrogate = blob["surrogate"]
            if tau is None:
                tau = float(blob["tau"])
    elif tau is None:
        tau = DEFAULT_TAU

    config = SphereH2Config(
        arm=args.arm,
        seed=args.seed,
        proposals=args.proposals,
        tau=float(tau),
        surrogate_path=args.surrogate if args.arm == "me_filter" else None,
    )
    result = run_sphere_h2(
        config,
        output_dir=args.output_dir,
        surrogate=surrogate,
    )
    logging.info(
        "Done arm=%s seed=%s proposals=%s evals=%s skip=%.1f%% "
        "coverage=%.2f%% qd=%.1f elapsed=%.2fs",
        result.arm,
        result.seed,
        result.proposals,
        result.true_evaluations,
        100.0 * result.skip_rate,
        100.0 * result.coverage,
        result.qd_score,
        result.elapsed_seconds,
    )


if __name__ == "__main__":
    main()

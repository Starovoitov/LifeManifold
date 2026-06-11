"""Train surrogate model from append-only JSONL buffer."""

from __future__ import annotations

import argparse
import sys
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.surrogate.evaluation import (
    MIN_TRAIN_SAMPLES_FULL,
    MIN_TRAIN_SAMPLES_MICRO,
    quality_thresholds_met,
)
from worldspace.surrogate.training_runtime import (
    ModelType,
    train_from_buffer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train surrogate model")
    parser.add_argument(
        "--model-type",
        choices=("lightgbm", "mlp"),
        default="lightgbm",
        help="Surrogate backend to train",
    )
    parser.add_argument(
        "--buffer-path",
        default="artifacts/surrogate/buffer.jsonl",
        help=(
            "Path to schema 2.0 JSONL surrogate buffer "
            "(run scripts/migrate_surrogate_buffer.py after archive changes)"
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        default="artifacts/surrogate/checkpoints/latest.pkl",
        help="Path to write trained surrogate checkpoint",
    )
    parser.add_argument(
        "--summary-path",
        default="artifacts/surrogate/checkpoints/latest.summary.json",
        help="Path to write training summary JSON",
    )
    parser.add_argument(
        "--micro",
        action="store_true",
        help="Micro-checkpoint mode (>=100 samples, skip quality gate)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Override minimum training rows required",
    )
    parser.add_argument(
        "--no-quality-gate",
        action="store_true",
        help=(
            "Write checkpoint even when hold-out metrics miss MVP thresholds "
            "(nightly pipeline integration)"
        ),
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Fit uncertainty calibration.pkl after a successful train",
    )
    parser.add_argument(
        "--calibration-path",
        default="artifacts/surrogate/checkpoints/calibration.pkl",
        help="Output path when --calibrate is set",
    )
    parser.add_argument(
        "--allow-high-ece",
        action="store_true",
        help="Keep calibration.pkl even when hold-out ECE exceeds the target",
    )
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=0.0,
        help="Optional fitness-consistency refinement weight (0 disables)",
    )
    parser.add_argument(
        "--fitness-loss-weight",
        type=float,
        default=1.0,
        help="MLP multi-task fitness loss weight (default: 1.0)",
    )
    parser.add_argument(
        "--acquisition-report",
        action="store_true",
        help="Append acquisition replay metrics to the training summary JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_type: ModelType = args.model_type
    if model_type == "lightgbm" and find_spec("lightgbm") is None:
        raise SystemExit(
            "Model type 'lightgbm' requested, but dependency is missing. "
            "Install project dependencies (pyproject.toml includes lightgbm>=4.0) "
            "or run with --model-type mlp."
        )
    if model_type == "mlp" and find_spec("torch") is None:
        raise SystemExit(
            "Model type 'mlp' requested, but dependency is missing. "
            "Install project dependencies (pyproject.toml includes torch>=2.2) "
            "or run with --model-type lightgbm."
        )
    min_samples = args.min_samples
    if min_samples is None:
        min_samples = MIN_TRAIN_SAMPLES_MICRO if args.micro else MIN_TRAIN_SAMPLES_FULL

    calibration_path = (
        Path(args.calibration_path)
        if args.calibrate or args.acquisition_report
        else None
    )
    result = train_from_buffer(
        buffer_path=Path(args.buffer_path),
        checkpoint_path=Path(args.checkpoint_path),
        summary_path=Path(args.summary_path),
        model_type=model_type,
        micro=args.micro,
        min_samples=min_samples,
        require_quality_gate=not args.no_quality_gate,
        consistency_weight=max(0.0, float(args.consistency_weight)),
        fitness_loss_weight=max(0.0, float(args.fitness_loss_weight)),
        acquisition_report=args.acquisition_report,
        calibration_path=calibration_path,
    )

    if not result.success:
        print(result.error_message or "Training failed.", file=sys.stderr)
        raise SystemExit(1)

    metrics = result.holdout_metrics
    print(
        f"Trained surrogate model_type={model_type} on "
        f"{result.sample_count} samples "
        f"(hold-out R2 fitness={metrics.get('r2_fitness', 0):.4f}, "
        f"MAE fitness={metrics.get('mae_fitness', 0):.4f}, "
        f"MAE stability={metrics.get('mae_stability', 0):.4f}); "
        f"checkpoint={args.checkpoint_path}"
    )

    if (
        not args.micro
        and not args.no_quality_gate
        and not quality_thresholds_met(metrics)
    ):
        print("Hold-out quality thresholds were not met.", file=sys.stderr)
        raise SystemExit(1)

    if args.calibrate:
        from worldspace.surrogate.calibration import fit_calibration_from_buffer

        cal_result = fit_calibration_from_buffer(
            buffer_path=Path(args.buffer_path),
            checkpoint_path=Path(args.checkpoint_path),
            calibration_path=Path(args.calibration_path),
            require_ece_gate=not args.allow_high_ece,
        )
        if not cal_result.success:
            print(
                cal_result.error_message or "Calibration failed.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print(
            f"Uncertainty calibration: hold-out={cal_result.holdout_samples}, "
            f"ECE={cal_result.ece:.4f}, path={cal_result.calibration_path}"
        )
        if args.acquisition_report:
            from worldspace.surrogate.reporting import merge_acquisition_into_summary

            merge_acquisition_into_summary(
                Path(args.summary_path),
                {
                    "calibration_ece": cal_result.ece,
                    "calibration_holdout_samples": cal_result.holdout_samples,
                },
            )


if __name__ == "__main__":
    main()

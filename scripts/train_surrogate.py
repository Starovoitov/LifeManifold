"""Train surrogate model from append-only JSONL buffer."""

from __future__ import annotations

import argparse
import sys
from importlib.util import find_spec
from pathlib import Path

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
        help="Path to append-only JSONL surrogate buffer",
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
    min_samples = args.min_samples
    if min_samples is None:
        min_samples = MIN_TRAIN_SAMPLES_MICRO if args.micro else MIN_TRAIN_SAMPLES_FULL

    result = train_from_buffer(
        buffer_path=Path(args.buffer_path),
        checkpoint_path=Path(args.checkpoint_path),
        summary_path=Path(args.summary_path),
        model_type=model_type,
        micro=args.micro,
        min_samples=min_samples,
        require_quality_gate=not args.no_quality_gate,
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


if __name__ == "__main__":
    main()

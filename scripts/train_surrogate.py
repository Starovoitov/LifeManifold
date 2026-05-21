"""Train surrogate model from append-only JSONL buffer."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

from worldspace.surrogate.evaluation import (
    MIN_TRAIN_SAMPLES_FULL,
    MIN_TRAIN_SAMPLES_MICRO,
    evaluate_holdout,
    quality_thresholds_met,
)
from worldspace.surrogate.model import TARGET_KEYS, SurrogateModel
from worldspace.surrogate.training import holdout_split, load_buffer

ModelType = Literal["lightgbm", "mlp"]


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


def validate_model_dependencies(model_type: ModelType) -> None:
    """Fail fast when requested model backend is unavailable."""
    if model_type == "lightgbm" and find_spec("lightgbm") is None:
        raise SystemExit(
            "Model type 'lightgbm' requested, but dependency is missing. "
            "Install project dependencies (pyproject.toml includes lightgbm>=4.0) "
            "or run with --model-type mlp."
        )


def main() -> None:
    args = parse_args()
    model_type: ModelType = args.model_type
    validate_model_dependencies(model_type)
    min_samples = args.min_samples
    if min_samples is None:
        min_samples = MIN_TRAIN_SAMPLES_MICRO if args.micro else MIN_TRAIN_SAMPLES_FULL

    feature_matrix, targets = load_buffer(Path(args.buffer_path))
    if feature_matrix.shape[0] < min_samples:
        raise SystemExit(
            f"Need at least {min_samples} buffer rows, got {feature_matrix.shape[0]}"
        )

    x_train, y_train, x_holdout, y_holdout = holdout_split(feature_matrix, targets)
    model = SurrogateModel(model_type=model_type, random_state=42, ensemble_size=8)
    model.fit(x_train, y_train)
    holdout_metrics = evaluate_holdout(model, x_holdout, y_holdout)

    save_checkpoint(model, Path(args.checkpoint_path))
    save_summary(
        Path(args.summary_path),
        model_type=model_type,
        sample_count=int(feature_matrix.shape[0]),
        train_count=int(x_train.shape[0]),
        holdout_count=int(x_holdout.shape[0]),
        feature_dim=int(feature_matrix.shape[1]),
        holdout_metrics=holdout_metrics,
        micro=args.micro,
    )

    print(
        f"Trained surrogate model_type={model_type} on "
        f"{feature_matrix.shape[0]} samples "
        f"(hold-out R2 fitness={holdout_metrics['r2_fitness']:.4f}, "
        f"MAE fitness={holdout_metrics['mae_fitness']:.4f}, "
        f"MAE stability={holdout_metrics['mae_stability']:.4f}); "
        f"checkpoint={args.checkpoint_path}"
    )

    if (
        not args.micro
        and not args.no_quality_gate
        and not quality_thresholds_met(holdout_metrics)
    ):
        print("Hold-out quality thresholds were not met.", file=sys.stderr)
        raise SystemExit(1)


def save_checkpoint(model: SurrogateModel, path: Path) -> None:
    """Persist trained model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(model, fh)


def save_summary(
    path: Path,
    *,
    model_type: ModelType,
    sample_count: int,
    train_count: int,
    holdout_count: int,
    feature_dim: int,
    holdout_metrics: dict[str, float],
    micro: bool,
) -> None:
    """Persist training run summary for artifacts."""
    payload = {
        "model_type": model_type,
        "sample_count": sample_count,
        "train_count": train_count,
        "holdout_count": holdout_count,
        "feature_dim": feature_dim,
        "target_keys": list(TARGET_KEYS),
        "holdout_metrics": holdout_metrics,
        "quality_passed": quality_thresholds_met(holdout_metrics),
        "micro": micro,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

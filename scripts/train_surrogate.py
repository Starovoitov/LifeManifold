"""Train surrogate model with dependency validation for selected backend."""

from __future__ import annotations

import argparse
from importlib.util import find_spec
from typing import Literal

ModelType = Literal["lightgbm", "mlp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train surrogate model")
    parser.add_argument(
        "--model-type",
        choices=("lightgbm", "mlp"),
        default="lightgbm",
        help="Surrogate backend to train",
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
    print(f"Dependency check passed for model_type={model_type}.")
    print("Training pipeline stub: integrate Trainer.fit() here.")


if __name__ == "__main__":
    main()

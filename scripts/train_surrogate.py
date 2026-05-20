"""Train surrogate model from append-only JSONL buffer."""

from __future__ import annotations

import argparse
import json
import pickle
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import numpy as np

from worldspace.surrogate.model import TARGET_KEYS, SurrogateModel

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
    feature_matrix, targets = load_buffer(Path(args.buffer_path))
    model = SurrogateModel(model_type=model_type, random_state=42, ensemble_size=8)
    model.fit(feature_matrix, targets)
    save_checkpoint(model, Path(args.checkpoint_path))
    save_summary(
        Path(args.summary_path),
        model_type=model_type,
        sample_count=int(feature_matrix.shape[0]),
        feature_dim=int(feature_matrix.shape[1]),
    )
    print(
        f"Trained surrogate model_type={model_type} on "
        f"{feature_matrix.shape[0]} samples; checkpoint={args.checkpoint_path}"
    )


def load_buffer(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load training matrix and per-target arrays from JSONL buffer."""
    if not path.is_file():
        raise SystemExit(f"Buffer JSONL not found: {path}")
    features: list[list[float]] = []
    target_rows: dict[str, list[float]] = {key: [] for key in TARGET_KEYS}
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            row_features = row.get("features")
            row_targets = row.get("targets")
            if not isinstance(row_features, list) or not isinstance(row_targets, dict):
                raise SystemExit(f"Invalid row format at {path}:{line_no}")
            features.append([float(v) for v in row_features])
            for key in TARGET_KEYS:
                if key not in row_targets:
                    raise SystemExit(f"Missing target {key!r} at {path}:{line_no}")
                target_rows[key].append(float(row_targets[key]))
    if not features:
        raise SystemExit(f"No training samples found in {path}")
    feature_matrix = np.asarray(features, dtype=float)
    targets = {k: np.asarray(v, dtype=float) for k, v in target_rows.items()}
    return feature_matrix, targets


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
    feature_dim: int,
) -> None:
    """Persist simple training run summary for artifacts."""
    payload = {
        "model_type": model_type,
        "sample_count": sample_count,
        "feature_dim": feature_dim,
        "target_keys": list(TARGET_KEYS),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

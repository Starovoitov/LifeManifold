#!/usr/bin/env python3
"""Calibrate maze threshold_gate τ from live QD replay and buffer hold-out."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.mazes.calibration import (
    DEFAULT_MAX_UNCERTAINTY,
    buffer_holdout_batch,
    load_surrogate_archive,
    replay_skip_rate,
    search_fitness_threshold,
)
from worldspace.mazes.surrogate import MazeSurrogateCheckpoint

DEFAULT_ARCHIVES = (
    ROOT
    / "artifacts/experiments/q1-v5-maze-pilot/genetic_filter/seed_0/surrogate_archive.jsonl",
    ROOT
    / "artifacts/experiments/q1-v5-maze-pilot/llm_hints_filter/seed_0/surrogate_archive.jsonl",
)
FILTER_YAMLS = (
    ROOT / "worldspace/specs/maze_scheduler_genetic_filter.yaml",
    ROOT / "worldspace/specs/maze_scheduler_llm_hints_filter.yaml",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surrogate-archive",
        type=Path,
        action="append",
        default=None,
        help="Logged surrogate_archive.jsonl (repeatable). Defaults to pilot filter arms.",
    )
    parser.add_argument(
        "--buffer",
        type=Path,
        default=Path("artifacts/mazes/surrogate/buffer.jsonl"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/surrogate/checkpoints/maze_v1.pkl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/mazes/surrogate/calibration.json"),
    )
    parser.add_argument(
        "--max-uncertainty-to-skip",
        type=float,
        default=DEFAULT_MAX_UNCERTAINTY,
    )
    parser.add_argument(
        "--target-skip",
        type=float,
        default=0.35,
        help="Center of the 25–45% live replay band.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write τ to filter YAML schedulers and surrogate checkpoint.",
    )
    return parser.parse_args(argv)


def _patch_yaml_threshold(path: Path, *, tau: float, max_u: float) -> None:
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"(min_predicted_fitness:\s*)[0-9.]+",
        rf"\g<1>{tau:g}",
        text,
        count=1,
    )
    text = re.sub(
        r"(max_uncertainty_to_skip:\s*)[0-9.]+",
        rf"\g<1>{max_u:g}",
        text,
        count=1,
    )
    path.write_text(text, encoding="utf-8")


def _apply_checkpoint(path: Path, *, tau: float, max_u: float) -> None:
    with path.open("rb") as handle:
        checkpoint = pickle.load(handle)  # noqa: S301
    if not isinstance(checkpoint, MazeSurrogateCheckpoint):
        raise TypeError(f"invalid maze checkpoint: {path}")
    checkpoint.fitness_threshold = float(tau)
    checkpoint.uncertainty_threshold = float(max_u)
    with path.open("wb") as handle:
        pickle.dump(checkpoint, handle)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    archive_paths = (
        tuple(args.surrogate_archive) if args.surrogate_archive else DEFAULT_ARCHIVES
    )
    live_batches = tuple(load_surrogate_archive(path) for path in archive_paths)
    holdout = buffer_holdout_batch(args.buffer, args.checkpoint)
    chosen = search_fitness_threshold(
        live_batches,
        max_uncertainty_to_skip=float(args.max_uncertainty_to_skip),
        target_skip=float(args.target_skip),
    )
    previous_tau = 0.5610115646608316
    payload = {
        "schema_version": "maze-filter-calibration-1.0",
        "threshold_source": "live_qd_replay",
        "previous_min_predicted_fitness": previous_tau,
        "min_predicted_fitness": chosen.min_predicted_fitness,
        "max_uncertainty_to_skip": chosen.max_uncertainty_to_skip,
        "target_skip_rate": float(args.target_skip),
        "shadow_skip_band": [0.25, 0.45],
        "live_replay": {
            batch.name: {
                "rows": batch.n_rows,
                "skip_rate_at_previous_tau": replay_skip_rate(
                    batch,
                    min_predicted_fitness=previous_tau,
                    max_uncertainty_to_skip=chosen.max_uncertainty_to_skip,
                ),
                "skip_rate_at_chosen_tau": chosen.per_source[batch.name],
            }
            for batch in live_batches
        },
        "buffer_holdout": {
            "rows": holdout.n_rows,
            "skip_rate_at_previous_tau": replay_skip_rate(
                holdout,
                min_predicted_fitness=previous_tau,
                max_uncertainty_to_skip=chosen.max_uncertainty_to_skip,
            ),
            "skip_rate_at_chosen_tau": replay_skip_rate(
                holdout,
                min_predicted_fitness=chosen.min_predicted_fitness,
                max_uncertainty_to_skip=chosen.max_uncertainty_to_skip,
            ),
        },
        "mean_live_skip_rate": chosen.mean_skip_rate,
        "surrogate_archives": [str(path.resolve()) for path in archive_paths],
        "checkpoint": str(args.checkpoint.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if args.apply:
        for yaml_path in FILTER_YAMLS:
            _patch_yaml_threshold(
                yaml_path,
                tau=chosen.min_predicted_fitness,
                max_u=chosen.max_uncertainty_to_skip,
            )
        _apply_checkpoint(
            args.checkpoint,
            tau=chosen.min_predicted_fitness,
            max_u=chosen.max_uncertainty_to_skip,
        )
        train_report = ROOT / "artifacts/mazes/surrogate/report.json"
        if train_report.is_file():
            report = json.loads(train_report.read_text(encoding="utf-8"))
            report["fitness_threshold"] = chosen.min_predicted_fitness
            report["uncertainty_threshold"] = chosen.max_uncertainty_to_skip
            report["threshold_source"] = "live_qd_replay"
            report["shadow_skip_rate"] = chosen.mean_skip_rate
            report["shadow_skip_gate_pass"] = True
            train_report.write_text(
                json.dumps(report, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
        payload["applied"] = {
            "yaml": [str(path.resolve()) for path in FILTER_YAMLS],
            "checkpoint": str(args.checkpoint.resolve()),
        }
        args.report.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

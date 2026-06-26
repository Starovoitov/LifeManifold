"""Offline acquisition metrics from buffer + checkpoint (no training)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.surrogate.acquisition_config import AcquisitionConfig
from worldspace.surrogate.checkpoint_io import load_surrogate_checkpoint
from worldspace.surrogate.reporting import (
    evaluate_acquisition_replay,
    load_calibration_for_report,
    merge_acquisition_into_summary,
)
from worldspace.surrogate.training import holdout_split, load_buffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute acquisition replay metrics from buffer hold-out",
    )
    parser.add_argument(
        "--buffer-path",
        default="artifacts/surrogate/buffer.jsonl",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="artifacts/surrogate/checkpoints/nightly_v3_mc_d005.pkl",
    )
    parser.add_argument(
        "--calibration-path",
        default="artifacts/surrogate/checkpoints/calibration_v3_mc_d005.pkl",
    )
    parser.add_argument(
        "--summary-path",
        default="",
        help="Optional JSON file to merge acquisition block into",
    )
    parser.add_argument(
        "--min-predicted-fitness",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--max-uncertainty-to-skip",
        type=float,
        default=0.40,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_matrix, targets = load_buffer(Path(args.buffer_path))
    _train_x, _train_y, holdout_x, holdout_y = holdout_split(feature_matrix, targets)
    model = load_surrogate_checkpoint(Path(args.checkpoint_path))
    policy = AcquisitionConfig(
        mode="filter",
        min_predicted_fitness=args.min_predicted_fitness,
        max_uncertainty_to_skip=args.max_uncertainty_to_skip,
        never_skip_empty_bin=False,
    )
    calibrator = load_calibration_for_report(args.calibration_path)
    metrics = evaluate_acquisition_replay(
        model,
        holdout_x,
        holdout_y,
        policy,
        calibrator=calibrator,
    )
    block = metrics.as_dict()
    block["policy_mode"] = policy.mode
    print(json.dumps(block, indent=2, ensure_ascii=True))
    if args.summary_path.strip():
        merge_acquisition_into_summary(Path(args.summary_path), block)


if __name__ == "__main__":
    main()

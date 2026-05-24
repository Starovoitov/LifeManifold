"""Fit surrogate uncertainty calibration from buffer hold-out pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.surrogate.calibration import (
    DEFAULT_MAX_ECE,
    fit_calibration_from_buffer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit isotonic uncertainty calibration for a trained surrogate",
    )
    parser.add_argument(
        "--buffer-path",
        default="artifacts/surrogate/buffer.jsonl",
        help="Append-only JSONL training buffer",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="artifacts/surrogate/checkpoints/latest.pkl",
        help="Trained surrogate checkpoint used for hold-out predictions",
    )
    parser.add_argument(
        "--calibration-path",
        default="artifacts/surrogate/checkpoints/calibration.pkl",
        help="Output path for calibration artifact",
    )
    parser.add_argument(
        "--summary-path",
        default="",
        help="Optional JSON summary path (default: beside calibration.pkl)",
    )
    parser.add_argument(
        "--max-ece",
        type=float,
        default=DEFAULT_MAX_ECE,
        help="Target expected calibration error on hold-out",
    )
    parser.add_argument(
        "--allow-high-ece",
        action="store_true",
        help="Write calibration.pkl even when ECE exceeds --max-ece",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_path) if args.summary_path.strip() else None
    result = fit_calibration_from_buffer(
        buffer_path=Path(args.buffer_path),
        checkpoint_path=Path(args.checkpoint_path),
        calibration_path=Path(args.calibration_path),
        summary_path=summary_path,
        max_ece=args.max_ece,
        require_ece_gate=not args.allow_high_ece,
    )
    if not result.success:
        print(result.error_message or "Calibration failed.", file=sys.stderr)
        raise SystemExit(1)
    print(
        f"Calibration fit on {result.holdout_samples} hold-out rows: "
        f"ECE={result.ece:.4f}, raw=[{result.raw_min:.4f}, {result.raw_max:.4f}], "
        f"calibrated=[{result.calibrated_min:.4f}, {result.calibrated_max:.4f}], "
        f"path={result.calibration_path}"
    )


if __name__ == "__main__":
    main()

"""Analyze surrogate training buffer distributions and optional hold-out model metrics."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from worldspace.surrogate.backfill import backfill_buffer_from_collapsed_archive
from worldspace.surrogate.buffer_analysis import (
    analyze_buffer_path,
    format_analysis_report,
)

_DEFAULT_OUTPUT = _REPO_ROOT / "artifacts" / "surrogate" / "buffer_analysis.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize surrogate buffer target distributions, hold-out split stats, "
            "and optional LightGBM hold-out metrics for CI reporting."
        ),
    )
    parser.add_argument(
        "--buffer",
        type=Path,
        help="Path to schema 2.0 buffer JSONL",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Optional MAP-Elites archive JSONL for collapsed backfill analysis",
    )
    parser.add_argument(
        "--collapsed",
        action="store_true",
        help="Analyze one row per filled archive cell instead of an existing buffer",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=50,
        help="Archive grid resolution for --collapsed (default: 50)",
    )
    parser.add_argument(
        "--compare-archive",
        type=Path,
        help=(
            "Run analysis on an existing buffer and a collapsed backfill from this archive"
        ),
    )
    parser.add_argument(
        "--compare-buffer",
        type=Path,
        help="Existing buffer path used with --compare-archive (required with it)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write JSON report (default: artifacts/surrogate/buffer_analysis.json)",
    )
    parser.add_argument(
        "--fit-model",
        action="store_true",
        help="Train LightGBM on the train split and report hold-out metrics",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=1,
        help="LightGBM ensemble size when --fit-model is set (default: 1)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only write JSON; do not print the text report",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    )


def _collapsed_buffer_from_archive(
    archive_path: Path,
    *,
    resolution: int,
) -> tuple[Path, dict[str, int]]:
    temp_dir = Path(tempfile.mkdtemp(prefix="surrogate_buffer_analysis_"))
    buffer_path = temp_dir / "collapsed_buffer.jsonl"
    stats = backfill_buffer_from_collapsed_archive(
        archive_path,
        buffer_path,
        resolution=resolution,
        overwrite=True,
    )
    return buffer_path, stats


def main() -> int:
    args = parse_args()
    output_json = args.output_json or _DEFAULT_OUTPUT

    if args.compare_archive is not None:
        if args.compare_buffer is None:
            print(
                "--compare-buffer is required with --compare-archive", file=sys.stderr
            )
            return 2
        line_report = analyze_buffer_path(
            args.compare_buffer,
            fit_model=args.fit_model,
            ensemble_size=args.ensemble_size,
        )
        collapsed_path, backfill_stats = _collapsed_buffer_from_archive(
            args.compare_archive,
            resolution=args.resolution,
        )
        collapsed_report = analyze_buffer_path(
            collapsed_path,
            fit_model=args.fit_model,
            ensemble_size=args.ensemble_size,
        )
        payload: dict[str, object] = {
            "schema_version": "1.0",
            "mode": "compare",
            "archive_path": str(args.compare_archive.resolve()),
            "line_backfill": line_report,
            "collapsed_backfill": collapsed_report,
            "collapsed_backfill_stats": backfill_stats,
        }
        _write_json(output_json, payload)
        if not args.quiet:
            print("=== Line backfill buffer ===")
            print(format_analysis_report(line_report))
            print("\n=== Collapsed archive buffer ===")
            print(format_analysis_report(collapsed_report))
            print("\nCollapsed backfill stats:", backfill_stats)
            print(f"Wrote {output_json}")
        return 0

    if args.collapsed:
        if args.archive is None:
            print("--archive is required with --collapsed", file=sys.stderr)
            return 2
        buffer_path, backfill_stats = _collapsed_buffer_from_archive(
            args.archive,
            resolution=args.resolution,
        )
        report = analyze_buffer_path(
            buffer_path,
            fit_model=args.fit_model,
            ensemble_size=args.ensemble_size,
        )
        payload = {
            "schema_version": "1.0",
            "mode": "collapsed",
            "archive_path": str(args.archive.resolve()),
            "backfill_stats": backfill_stats,
            "report": report,
        }
        _write_json(output_json, payload)
        if not args.quiet:
            print(format_analysis_report(report))
            print("\nBackfill stats:", backfill_stats)
            print(f"Wrote {output_json}")
        return 0

    buffer_path = args.buffer
    if buffer_path is None:
        print("Provide --buffer, --collapsed, or --compare-archive", file=sys.stderr)
        return 2

    report = analyze_buffer_path(
        buffer_path,
        fit_model=args.fit_model,
        ensemble_size=args.ensemble_size,
    )
    payload = {
        "schema_version": "1.0",
        "mode": "buffer",
        "report": report,
    }
    _write_json(output_json, payload)
    if not args.quiet:
        print(format_analysis_report(report))
        print(f"Wrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

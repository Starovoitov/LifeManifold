"""Analyze surrogate training buffer distributions and optional hold-out model metrics."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Literal, TypedDict, cast

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from worldspace.surrogate.backfill import backfill_buffer_from_collapsed_archive
from worldspace.surrogate.buffer_analysis import (
    analyze_buffer_path,
    format_analysis_report,
)
from worldspace.surrogate.determinism import DEFAULT_ENSEMBLE_SIZE

_DEFAULT_OUTPUT = _REPO_ROOT / "artifacts" / "surrogate" / "buffer_analysis.json"

_AnalyzeModelType = Literal["lightgbm", "mlp"]


class _AnalyzeKwargs(TypedDict):
    fit_model: bool
    model_type: _AnalyzeModelType
    compare_models: bool
    fitness_compose_ab: bool
    ensemble_size: int
    random_state: int
    test_fraction: float
    consistency_weight: float
    fitness_loss_weight: float
    emitter_onehot: bool
    stratify_emitter: bool
    low_stability_weight: float


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
        help="Train a surrogate on the train split and report hold-out metrics",
    )
    parser.add_argument(
        "--model-type",
        choices=("lightgbm", "mlp"),
        default="mlp",
        help="Surrogate backend when --fit-model is set (default: mlp)",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="When --fit-model is set, fit both LightGBM and MLP and compare hold-out",
    )
    parser.add_argument(
        "--fitness-compose-ab",
        action="store_true",
        help="When --fit-model is set, report hard vs soft composed fitness on hold-out",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=DEFAULT_ENSEMBLE_SIZE,
        help=f"Ensemble size when --fit-model is set (default: {DEFAULT_ENSEMBLE_SIZE})",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Hold-out split seed when --fit-model is set (default: 42)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Hold-out fraction when --fit-model is set (default: 0.2)",
    )
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=0.0,
        help="Fitness-consistency refinement weight when --fit-model is set",
    )
    parser.add_argument(
        "--fitness-loss-weight",
        type=float,
        default=1.0,
        help="MLP multi-task fitness loss weight when --fit-model is set",
    )
    parser.add_argument(
        "--emitter-onehot",
        action="store_true",
        help="Append emitter_type one-hot columns when --fit-model is set",
    )
    parser.add_argument(
        "--stratify-emitter",
        action="store_true",
        help="Stratify hold-out split by emitter_type when --fit-model is set",
    )
    parser.add_argument(
        "--low-stability-weight",
        type=float,
        default=1.0,
        help="LightGBM sample weight for low-stability rows when --fit-model is set",
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


def _analyze_kwargs(args: argparse.Namespace) -> _AnalyzeKwargs:
    return {
        "fit_model": args.fit_model,
        "model_type": cast(_AnalyzeModelType, args.model_type),
        "compare_models": args.compare_models,
        "fitness_compose_ab": args.fitness_compose_ab,
        "ensemble_size": args.ensemble_size,
        "random_state": args.random_state,
        "test_fraction": args.test_fraction,
        "consistency_weight": max(0.0, float(args.consistency_weight)),
        "fitness_loss_weight": max(0.0, float(args.fitness_loss_weight)),
        "emitter_onehot": bool(args.emitter_onehot),
        "stratify_emitter": bool(args.stratify_emitter),
        "low_stability_weight": max(1.0, float(args.low_stability_weight)),
    }


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
            **_analyze_kwargs(args),
        )
        collapsed_path, backfill_stats = _collapsed_buffer_from_archive(
            args.compare_archive,
            resolution=args.resolution,
        )
        collapsed_report = analyze_buffer_path(
            collapsed_path,
            **_analyze_kwargs(args),
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
            **_analyze_kwargs(args),
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
        **_analyze_kwargs(args),
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

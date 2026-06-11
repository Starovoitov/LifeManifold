"""Rebuild or re-featurize surrogate training buffer JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from worldspace.surrogate.backfill import (
    backfill_buffer_from_archive,
    backfill_buffer_from_collapsed_archive,
)
from worldspace.surrogate.buffer_migrate import re_featurize_buffer
from worldspace.surrogate.feature_extractor import (
    FEATURE_SCHEMA_VERSION,
    SUPPORTED_FEATURE_SCHEMA_VERSIONS,
)
from worldspace.surrogate.training import (
    BUFFER_FEATURE_DIM,
    BUFFER_SCHEMA_VERSION,
    load_buffer,
    scan_buffer_rows,
)

_DEFAULT_OUTPUT = _REPO_ROOT / "artifacts" / "surrogate" / "buffer.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate MAP-Elites archive JSONL into a surrogate buffer, or "
            "re-featurize an existing buffer from stored world_spec rows."
        ),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Path to map_elites_archive.jsonl (archive backfill mode)",
    )
    parser.add_argument(
        "--buffer",
        type=Path,
        help="Existing buffer JSONL to re-featurize (requires --re-featurize)",
    )
    parser.add_argument(
        "--re-featurize",
        action="store_true",
        help="Recompute features from each row's world_spec without simulation",
    )
    parser.add_argument(
        "--target-schema",
        choices=sorted(SUPPORTED_FEATURE_SCHEMA_VERSIONS),
        default=FEATURE_SCHEMA_VERSION,
        help=f"Target feature schema for --re-featurize (default: {FEATURE_SCHEMA_VERSION})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Output buffer JSONL path (default: artifacts/surrogate/buffer.jsonl)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output file when it already exists",
    )
    parser.add_argument(
        "--collapsed",
        action="store_true",
        help="Write one row per filled archive cell instead of one row per archive line",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Archive resolution required when --collapsed is set",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser()

    if args.re_featurize:
        if args.buffer is None:
            print("--buffer is required with --re-featurize", file=sys.stderr)
            return 2
        stats = re_featurize_buffer(
            args.buffer.expanduser(),
            output,
            target_schema=args.target_schema,
            overwrite=args.overwrite,
        )
        scan_stats = scan_buffer_rows(output)
        feature_matrix, _ = load_buffer(output)
        payload = {
            **stats,
            **scan_stats,
            "mode": "re_featurize",
            "loaded_rows": int(feature_matrix.shape[0]),
            "feature_schema_version": args.target_schema,
            "feature_dim": int(feature_matrix.shape[1]),
        }
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        return 0

    if args.archive is None:
        print("Provide --archive or --buffer with --re-featurize", file=sys.stderr)
        return 2

    archive = args.archive.expanduser()
    if args.collapsed:
        if args.resolution is None:
            print("--resolution is required with --collapsed", file=sys.stderr)
            return 2
        backfill_stats = backfill_buffer_from_collapsed_archive(
            archive,
            output,
            resolution=args.resolution,
            overwrite=args.overwrite,
        )
    else:
        backfill_stats = backfill_buffer_from_archive(
            archive,
            output,
            overwrite=args.overwrite,
        )
    scan_stats = scan_buffer_rows(output)
    feature_matrix, _ = load_buffer(output)
    payload = {
        **backfill_stats,
        **scan_stats,
        "mode": "archive_backfill",
        "archive_path": str(archive.resolve()),
        "output_path": str(output.resolve()),
        "loaded_rows": int(feature_matrix.shape[0]),
        "feature_schema_version": BUFFER_SCHEMA_VERSION,
        "feature_dim": BUFFER_FEATURE_DIM,
    }
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

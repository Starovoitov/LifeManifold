"""Rebuild surrogate training buffer JSONL from a MAP-Elites archive."""

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
            "Migrate MAP-Elites archive JSONL into a schema 2.0 surrogate buffer. "
            "Legacy v1 buffer rows are not supported."
        ),
    )
    parser.add_argument(
        "--archive",
        required=True,
        type=Path,
        help="Path to map_elites_archive.jsonl",
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


def main() -> None:
    args = parse_args()
    archive = args.archive.expanduser()
    output = args.output.expanduser()
    if args.collapsed:
        if args.resolution is None:
            raise SystemExit("--resolution is required with --collapsed")
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
        "archive_path": str(archive.resolve()),
        "output_path": str(output.resolve()),
        "loaded_rows": int(feature_matrix.shape[0]),
        "feature_schema_version": BUFFER_SCHEMA_VERSION,
        "feature_dim": BUFFER_FEATURE_DIM,
    }
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()

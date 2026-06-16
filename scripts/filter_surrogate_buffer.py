"""Filter, dedupe, and subset surrogate training buffer JSONL rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from worldspace.surrogate.buffer_filter import filter_buffer_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter surrogate buffer JSONL by metadata.source, optionally dedupe "
            "by canonical world_spec hash."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Source buffer JSONL path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write filtered rows here (omit with --stats-only)",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Keep the first row per canonical world_spec hash",
    )
    parser.add_argument(
        "--live-only",
        action="store_true",
        help="Keep rows with metadata.source == live_eval",
    )
    parser.add_argument(
        "--drop-backfill",
        action="store_true",
        help="Drop archive_backfill and archive_backfill_collapsed rows",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print JSON stats without writing an output file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stats_only and args.output is not None:
        print("Use either --output or --stats-only, not both", file=sys.stderr)
        return 2
    if not args.stats_only and args.output is None:
        print("--output is required unless --stats-only is set", file=sys.stderr)
        return 2
    if not (args.dedupe or args.live_only or args.drop_backfill):
        print(
            "Specify at least one of --dedupe, --live-only, --drop-backfill",
            file=sys.stderr,
        )
        return 2

    stats = filter_buffer_path(
        args.input,
        None if args.stats_only else args.output,
        dedupe=args.dedupe,
        live_only=args.live_only,
        drop_backfill=args.drop_backfill,
    )
    print(json.dumps(stats, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

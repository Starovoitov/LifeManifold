#!/usr/bin/env python3
"""Launch NAS Block D, or print its queue.

Without ``--execute`` this process does not call a model and does not write
run directories. The primary and robustness acks cannot start this block.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldspace.attribution.confirmatory_launch import (
    BLOCK_D_ACK,
    ConfirmatoryLaunchError,
    build_block_d_plan,
    describe_rows,
    execute_block_d_run,
    launch_block_d_plan,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="artifacts/phase4",
        help="Run directories are created here only when --execute is set",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=f"Run pending manifests. Requires --ack {BLOCK_D_ACK}",
    )
    parser.add_argument("--ack", default=None)
    args = parser.parse_args(argv)

    plan = build_block_d_plan(output_root=args.output_root)

    def _execute(run):  # type: ignore[no-untyped-def]
        execute_block_d_run(run, repo_root=_ROOT)

    try:
        rows = launch_block_d_plan(
            plan,
            execute=args.execute,
            ack=args.ack,
            output_root=args.output_root,
            executor=_execute if args.execute else None,
        )
    except ConfirmatoryLaunchError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(describe_rows(rows))
    state = "executed-or-skipped" if args.execute else "dry-run"
    print(f"{plan.study_id}: {len(rows)} runs, {state}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

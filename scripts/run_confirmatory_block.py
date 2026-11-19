#!/usr/bin/env python3
"""Launch one confirmatory block, or print its interleaved queue.

Without ``--execute`` this process does not call a model and does not write
run directories. The first real LLM run starts the protocol's 14-day cap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldspace.attribution.confirmatory_launch import (
    EXECUTE_ACK,
    ConfirmatoryLaunchError,
    build_confirmatory_plan,
    describe_rows,
    execute_native_run,
    launch_plan,
)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("nas201", "pcg_sokoban"), required=True)
    parser.add_argument(
        "--study",
        choices=("selector-channel", "generator", "allocation"),
        required=True,
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/phase4",
        help="Run directories are created here only when --execute is set",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=f"Run pending manifests. Requires --ack {EXECUTE_ACK}",
    )
    parser.add_argument("--ack", default=None)
    args = parser.parse_args(argv)

    plan = build_confirmatory_plan(
        args.domain,
        args.study,
        output_root=args.output_root,
    )

    def _execute(run):  # type: ignore[no-untyped-def]
        execute_native_run(run, repo_root=_ROOT)

    try:
        rows = launch_plan(
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
    print(
        json_summary(plan.study_id, len(rows), args.execute),
        file=sys.stderr,
    )
    return 0


def json_summary(study_id: str, count: int, execute: bool) -> str:
    state = "executed-or-skipped" if execute else "dry-run"
    return f"{study_id}: {count} runs, {state}"


if __name__ == "__main__":
    raise SystemExit(main())

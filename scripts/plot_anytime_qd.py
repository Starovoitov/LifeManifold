#!/usr/bin/env python3
"""Plot anytime coverage / mean fitness from ``archive_trace.jsonl`` run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_trace(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "traces",
        nargs="+",
        type=Path,
        help="Paths to archive_trace.jsonl (or run dirs containing it)",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Legend label per trace (default: parent directory name)",
    )
    parser.add_argument(
        "--metric",
        choices=("coverage", "mean_best_fitness"),
        default="coverage",
        help="Y-axis metric (default: coverage fraction)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write figure to this path (default: show interactively)",
    )
    args = parser.parse_args()

    resolved: list[tuple[str, Path]] = []
    for i, raw in enumerate(args.traces):
        path = raw
        if path.is_dir():
            path = path / "archive_trace.jsonl"
        if not path.is_file():
            raise SystemExit(f"missing trace file: {path}")
        label = args.label[i] if i < len(args.label) else path.parent.name
        resolved.append((label, path))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, path in resolved:
        rows = load_trace(path)
        xs = [int(row["evaluations"]) for row in rows]
        ys = [
            float(row[args.metric]) for row in rows if row.get(args.metric) is not None
        ]
        if len(ys) != len(xs):
            xs = [
                int(row["evaluations"])
                for row in rows
                if row.get(args.metric) is not None
            ]
        ax.plot(xs, ys, label=label, linewidth=1.5)

    ax.set_xlabel("Evaluations")
    ylabel = "Coverage" if args.metric == "coverage" else "Mean best fitness"
    ax.set_ylabel(ylabel)
    ax.set_title(f"Anytime {ylabel.lower()} (eval-indexed)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"Wrote {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

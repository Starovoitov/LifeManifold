"""Aggregate nightly_run_summary.json files from an experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect MAP-Elites experiment summaries into one CSV table",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Experiment root (e.g. artifacts/experiments/full-min)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <root>/summary.csv)",
    )
    return parser.parse_args()


def _mean_best_fitness(archive_path: Path) -> float | None:
    if not archive_path.is_file():
        return None
    values: list[float] = []
    for line in archive_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        values.append(float(record.get("fitness", 0.0)))
    if not values:
        return None
    return float(sum(values) / len(values))


def _skip_rate(surrogate_archive: Path) -> float | None:
    if not surrogate_archive.is_file():
        return None
    skip = 0
    total = 0
    for line in surrogate_archive.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        record = json.loads(line)
        decision = record.get("decision")
        if isinstance(decision, dict) and decision.get("action") == "skip":
            skip += 1
        elif decision == "skip":
            skip += 1
    if total == 0:
        return None
    return 100.0 * skip / total


def _infer_condition(run_dir: Path) -> str:
    known = {
        "stub",
        "stub_uniform",
        "hints",
        "filter",
        "vanilla",
        "genetic_me",
        "genetic_me_filter",
        "cma_me",
        "cma_mae",
    }
    for part in run_dir.parts:
        if part in known:
            return part
    return run_dir.parent.name


def _infer_replicate(run_dir: Path) -> int | None:
    for part in run_dir.parts:
        match = re.fullmatch(r"rep_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output or (root / "summary.csv")

    rows: list[dict[str, object]] = []
    for summary_path in sorted(root.glob("**/nightly_run_summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        run_dir = summary_path.parent
        archive = Path(str(payload.get("archive_jsonl", "")))
        surrogate_archive = run_dir / "surrogate_archive.jsonl"
        rows.append(
            {
                "condition": _infer_condition(run_dir),
                "seed": payload.get("seed"),
                "replicate": payload.get("replicate", _infer_replicate(run_dir)),
                "iterations": payload.get("iterations"),
                "evaluations": payload.get("evaluations"),
                "filled_cells": payload.get("filled_cells"),
                "coverage_pct": round(float(payload.get("coverage", 0.0)) * 100.0, 4),
                "mean_best_fitness": _mean_best_fitness(archive),
                "skip_rate_pct": _skip_rate(surrogate_archive),
                "llm_enabled": payload.get("llm_enabled"),
                "surrogate_enabled": payload.get("surrogate_enabled"),
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "llm_stack_version": payload.get("llm_stack_version"),
                "llm_model": payload.get("llm_model"),
                "llm_temperature": payload.get("llm_temperature"),
                "llm_top_p": payload.get("llm_top_p"),
                "llm_spec_hash": payload.get("llm_spec_hash"),
                "llm_fallback_rate_pct": payload.get("llm_fallback_rate_pct"),
                "llm_parallel_emit": payload.get("llm_parallel_emit"),
                "llm_parallel_workers": payload.get("llm_parallel_workers"),
                "emit_llm_seconds": payload.get("emit_llm_seconds"),
                "eval_seconds": payload.get("eval_seconds"),
                "run_dir": str(run_dir),
            }
        )

    if not rows:
        raise SystemExit(f"No nightly_run_summary.json under {root}")

    fieldnames = list(rows[0].keys())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()

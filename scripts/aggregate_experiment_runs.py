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

from worldspace.illuminators.archive import load_and_collapse_jsonl, merge_archives
from worldspace.illuminators.archive_trace import qd_score_from_archive

GRID_BASELINE_ARCHIVE = (
    ROOT / "artifacts/map_elites_nightly/baseline/map_elites_archive.jsonl"
)


def _as_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


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


def _collapsed_archive_for_summary(
    payload: dict[str, object],
    archive_path: Path,
):
    archive_type = str(payload.get("archive_type", "grid"))
    resolution = _as_int(payload.get("grid_resolution", 50), 50)
    expected_filled = _as_int(payload.get("filled_cells", 0), 0)

    run_archive = load_and_collapse_jsonl(
        archive_path,
        archive_type=archive_type,  # type: ignore[arg-type]
        resolution=resolution,
    )
    if (
        not bool(payload.get("standard_benchmark", False))
        and not bool(payload.get("dungeon_benchmark", False))
        and archive_type == "grid"
        and expected_filled > run_archive.filled_count()
        and GRID_BASELINE_ARCHIVE.is_file()
    ):
        base = load_and_collapse_jsonl(
            GRID_BASELINE_ARCHIVE,
            archive_type="grid",
            resolution=resolution,
        )
        merge_archives(base, run_archive)
        return base
    return run_archive


def _archive_metrics(
    payload: dict[str, object],
    archive_path: Path,
) -> tuple[float | None, float | None]:
    """Return ``(mean_best_fitness, qd_score)`` from collapsed archive state."""
    if (
        payload.get("qd_score") is not None
        and payload.get("mean_best_fitness") is not None
    ):
        mean_fit = _as_float(payload["mean_best_fitness"])
        qd_val = _as_float(payload["qd_score"])
        if mean_fit is not None and qd_val is not None:
            return mean_fit, qd_val
    if not archive_path.is_file():
        return None, None
    collapsed = _collapsed_archive_for_summary(payload, archive_path)
    qd = qd_score_from_archive(collapsed)
    filled = collapsed.filled_count()
    mean = qd / float(filled) if filled else None
    return mean, qd


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
        "hints_rich",
        "hints_parent",
        "hints_direction",
        "filter",
        "vanilla",
        "genetic_me",
        "genetic_me_uniform",
        "genetic_me_filter",
        "cma_me",
        "cma_mae",
        "cma_me_threshold",
        "cma_me_bernoulli",
        "cma_me_discrete",
        "cma_me_pbcma",
        "cma_me_cold",
        "me_random",
        "random",
        "genetic",
        "genetic_filter",
        "llm_stub",
        "llm_hints",
        "llm_hints_filter",
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
        mean_fit, qd = _archive_metrics(payload, archive)
        rows.append(
            {
                "condition": _infer_condition(run_dir),
                "benchmark": payload.get("benchmark"),
                "standard_benchmark": payload.get("standard_benchmark", False),
                "dungeon_benchmark": payload.get("dungeon_benchmark", False),
                "seed": payload.get("seed"),
                "replicate": payload.get("replicate", _infer_replicate(run_dir)),
                "iterations": payload.get("iterations"),
                "evaluations": payload.get("evaluations"),
                "proposals": payload.get("proposals"),
                "skipped": payload.get("skipped"),
                "filled_cells": payload.get("filled_cells"),
                "coverage_pct": round(
                    (_as_float(payload.get("coverage", 0.0)) or 0.0) * 100.0, 4
                ),
                "mean_best_fitness": (
                    round(mean_fit, 6) if mean_fit is not None else None
                ),
                "qd_score": round(qd, 6) if qd is not None else None,
                "skip_rate_pct": (
                    round(
                        (_as_float(payload.get("skip_rate", 0.0)) or 0.0) * 100.0,
                        4,
                    )
                    if payload.get("dungeon_benchmark")
                    else _skip_rate(surrogate_archive)
                ),
                "llm_enabled": payload.get("llm_enabled"),
                "surrogate_enabled": payload.get("surrogate_enabled"),
                "pyribs_version": payload.get("pyribs_version"),
                "pyribs_algo": payload.get("pyribs_algo"),
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "llm_stack_version": payload.get("llm_stack_version"),
                "llm_model": payload.get("llm_model"),
                "llm_temperature": payload.get("llm_temperature"),
                "llm_top_p": payload.get("llm_top_p"),
                "llm_spec_hash": payload.get("llm_spec_hash"),
                "llm_fallback_rate_pct": payload.get("llm_fallback_rate_pct"),
                "llm_calls": payload.get("llm_calls"),
                "llm_parse_success_rate": payload.get("llm_parse_success_rate"),
                "llm_mean_tile_distance": payload.get("llm_mean_tile_distance"),
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

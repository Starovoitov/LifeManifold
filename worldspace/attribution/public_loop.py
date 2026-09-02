"""Shared public-domain MAP-Elites loop utilities."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

AllocationKind = Literal["static", "state_aware_median"]
SUMMARY_SCHEMA = "public-qd-1.0"
SUMMARY_FILENAME = "nightly_run_summary.json"
TRACE_FILENAME = "archive_trace.jsonl"


@dataclass(frozen=True)
class PublicRunConfig:
    """Resolved treatment knobs for one native public-domain run."""

    seed: int
    generator: Literal["random", "genetic", "llm"]
    selector: Literal[
        "uniform_frontier",
        "min_fitness_frontier",
        "max_fitness_frontier",
    ]
    allocation: AllocationKind = "static"
    prompt_channel: Literal["not_applicable", "constant", "live"] = "not_applicable"
    repair_kind: str = "identity"
    floor_random: int = 20
    search_horizon: int = 200
    llm_call_cap: int = 200
    capture_events: bool = True


def should_use_llm(
    *,
    allocation: AllocationKind,
    archive_fitnesses: list[float],
    target_empty: bool,
    target_fitness: float | None,
    completed_llm_calls: int,
    llm_call_cap: int,
) -> bool:
    """Return whether the next slot should spend an LLM call."""
    if completed_llm_calls >= llm_call_cap:
        return False
    if allocation == "static":
        return True
    if allocation != "state_aware_median":
        raise ValueError(f"unknown allocation {allocation!r}")
    if target_empty:
        return True
    if not archive_fitnesses:
        return True
    if target_fitness is None:
        return True
    return float(target_fitness) <= _median(archive_fitnesses)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def anytime_auc(values: list[float]) -> float:
    """Trapezoidal AUC over equal-spaced proposal indices 1..n, / n."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    area = 0.0
    for left, right in zip(values[:-1], values[1:], strict=True):
        area += 0.5 * (left + right)
    return area / float(len(values) - 1)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def is_close(a: float, b: float, *, tol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)

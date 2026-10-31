"""Shared public-domain MAP-Elites loop utilities."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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
    expected_response_model: str | None = None


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


def anytime_auc(values: Sequence[float]) -> float:
    """Mean trapezoid of an equal-spaced QD curve: area / (n − 1).

    One point returns that point. An empty curve returns 0. This is the only
    QD-score anytime AUC used by the public runners and by confirmatory
    analysis of ``qd_score_anytime_auc``.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    area = 0.0
    for left, right in zip(values[:-1], values[1:], strict=True):
        area += 0.5 * (float(left) + float(right))
    return area / float(len(values) - 1)


def search_curve_auc(values: Sequence[float], floor_proposals: int) -> float:
    """AUC of the curve after the random floor, via :func:`anytime_auc`."""
    if floor_proposals < 0:
        raise ValueError("floor_proposals cannot be negative")
    if floor_proposals > len(values):
        raise ValueError("floor_proposals exceeds the recorded curve")
    return anytime_auc(values[floor_proposals:])


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


@dataclass
class LlmCallLedger:
    """HTTP and token totals for one LLM arm, including failed retries."""

    http_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_complete: bool = True
    calls: int = 0
    channels: set[str] = field(default_factory=set)
    template_hashes: set[str] = field(default_factory=set)
    response_models: set[str] = field(default_factory=set)


def record_llm_meta(ledger: LlmCallLedger, meta: Mapping[str, Any]) -> None:
    """Add one runner callback result. Missing token fields void the sum."""
    ledger.calls += 1
    http = meta.get("http_calls")
    if http is not None:
        ledger.http_calls += int(http)
    channel = meta.get("prompt_channel")
    if isinstance(channel, str) and channel:
        ledger.channels.add(channel)
    digest = meta.get("prompt_template_sha256")
    if isinstance(digest, str) and digest:
        ledger.template_hashes.add(digest)
    model = meta.get("response_model")
    if isinstance(model, str) and model:
        ledger.response_models.add(model)
    prompt = meta.get("prompt_tokens")
    completion = meta.get("completion_tokens")
    total = meta.get("total_tokens")
    if prompt is None or completion is None or total is None:
        ledger.tokens_complete = False
        return
    ledger.prompt_tokens += int(prompt)
    ledger.completion_tokens += int(completion)
    ledger.total_tokens += int(total)


def llm_summary_fields(
    ledger: LlmCallLedger,
    config: PublicRunConfig,
) -> dict[str, Any]:
    """Summary slice for an LLM arm.

    ``prompt_channel`` is the channel the emitter rendered. When an expected
    response snapshot is set and the observed models are not exactly that
    snapshot, ``response_model_mismatch`` is true.
    """
    if len(ledger.channels) == 1:
        channel = next(iter(ledger.channels))
    else:
        channel = config.prompt_channel
    template = (
        next(iter(ledger.template_hashes)) if len(ledger.template_hashes) == 1 else None
    )
    if len(ledger.response_models) == 1:
        response_model = next(iter(ledger.response_models))
    else:
        response_model = None
    expected = config.expected_response_model
    mismatch = (
        bool(expected)
        and ledger.calls > 0
        and ledger.response_models != {expected}
    )
    tokens_ready = ledger.tokens_complete and ledger.calls > 0
    return {
        "prompt_channel": channel,
        "prompt_template_sha256": template,
        "response_model": response_model,
        "response_models": sorted(ledger.response_models),
        "expected_response_model": expected,
        "response_model_mismatch": mismatch,
        "llm_http_calls": ledger.http_calls,
        "prompt_tokens": ledger.prompt_tokens if tokens_ready else None,
        "completion_tokens": ledger.completion_tokens if tokens_ready else None,
        "total_tokens": ledger.total_tokens if tokens_ready else None,
    }

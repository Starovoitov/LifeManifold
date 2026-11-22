"""Public slice of a private LLM call record.

Section 11 publishes model, token counts, latency, and prompt hashes.
Rendered messages and provider response text stay out of the release.
"""

from __future__ import annotations

from typing import Any

SANITIZED_SCHEMA = "phase4-sanitized-llm-call-1"


def sanitize_llm_call_record(
    record: dict[str, Any],
    *,
    prompt_template_sha256: str | None,
) -> dict[str, Any]:
    """Copy the public fields of one call. Unknown keys are dropped."""
    usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
    details = usage.get("prompt_tokens_details")
    cached = _int_or_none(details.get("cached_tokens")) if isinstance(details, dict) else None
    return {
        "schema": SANITIZED_SCHEMA,
        "ts_utc": record.get("ts_utc"),
        "call_id": record.get("call_id"),
        "ok": record.get("ok"),
        "attempts": record.get("attempts"),
        "latency_ms": record.get("latency_ms"),
        "requested_model": record.get("requested_model"),
        "response_model": record.get("response_model"),
        "finish_reason": record.get("finish_reason"),
        "request_body_sha256": record.get("request_body_sha256"),
        "prompt_template_sha256": prompt_template_sha256,
        "prompt_tokens": _int_or_none(usage.get("prompt_tokens")),
        "completion_tokens": _int_or_none(usage.get("completion_tokens")),
        "total_tokens": _int_or_none(usage.get("total_tokens")),
        "cached_prompt_tokens": cached,
    }


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value

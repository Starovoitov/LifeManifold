"""LLM emitter for NAS-Bench-201: JSON ops, identity repair, genetic fallback."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from worldspace.generators import llm_retry_backoff_seconds
from worldspace.generators.llm_config import LlmTextCaller, load_llm_config
from worldspace.generators.llm_call_log import get_llm_call_record
from worldspace.nas201.emitters import Nas201EmitterResult, mutate_one_edge
from worldspace.nas201.prompt_scan import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    assert_prompt_templates,
    assert_runtime_user_prompt,
)
from worldspace.nas201.spec import Nas201Spec, hamming_ops, try_parse_ops_payload

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLM_SPEC = _ROOT / "worldspace/specs/llm_world_generator_openai.yaml"


@dataclass
class Nas201LlmAudit:
    attempts: int = 0
    api_calls: int = 0
    retries: int = 0
    parse_successes: int = 0
    fallbacks: int = 0
    exact_duplicates: int = 0
    total_hamming: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "attempts": self.attempts,
            "api_calls": self.api_calls,
            "retries": self.retries,
            "parse_successes": self.parse_successes,
            "parse_success_rate": (
                self.parse_successes / self.attempts if self.attempts else 0.0
            ),
            "fallbacks": self.fallbacks,
            "fallback_rate": self.fallbacks / self.attempts if self.attempts else 0.0,
            "exact_duplicates": self.exact_duplicates,
            "mean_hamming_parse_valid": (
                self.total_hamming / self.parse_successes
                if self.parse_successes
                else 0.0
            ),
            "failure_reasons": dict(sorted(self.failure_reasons.items())),
        }


@dataclass(frozen=True)
class Nas201LlmProposal:
    parent: Nas201Spec
    child: Nas201Spec
    emitter_type: str
    schema_valid: bool
    used_fallback: bool
    hamming: int
    exact_duplicate: bool
    raw_response: str | None
    miss_reason: str | None
    api_calls: int
    retries: int
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


class Nas201LlmEmitter:
    """Constant-grammar prompt channel. Repair is identity (no op rewriting)."""

    def __init__(
        self,
        *,
        llm_spec_path: Path = DEFAULT_LLM_SPEC,
        call_llm_text: LlmTextCaller | None = None,
        system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT,
        user_prompt_path: Path = DEFAULT_USER_PROMPT,
        max_retries: int = 2,
    ) -> None:
        templates = assert_prompt_templates(system_prompt_path, user_prompt_path)
        self.llm_spec_path = Path(llm_spec_path)
        self.config = load_llm_config(self.llm_spec_path)
        self.call_llm_text = call_llm_text
        self.system_prompt = templates["system"]
        self.user_prompt = templates["user"]
        self.max_retries = max(0, max_retries)
        self.audit = Nas201LlmAudit()
        self.prompt_version = hashlib.sha256(
            f"{self.system_prompt}\n---\n{self.user_prompt}".encode()
        ).hexdigest()[:16]

    def emit(
        self,
        parent: Nas201Spec,
        rng: np.random.Generator,
        *,
        proposal_index: int = 0,
    ) -> Nas201LlmProposal:
        prompt = self.user_prompt.format(parent_json=parent.canonical_json())
        assert_runtime_user_prompt(prompt, source=f"proposal {proposal_index}")
        self.audit.attempts += 1
        started = time.perf_counter()
        child: Nas201Spec | None = None
        raw: str | None = None
        last_reason = "unknown"
        api_calls = 0
        retries = 0
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None
        for request_index in range(self.max_retries + 1):
            if request_index:
                retries += 1
                self.audit.retries += 1
                time.sleep(llm_retry_backoff_seconds(request_index))
            try:
                api_calls += 1
                self.audit.api_calls += 1
                call_id = f"nas201-isolated-{proposal_index}-{request_index}-{uuid.uuid4().hex[:8]}"
                raw = self._request(prompt, call_id=call_id)
                child = parse_nas201_response(raw)
                usage = _usage_from_call_id(call_id)
                if usage is not None:
                    prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    completion_tokens = int(usage.get("completion_tokens") or 0)
                    total_tokens = int(usage.get("total_tokens") or 0)
                break
            except (ValueError, json.JSONDecodeError) as error:
                last_reason = _failure_reason(error)
            except RuntimeError as error:
                if "LLM request failed" not in str(error):
                    raise
                last_reason = "network"
        latency_ms = (time.perf_counter() - started) * 1000.0
        if child is None:
            self.audit.fallbacks += 1
            self.audit.failure_reasons[last_reason] = (
                self.audit.failure_reasons.get(last_reason, 0) + 1
            )
            fallback = mutate_one_edge(parent, rng)
            return Nas201LlmProposal(
                parent=parent,
                child=fallback,
                emitter_type="llm_fallback_genetic",
                schema_valid=False,
                used_fallback=True,
                hamming=hamming_ops(parent, fallback),
                exact_duplicate=False,
                raw_response=raw,
                miss_reason=last_reason,
                api_calls=api_calls,
                retries=retries,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        hamming = hamming_ops(parent, child)
        duplicate = hamming == 0
        self.audit.parse_successes += 1
        self.audit.total_hamming += hamming
        self.audit.exact_duplicates += int(duplicate)
        return Nas201LlmProposal(
            parent=parent,
            child=child,
            emitter_type="llm",
            schema_valid=True,
            used_fallback=False,
            hamming=hamming,
            exact_duplicate=duplicate,
            raw_response=raw,
            miss_reason=None,
            api_calls=api_calls,
            retries=retries,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def _request(self, prompt: str, *, call_id: str) -> str:
        if self.call_llm_text is None:
            from worldspace.generators import call_llm

            caller: LlmTextCaller = call_llm
        else:
            caller = self.call_llm_text
        return caller(
            mode=self.config.mode,
            provider_name=self.config.active_provider,
            providers=self.config.providers,
            prompt=prompt,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=max(self.config.max_tokens, 200),
            system_content=self.system_prompt,
            audit_context={"llm_call_id": call_id, "stage": "nas201_isolated"},
        )


def parse_nas201_response(response: str) -> Nas201Spec:
    """Strict ops-JSON. Official arch strings are not schema-valid."""
    start = response.find("{")
    end = response.rfind("}")
    if start < 0 or end < start:
        raise json.JSONDecodeError("no JSON object", response, 0)
    payload = json.loads(response[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("response must be a JSON object")
    if set(payload) != {"ops"}:
        raise ValueError("response must contain exactly the ops key")
    spec = try_parse_ops_payload(payload)
    if spec is None:
        raise ValueError("ops must be six names from the closed operation list")
    return spec


def to_emitter_result(proposal: Nas201LlmProposal) -> Nas201EmitterResult:
    return Nas201EmitterResult(
        spec=proposal.child,
        parent_id=proposal.parent.candidate_hash(),
        emitter_type=proposal.emitter_type,
    )


def _usage_from_call_id(call_id: str) -> dict[str, Any] | None:
    record = get_llm_call_record(call_id)
    if record is None:
        return None
    usage = record.get("usage")
    return usage if isinstance(usage, dict) else None


def _failure_reason(error: Exception) -> str:
    if isinstance(error, json.JSONDecodeError):
        return "json_decode"
    message = str(error).lower()
    if "exactly the ops key" in message:
        return "extra_keys"
    if "six names" in message:
        return "ops_schema"
    return "payload"

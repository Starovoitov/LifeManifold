"""LLM emitter for sokoban-v0: JSON grid, identity repair, genetic fallback."""

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
from worldspace.generators.llm_call_log import get_llm_call_record
from worldspace.generators.llm_config import LlmTextCaller, load_llm_config
from worldspace.pcg.copy_audit import copy_readme_example
from worldspace.pcg.emitters import PcgEmitterResult, mutate_one_tile
from worldspace.pcg.p7 import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    assert_p7_runtime_user_prompt,
    assert_p7_templates,
)
from worldspace.pcg.spec import PcgSpec, SOKOBAN_V0, hamming_tiles, try_parse_grid

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLM_SPEC = _ROOT / "worldspace/specs/llm_world_generator_openai.yaml"


@dataclass
class PcgLlmAudit:
    attempts: int = 0
    api_calls: int = 0
    retries: int = 0
    parse_successes: int = 0
    fallbacks: int = 0
    exact_duplicates: int = 0
    copy_readme: int = 0
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
            "copy_readme": self.copy_readme,
            "mean_hamming_parse_valid": (
                self.total_hamming / self.parse_successes
                if self.parse_successes
                else 0.0
            ),
            "failure_reasons": dict(sorted(self.failure_reasons.items())),
        }


@dataclass(frozen=True)
class PcgLlmProposal:
    parent: PcgSpec
    child: PcgSpec
    emitter_type: str
    schema_valid: bool
    used_fallback: bool
    hamming: int
    exact_duplicate: bool
    copy_readme_example: bool
    raw_response: str | None
    miss_reason: str | None
    api_calls: int
    retries: int
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_model: str | None


class PcgSokobanLlmEmitter:
    """Constant-grammar prompt channel. Repair is identity (no tile rewriting)."""

    def __init__(
        self,
        *,
        llm_spec_path: Path = DEFAULT_LLM_SPEC,
        call_llm_text: LlmTextCaller | None = None,
        system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT,
        user_prompt_path: Path = DEFAULT_USER_PROMPT,
        max_retries: int = 2,
    ) -> None:
        templates = assert_p7_templates(system_prompt_path, user_prompt_path)
        self.llm_spec_path = Path(llm_spec_path)
        self.config = load_llm_config(self.llm_spec_path)
        self.call_llm_text = call_llm_text
        self.system_prompt = templates["system"]
        self.user_prompt = templates["user"]
        self.max_retries = max(0, max_retries)
        self.audit = PcgLlmAudit()
        self.prompt_version = hashlib.sha256(
            f"{self.system_prompt}\n---\n{self.user_prompt}".encode()
        ).hexdigest()[:16]
        self.last_response_model: str | None = None

    def emit(
        self,
        parent: PcgSpec,
        rng: np.random.Generator,
        *,
        proposal_index: int = 0,
    ) -> PcgLlmProposal:
        if parent.problem_name != SOKOBAN_V0.problem_name:
            raise ValueError("P2.4 emitter is sokoban-v0 only")
        parent_json = json.dumps(
            {"grid": parent.to_nested_list()}, separators=(",", ":")
        )
        prompt = self.user_prompt.format(parent_json=parent_json)
        assert_p7_runtime_user_prompt(prompt, source=f"proposal {proposal_index}")
        self.audit.attempts += 1
        started = time.perf_counter()
        child: PcgSpec | None = None
        raw: str | None = None
        last_reason = "unknown"
        api_calls = 0
        retries = 0
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None
        response_model: str | None = None
        for request_index in range(self.max_retries + 1):
            if request_index:
                retries += 1
                self.audit.retries += 1
                time.sleep(llm_retry_backoff_seconds(request_index))
            try:
                api_calls += 1
                self.audit.api_calls += 1
                call_id = (
                    f"pcg-p2.4-{proposal_index}-{request_index}-{uuid.uuid4().hex[:8]}"
                )
                raw = self._request(prompt, call_id=call_id)
                child = parse_sokoban_response(raw)
                usage, response_model = _usage_and_model_from_call_id(call_id)
                if usage is not None:
                    prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    completion_tokens = int(usage.get("completion_tokens") or 0)
                    total_tokens = int(usage.get("total_tokens") or 0)
                if response_model:
                    self.last_response_model = response_model
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
            fallback = mutate_one_tile(parent, rng)
            copied = copy_readme_example(fallback)
            self.audit.copy_readme += int(copied)
            return PcgLlmProposal(
                parent=parent,
                child=fallback,
                emitter_type="llm_fallback_genetic",
                schema_valid=False,
                used_fallback=True,
                hamming=hamming_tiles(parent, fallback),
                exact_duplicate=False,
                copy_readme_example=copied,
                raw_response=raw,
                miss_reason=last_reason,
                api_calls=api_calls,
                retries=retries,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                response_model=response_model,
            )
        hamming = hamming_tiles(parent, child)
        duplicate = hamming == 0
        copied = copy_readme_example(child)
        self.audit.parse_successes += 1
        self.audit.total_hamming += hamming
        self.audit.exact_duplicates += int(duplicate)
        self.audit.copy_readme += int(copied)
        return PcgLlmProposal(
            parent=parent,
            child=child,
            emitter_type="llm",
            schema_valid=True,
            used_fallback=False,
            hamming=hamming,
            exact_duplicate=duplicate,
            copy_readme_example=copied,
            raw_response=raw,
            miss_reason=None,
            api_calls=api_calls,
            retries=retries,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            response_model=response_model,
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
            audit_context={"llm_call_id": call_id, "slice": "P2.4"},
        )


def parse_sokoban_response(response: str) -> PcgSpec:
    """Strict grid JSON: {"grid": ...} with no extra keys, or a bare 5×5 array."""
    object_start = response.find("{")
    array_start = response.find("[")
    if object_start >= 0 and (array_start < 0 or object_start < array_start):
        end = response.rfind("}")
        if end < object_start:
            raise json.JSONDecodeError("no JSON object", response, 0)
        payload = json.loads(response[object_start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("response must be a JSON object")
        if set(payload) != {"grid"}:
            raise ValueError("response must contain exactly the grid key")
        spec = try_parse_grid(payload, SOKOBAN_V0)
        if spec is None:
            raise ValueError("grid must be a 5x5 integer matrix with tiles 0-4")
        return spec
    if array_start >= 0:
        end = response.rfind("]")
        if end < array_start:
            raise json.JSONDecodeError("no JSON array", response, 0)
        payload = json.loads(response[array_start : end + 1])
        spec = try_parse_grid(payload, SOKOBAN_V0)
        if spec is None:
            raise ValueError("grid must be a 5x5 integer matrix with tiles 0-4")
        return spec
    raise json.JSONDecodeError("no JSON object", response, 0)


def to_emitter_result(proposal: PcgLlmProposal) -> PcgEmitterResult:
    return PcgEmitterResult(
        spec=proposal.child,
        parent_id=proposal.parent.candidate_hash(),
        emitter_type=proposal.emitter_type,
    )


def _usage_and_model_from_call_id(
    call_id: str,
) -> tuple[dict[str, Any] | None, str | None]:
    record = get_llm_call_record(call_id)
    if record is None:
        return None, None
    usage = record.get("usage")
    usage_dict = usage if isinstance(usage, dict) else None
    model = record.get("response_model")
    model_str = str(model) if isinstance(model, str) and model else None
    return usage_dict, model_str


def _failure_reason(error: Exception) -> str:
    if isinstance(error, json.JSONDecodeError):
        return "json_decode"
    message = str(error).lower()
    if "exactly the grid key" in message:
        return "extra_keys"
    if "5x5" in message or "tiles" in message:
        return "grid_schema"
    return "payload"

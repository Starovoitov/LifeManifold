"""Strict JSON LLM emitter for Fontaine Sphere RQ1 / H1."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from threading import Lock

import numpy as np
from numpy.typing import NDArray

from worldspace.benchmarks.qd_sphere import CLIP_BOUND, SPHERE_SHIFT, clip_solution
from worldspace.benchmarks.sphere_rq1 import (
    DEFAULT_SIGMA,
    STUB_FITNESS,
    STUB_UNCERTAINTY,
    SphereEmitterResult,
    SpherePrediction,
    SphereTarget,
    emit_genetic,
)
from worldspace.generators import llm_retry_backoff_seconds
from worldspace.generators.llm_config import (
    LlmTextCaller,
    load_llm_config,
)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYSTEM_PROMPT = _ROOT / "prompts/sphere_llm_emitter_system.txt"
DEFAULT_USER_PROMPT = _ROOT / "prompts/sphere_llm_emitter_user.txt"
DEFAULT_LLM_SPEC = _ROOT / "worldspace/specs/llm_world_generator_rq1_fixed_openai.yaml"
MAX_ABS_DELTA = 1.5
MIN_ACTIVE = 4
MIN_ACTIVE_ABS = 0.05
MIN_L2 = 0.1
SOLUTION_DIM = 20
FloatArray = NDArray[np.float64]


@dataclass
class SphereLlmAudit:
    attempts: int = 0
    api_calls: int = 0
    retries: int = 0
    parse_successes: int = 0
    fallbacks: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)
    invalid_response_reasons: dict[str, int] = field(default_factory=dict)
    total_l2: float = 0.0

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
            "failure_reasons": dict(sorted(self.failure_reasons.items())),
            "invalid_response_reasons": dict(
                sorted(self.invalid_response_reasons.items())
            ),
            "mean_l2": (
                self.total_l2 / self.parse_successes if self.parse_successes else 0.0
            ),
        }


class SphereLlmEmitter:
    def __init__(
        self,
        *,
        prompt_mode: str,
        llm_spec_path: Path = DEFAULT_LLM_SPEC,
        call_llm_text: LlmTextCaller | None = None,
        system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT,
        user_prompt_path: Path = DEFAULT_USER_PROMPT,
        max_retries: int = 2,
        sigma: float = DEFAULT_SIGMA,
    ) -> None:
        if prompt_mode not in ("stub", "hints"):
            raise ValueError("prompt_mode must be stub or hints")
        self.prompt_mode = prompt_mode
        self.llm_spec_path = Path(llm_spec_path)
        self.config = load_llm_config(self.llm_spec_path)
        self.call_llm_text = call_llm_text
        self.system_prompt = system_prompt_path.read_text(encoding="utf-8")
        self.user_prompt = user_prompt_path.read_text(encoding="utf-8")
        self.max_retries = max(0, max_retries)
        self.sigma = sigma
        self.audit = SphereLlmAudit()
        self._audit_lock = Lock()
        self.prompt_version = _prompt_hash(self.system_prompt, self.user_prompt)

    def emit(
        self,
        *,
        target: SphereTarget,
        rng: np.random.Generator,
        prediction: SpherePrediction | None,
    ) -> SphereEmitterResult:
        if target.parent is None:
            with self._audit_lock:
                self.audit.attempts += 1
                self.audit.fallbacks += 1
                self.audit.failure_reasons["no_parent"] = (
                    self.audit.failure_reasons.get("no_parent", 0) + 1
                )
            fallback = emit_genetic(target, rng, sigma=self.sigma)
            return SphereEmitterResult(
                solution=fallback.solution,
                parent_id=fallback.parent_id,
                emitter_type="llm_fallback_genetic",
            )
        parent = np.asarray(target.parent.solution, dtype=np.float64)
        effective = (
            prediction
            if self.prompt_mode == "hints" and prediction is not None
            else SpherePrediction(STUB_FITNESS, STUB_UNCERTAINTY)
        )
        prompt = self.user_prompt.format(
            target_m0=target.center[0],
            target_m1=target.center[1],
            parent_objective=target.parent.objective,
            parent_m0=target.parent.measures[0],
            parent_m1=target.parent.measures[1],
            surrogate_fitness=effective.fitness,
            surrogate_uncertainty=effective.uncertainty,
            parent_json=json.dumps([round(float(x), 4) for x in parent]),
        )
        with self._audit_lock:
            self.audit.attempts += 1
        child: FloatArray | None = None
        distance = 0.0
        last_reason = "unknown"
        for request_index in range(self.max_retries + 1):
            if request_index:
                with self._audit_lock:
                    self.audit.retries += 1
                time.sleep(llm_retry_backoff_seconds(request_index))
            try:
                with self._audit_lock:
                    self.audit.api_calls += 1
                request_prompt = (
                    prompt
                    if request_index == 0
                    else prompt + "\n\n" + _retry_guidance(parent)
                )
                response = self._request(request_prompt)
                child = apply_sphere_deltas(parent, parse_sphere_deltas(response))
                distance = float(np.linalg.norm(child - parent))
                break
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                last_reason = _failure_reason(error)
                _record_invalid_response(self, last_reason)
            except RuntimeError as error:
                if not _is_provider_runtime_error(error):
                    raise
                last_reason = "network"
                _record_invalid_response(self, last_reason)
            except (AttributeError, KeyError, IndexError):
                last_reason = "envelope"
                _record_invalid_response(self, last_reason)
        if child is None:
            with self._audit_lock:
                self.audit.fallbacks += 1
                self.audit.failure_reasons[last_reason] = (
                    self.audit.failure_reasons.get(last_reason, 0) + 1
                )
            fallback = emit_genetic(target, rng, sigma=self.sigma)
            return SphereEmitterResult(
                solution=fallback.solution,
                parent_id=fallback.parent_id,
                emitter_type="llm_fallback_genetic",
            )
        with self._audit_lock:
            self.audit.parse_successes += 1
            self.audit.total_l2 += distance
        return SphereEmitterResult(
            solution=child,
            parent_id=target.parent.candidate_id,
            emitter_type="llm",
        )

    def emit_batch(
        self,
        jobs: list[tuple[SphereTarget, np.random.Generator, SpherePrediction | None]],
        *,
        max_workers: int = 4,
    ) -> list[SphereEmitterResult]:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(
                executor.map(
                    lambda job: self.emit(
                        target=job[0],
                        rng=job[1],
                        prediction=job[2],
                    ),
                    jobs,
                )
            )

    def _request(self, prompt: str) -> str:
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
            max_tokens=max(self.config.max_tokens, 400),
            system_content=self.system_prompt,
        )


class MockSphereLlmEmitter:
    """Gaussian local mutation without a remote LLM (CI / smoke)."""

    prompt_version = "mock-sphere-v1"

    def __init__(self, *, sigma: float = DEFAULT_SIGMA) -> None:
        self.sigma = sigma
        self.audit = SphereLlmAudit()

    def emit(
        self,
        *,
        target: SphereTarget,
        rng: np.random.Generator,
        prediction: SpherePrediction | None,
    ) -> SphereEmitterResult:
        del prediction
        self.audit.attempts += 1
        if target.parent is None:
            fallback = emit_genetic(target, rng, sigma=self.sigma)
            self.audit.fallbacks += 1
            return SphereEmitterResult(
                solution=fallback.solution,
                parent_id=fallback.parent_id,
                emitter_type="llm_fallback_genetic",
            )
        emitted = emit_genetic(target, rng, sigma=self.sigma)
        parent = np.asarray(target.parent.solution, dtype=np.float64)
        self.audit.parse_successes += 1
        self.audit.total_l2 += float(np.linalg.norm(emitted.solution - parent))
        return SphereEmitterResult(
            solution=emitted.solution,
            parent_id=target.parent.candidate_id,
            emitter_type="llm",
        )

    def emit_batch(
        self,
        jobs: list[tuple[SphereTarget, np.random.Generator, SpherePrediction | None]],
        *,
        max_workers: int = 4,
    ) -> list[SphereEmitterResult]:
        del max_workers
        return [
            self.emit(target=job[0], rng=job[1], prediction=job[2]) for job in jobs
        ]


def parse_sphere_deltas(response: str) -> FloatArray:
    if not isinstance(response, str) or not response.strip():
        raise ValueError("empty LLM content")
    start = response.find("{")
    end = response.rfind("}")
    if start < 0 or end < start:
        raise json.JSONDecodeError("no JSON object", response, 0)
    payload = json.loads(response[start : end + 1])
    if not isinstance(payload, dict) or "deltas" not in payload:
        raise ValueError("response must contain deltas")
    raw = payload["deltas"]
    if not isinstance(raw, list) or len(raw) != SOLUTION_DIM:
        raise ValueError("deltas must contain exactly 20 numbers")
    try:
        deltas = np.asarray([float(item) for item in raw], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("deltas must be 20 finite floats") from exc
    if deltas.shape != (SOLUTION_DIM,) or not np.all(np.isfinite(deltas)):
        raise ValueError("deltas must be 20 finite floats")
    if np.max(np.abs(deltas)) > MAX_ABS_DELTA:
        raise ValueError("delta magnitude exceeds 1.5")
    if int(np.sum(np.abs(deltas) >= MIN_ACTIVE_ABS)) < MIN_ACTIVE:
        raise ValueError("need at least four |delta| >= 0.05")
    return deltas


def apply_sphere_deltas(parent: FloatArray, deltas: FloatArray) -> FloatArray:
    child = clip_solution(np.asarray(parent, dtype=np.float64) + deltas)
    distance = float(np.linalg.norm(child - parent))
    if distance < MIN_L2:
        raise ValueError("mutation too small after clip")
    if np.max(np.abs(child)) > CLIP_BOUND + 1e-12:
        raise ValueError("child escaped the search box")
    return child


def _retry_guidance(parent: FloatArray) -> str:
    order = np.argsort(np.abs(parent - SPHERE_SHIFT))[::-1][:4]
    coords = [int(i) for i in order]
    return (
        "Your previous response was invalid or too small. Return exactly 20 deltas. "
        f"Perturb coordinates {coords} by at least 0.05 and at most 1.5, and keep "
        "the other coordinates small. JSON only."
    )


def _is_provider_runtime_error(error: RuntimeError) -> bool:
    """True for transport failures and malformed provider envelopes.

    ``call_llm`` wraps HTTP 429/5xx/520–524, timeouts, and SSL as
    ``LLM request failed``. A 200 body that is HTML, missing ``choices``,
    or empty ``message.content`` is ``LLM response ...`` and must not
    kill the seed — retry, then genetic fallback, same as maze content
    parse failures.
    """
    text = str(error)
    return "LLM request failed" in text or "LLM response" in text


def _failure_reason(error: Exception) -> str:
    text = str(error).lower()
    if "json" in text:
        return "json"
    if "20" in text or "deltas" in text:
        return "schema"
    if "1.5" in text or "magnitude" in text:
        return "magnitude"
    if "four" in text or "0.05" in text:
        return "too_sparse"
    if "small" in text:
        return "too_small"
    return "invalid"


def _record_invalid_response(emitter: SphereLlmEmitter, reason: str) -> None:
    with emitter._audit_lock:
        emitter.audit.invalid_response_reasons[reason] = (
            emitter.audit.invalid_response_reasons.get(reason, 0) + 1
        )


def _prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\n---\n{user}".encode()).hexdigest()[:16]

#!/usr/bin/env python3
"""Pre-flight smoke test for RQ1f weak LLM: JSON + WorldSpec parse rate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.generators import call_llm
from worldspace.generators.llm_config import load_llm_config
from worldspace.specs.spec import WorldSpec
from worldspace.specs.world_spec_from_llm import (
    extract_json_object_from_text,
    world_spec_from_llm_payload,
)
from scripts.run_github_llm_map_elites import resolve_llm_spec_path

DEFAULT_PROVIDER = "weak"
DEFAULT_CALLS = 50
DEFAULT_GO_THRESHOLD = 0.95
DEFAULT_GRID_SIZE = 8
DEFAULT_STEPS = 200

_BASE_SPEC = WorldSpec(
    birth=[1, 3],
    survival=[2, 3],
    noise=0.02,
    resource_regen=0.05,
    predation=0.1,
    cell_types=["life", "food"],
    grid_size=DEFAULT_GRID_SIZE,
    steps=DEFAULT_STEPS,
    seed=0,
)

_SYSTEM_PROMPT = (
    "You emit valid JSON for a cellular-automata WorldSpec. "
    "Return only JSON with a top-level world_spec object."
)

_USER_PROMPT = """Target niche: stability ≈ 0.50, diversity ≈ 0.60
Surrogate predicts fitness ≈ 0.487, uncertainty = 0.71

Generate a new WorldSpec JSON:
{
  "world_spec": {
    "birth": [1, 2],
    "survival": [2, 3],
    "noise": 0.04,
    "resource_regen": 0.06,
    "predation": 0.10,
    "cell_types": ["life", "food"],
    "neighborhood": "moore"
  }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument(
        "--model",
        default="",
        help="Override model ID from llm_world_generator_weak.yaml",
    )
    parser.add_argument("--calls", type=int, default=DEFAULT_CALLS)
    parser.add_argument(
        "--go-threshold",
        type=float,
        default=DEFAULT_GO_THRESHOLD,
        help="Minimum fraction of calls yielding parseable WorldSpec",
    )
    return parser.parse_args()


def _resolve_api_key(preferred_env: str) -> tuple[str | None, str]:
    for env_name in (preferred_env, "QWEN_API_KEY", "DASHSCOPE_API_KEY"):
        if not env_name:
            continue
        value = os.environ.get(env_name)
        if value:
            return value, env_name
    return None, preferred_env or "QWEN_API_KEY"


def _parse_one(response: str) -> bool:
    parsed = extract_json_object_from_text(response)
    if parsed is None:
        return False
    spec = world_spec_from_llm_payload(
        parsed,
        grid_size=DEFAULT_GRID_SIZE,
        steps=DEFAULT_STEPS,
        base=_BASE_SPEC,
    )
    return spec is not None


def _classify_request_error(message: str) -> str | None:
    lowered = message.lower()
    if "403" in lowered or "forbidden" in lowered:
        return (
            "HTTP 403 — model not enabled on this DashScope workspace. "
            "Run: uv run python scripts/probe_dashscope_models.py "
            "Default weak arm uses qwen2.5-omni-7b; qwen2.5-7b-instruct often "
            "returns HTTP 403 — use scripts/probe_dashscope_models.py or "
            "preflight_llm_weak.py --model …"
        )
    if "401" in lowered or "unauthorized" in lowered:
        return "HTTP 401 — check QWEN_API_KEY / DASHSCOPE_API_KEY."
    return None


def main() -> int:
    args = parse_args()
    spec_path = resolve_llm_spec_path(args.provider)
    cfg = load_llm_config(spec_path)
    provider_name = cfg.active_provider
    providers = deepcopy(cfg.providers)
    provider = dict(providers[provider_name])
    if args.model.strip():
        provider["model"] = args.model.strip()
    providers[provider_name] = provider

    api_key_env = str(provider.get("api_key_env", "QWEN_API_KEY"))
    api_key, resolved_env = _resolve_api_key(api_key_env)
    if api_key is None:
        print(
            f"Missing API key env: {api_key_env} (or QWEN_API_KEY / DASHSCOPE_API_KEY)",
            file=sys.stderr,
        )
        return 2
    os.environ.setdefault(resolved_env, api_key)
    provider["api_key_env"] = resolved_env
    providers[provider_name] = provider

    successes = 0
    failures: list[str] = []
    auth_hint: str | None = None
    for index in range(int(args.calls)):
        try:
            response = call_llm(
                mode=cfg.mode,
                provider_name=provider_name,
                providers=providers,
                prompt=_USER_PROMPT,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_tokens,
                system_content=_SYSTEM_PROMPT,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            message = str(exc)
            failures.append(f"call_{index}:request:{message}")
            if auth_hint is None:
                auth_hint = _classify_request_error(message)
            continue
        if _parse_one(response):
            successes += 1
        else:
            preview = response.strip().replace("\n", " ")[:120]
            failures.append(f"call_{index}:parse:{preview!r}")

    total = int(args.calls)
    rate = float(successes) / float(total) if total > 0 else 0.0
    payload = {
        "provider": args.provider,
        "llm_spec": str(spec_path),
        "model": provider.get("model"),
        "api_key_env": resolved_env,
        "calls": total,
        "parse_successes": successes,
        "parse_rate": rate,
        "go_threshold": float(args.go_threshold),
        "go": rate >= float(args.go_threshold),
        "sample_failures": failures[:5],
    }
    if auth_hint is not None:
        payload["diagnosis"] = auth_hint
    print(json.dumps(payload, indent=2, sort_keys=True))
    if auth_hint is not None and successes == 0:
        print(f"VERDICT: NO-GO — {auth_hint}")
        return 1
    if payload["go"]:
        print("VERDICT: GO — weak model emits parseable WorldSpec often enough.")
        return 0
    print("VERDICT: NO-GO — parse rate too low; fix model/spec before pilot.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Probe which DashScope compatible-mode models respond with the current API key."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.generators import call_llm
from worldspace.generators.llm_config import load_llm_config

DEFAULT_MODELS = (
    "qwen-turbo",
    "qwen-flash",
    "qwen2.5-omni-7b",
    "qwen2.5-7b-instruct",
    "qwen2.5-7b-instruct-1m",
    "qwen2-7b-instruct",
)
DEFAULT_SPEC = ROOT / "worldspace/specs/llm_world_generator_qwen.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model IDs to probe (default: turbo, flash, 7B variants)",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC,
        help="Base LLM YAML (qwen stack; model field overridden per probe)",
    )
    return parser.parse_args()


def _resolve_api_key() -> tuple[str | None, str]:
    for env_name in ("QWEN_API_KEY", "DASHSCOPE_API_KEY"):
        value = os.environ.get(env_name)
        if value:
            return value, env_name
    return None, ""


def main() -> int:
    args = parse_args()
    api_key, env_name = _resolve_api_key()
    if api_key is None:
        print("Missing QWEN_API_KEY or DASHSCOPE_API_KEY", file=sys.stderr)
        return 2

    cfg = load_llm_config(args.spec)
    base_provider = dict(cfg.providers[cfg.active_provider])
    base_provider["api_key_env"] = env_name
    os.environ.setdefault(env_name, api_key)

    rows: list[dict[str, object]] = []
    for model in args.models:
        providers = {
            cfg.active_provider: {**base_provider, "model": model},
        }
        try:
            response = call_llm(
                mode=cfg.mode,
                provider_name=cfg.active_provider,
                providers=providers,
                prompt="Reply with the single word OK.",
                temperature=0.0,
                max_tokens=8,
                system_content="",
            )
            rows.append(
                {
                    "model": model,
                    "ok": True,
                    "preview": response.strip()[:80],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": model,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    payload = {
        "api_key_env": env_name,
        "spec": str(args.spec.resolve()),
        "results": rows,
        "working_models": [row["model"] for row in rows if row.get("ok")],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["working_models"]:
        print("No models responded — check key, region, and Model Studio access.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

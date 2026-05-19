"""Parse LLM JSON payloads into validated ``WorldSpec`` instances."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec
from worldspace.specs.world_param_bounds import (
    NOISE_MAX,
    NOISE_MIN,
    PREDATION_MAX,
    PREDATION_MIN,
    RESOURCE_REGEN_MAX,
    RESOURCE_REGEN_MIN,
)

__all__ = ["extract_json_object_from_text", "world_spec_from_llm_payload"]


def extract_json_object_from_text(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from model output."""
    stripped = text.strip()
    candidates = [stripped]
    if "```" in stripped:
        chunks = stripped.split("```")
        candidates.extend(chunk.strip() for chunk in chunks if chunk.strip())
    for candidate in candidates:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def world_spec_from_llm_payload(
    parsed: dict[str, Any],
    *,
    grid_size: int,
    steps: int,
    base: WorldSpec,
) -> WorldSpec | None:
    """Build a ``WorldSpec`` from an LLM JSON object (``world_spec`` key or rule fields)."""
    body = parsed.get("world_spec")
    if not isinstance(body, dict):
        if "birth" in parsed or "survival" in parsed:
            body = parsed
        else:
            return None
    birth = _normalize_rule_list(body.get("birth"), base.birth)
    survival = _normalize_rule_list(body.get("survival"), base.survival)
    if not birth or not survival:
        return None
    neighborhood = body.get("neighborhood", base.neighborhood)
    if not isinstance(neighborhood, str) or not neighborhood.strip():
        neighborhood = base.neighborhood
    return WorldSpec(
        birth=birth,
        survival=survival,
        noise=_clip_float(body.get("noise"), base.noise, NOISE_MIN, NOISE_MAX),
        resource_regen=_clip_float(
            body.get("resource_regen"),
            base.resource_regen,
            RESOURCE_REGEN_MIN,
            RESOURCE_REGEN_MAX,
        ),
        predation=_clip_float(
            body.get("predation"), base.predation, PREDATION_MIN, PREDATION_MAX
        ),
        cell_types=list(CANONICAL_CELL_TYPES),
        neighborhood=str(neighborhood),
        grid_size=grid_size,
        steps=steps,
        seed=0,
    )


def _normalize_rule_list(value: Any, fallback: list[int]) -> list[int]:
    if not isinstance(value, list):
        return list(fallback)
    vals: set[int] = set()
    for item in value:
        try:
            v = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= v <= 8:
            vals.add(v)
    if not vals:
        return list(fallback)
    return sorted(vals)


def _clip_float(value: Any, fallback: float, low: float, high: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return float(np.clip(f, low, high))

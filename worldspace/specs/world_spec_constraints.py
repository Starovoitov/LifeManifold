"""Shared ``WorldSpec`` field constraints for LLM patch and MAP-Elites emitters."""

from __future__ import annotations

from worldspace.specs.world_param_bounds import (
    NOISE_MAX,
    NOISE_MIN,
    PREDATION_MAX,
    PREDATION_MIN,
    RESOURCE_REGEN_MAX,
    RESOURCE_REGEN_MIN,
)

WORLD_SPEC_CONSTRAINTS: dict[str, str] = {
    "birth": "unique integers in [0,8], at least 1 item",
    "survival": "unique integers in [0,8], at least 1 item",
    "noise": f"[{NOISE_MIN},{NOISE_MAX}]",
    "resource_regen": f"[{RESOURCE_REGEN_MIN},{RESOURCE_REGEN_MAX}]",
    "predation": f"[{PREDATION_MIN},{PREDATION_MAX}]",
    "cell_types": 'normalize to ["life", "food"]',
}

__all__ = ["WORLD_SPEC_CONSTRAINTS", "format_world_spec_constraints"]


def format_world_spec_constraints() -> str:
    """Format constraints as bullet lines for LLM prompts."""
    lines = [f"- {key}: {value}" for key, value in WORLD_SPEC_CONSTRAINTS.items()]
    return "\n".join(lines)

"""Scan live PCG prompts: allow parent metrics; forbid archive/QD/few-shot."""

from __future__ import annotations

import re
from pathlib import Path

from worldspace.pcg.prompt_scan import (
    DEFAULT_SYSTEM_PROMPT,
    SokobanPromptError,
    assert_prompt_safe,
)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIVE_USER_PROMPT = _ROOT / "prompts/pcg_sokoban_llm_emitter_live_user.txt"

ALLOWED_PLACEHOLDERS = frozenset(
    {
        "parent_json",
        "parent_fitness",
        "parent_measure_0",
        "parent_measure_1",
        "parent_playable",
    }
)

LIVE_FORBIDDEN: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "archive_or_population",
        re.compile(r"\barchive\b|\bpopulation\b|\bqd[- ]?score\b", re.I),
    ),
    ("coverage", re.compile(r"\bcoverage\b", re.I)),
    ("few_shot", re.compile(r"example grid|for example|few[- ]shot|microban", re.I)),
    ("solution_leak", re.compile(r"\bsolution\b|\boptimal path\b", re.I)),
)


def live_prompt_violations(text: str) -> list[str]:
    found: list[str] = []
    for label, pattern in LIVE_FORBIDDEN:
        if pattern.search(text):
            found.append(label)
    placeholders = set(re.findall(r"\{([a-z0-9_]+)\}", text))
    unknown = sorted(placeholders - ALLOWED_PLACEHOLDERS)
    if unknown:
        found.append(f"unknown_placeholders:{','.join(unknown)}")
    return found


def assert_live_user_prompt(text: str, *, source: str) -> None:
    labels = live_prompt_violations(text)
    if labels:
        raise SokobanPromptError(
            f"PCG live prompt scan failed for {source}: {', '.join(labels)}"
        )


def assert_live_prompt_templates(
    system_path: Path = DEFAULT_SYSTEM_PROMPT,
    user_path: Path = DEFAULT_LIVE_USER_PROMPT,
) -> dict[str, str]:
    system = system_path.read_text(encoding="utf-8")
    user = user_path.read_text(encoding="utf-8")
    assert_prompt_safe(system, source=str(system_path))
    assert_live_user_prompt(user, source=str(user_path))
    return {"system": system, "user": user}

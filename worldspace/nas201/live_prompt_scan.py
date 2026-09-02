"""Scan live NAS prompts: allow parent search metrics; forbid test/archive/QD."""

from __future__ import annotations

import re
from pathlib import Path

from worldspace.nas201.prompt_scan import (
    DEFAULT_SYSTEM_PROMPT,
    Nas201PromptError,
    assert_prompt_safe,
)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIVE_USER_PROMPT = _ROOT / "prompts/nas201_llm_emitter_live_user.txt"

ALLOWED_PLACEHOLDERS = frozenset(
    {
        "parent_json",
        "parent_valid_accuracy",
        "parent_log_params",
        "parent_log_flops",
    }
)

# Live channel may name parent_* metrics; still forbid test / archive / few-shot.
LIVE_FORBIDDEN: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("test_split", re.compile(r"\btest[- ]?(set|acc|accuracy|split)\b", re.I)),
    (
        "dataset_goal",
        re.compile(r"\bcifar\b|\bimagenet\b|\bmnist\b", re.I),
    ),
    (
        "archive_or_population",
        re.compile(r"\barchive\b|\bpopulation\b|\bqd[- ]?score\b", re.I),
    ),
    ("coverage", re.compile(r"\bcoverage\b", re.I)),
    ("few_shot", re.compile(r"example cell|for example|few[- ]shot", re.I)),
    ("leaderboard", re.compile(r"leaderboard|\bsota\b|state-of-the-art", re.I)),
    (
        "bare_accuracy_goal",
        re.compile(r"(?<!parent_valid_)accuracy", re.I),
    ),
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
        raise Nas201PromptError(
            f"NAS live prompt scan failed for {source}: {', '.join(labels)}"
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

"""Refuse NAS prompts that mention metrics, datasets, or leaderboards.

Scan committed templates before any LLM batch. A failed scan means REVISE the
prompt; do not read the batch as GO.
"""

from __future__ import annotations

import re
from pathlib import Path

from worldspace.nas201.spec import OPERATIONS

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYSTEM_PROMPT = _ROOT / "prompts/nas201_llm_emitter_system.txt"
DEFAULT_USER_PROMPT = _ROOT / "prompts/nas201_llm_emitter_user.txt"

# Dataset names, search metrics, and prestige cues. Grammar (op names, DAG) is
# allowed; a filled six-op example cell is not (zero-shot, no few-shot cells).
FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("accuracy", re.compile(r"accuracy|\btop-?1\b|\bacc(?:uracy)?\b", re.I)),
    ("dataset_goal", re.compile(r"cifar|imagenet|mnist", re.I)),
    (
        "leaderboard",
        re.compile(r"leaderboard|\bsota\b|state-of-the-art|state of the art", re.I),
    ),
    ("benchmark_name", re.compile(r"nas-?bench|nats-?bench", re.I)),
    ("fitness_or_qd", re.compile(r"\bfitness\b|qd-?score|\bcoverage\b", re.I)),
    ("test_split", re.compile(r"\btest[- ]?(set|acc|accuracy|split)\b", re.I)),
    (
        "published_best",
        re.compile(r"state-of-the-art cell|best cell|famous arch", re.I),
    ),
)

_EXAMPLE_CELL = re.compile(
    r"\[\s*\"(?:"
    + "|".join(re.escape(op) for op in OPERATIONS)
    + r")\"\s*(?:,\s*\"(?:"
    + "|".join(re.escape(op) for op in OPERATIONS)
    + r")\"\s*){5}\]",
    re.I,
)
_ARCH_STRING_EXAMPLE = re.compile(
    r"\|[A-Za-z0-9_]+~0\|\+",
)


class Nas201PromptError(ValueError):
    """Raised when a NAS prompt template mentions metrics, datasets, or leaderboards."""


def prompt_violations(text: str) -> list[str]:
    """Return human-readable violation labels (empty if the text is allowed)."""
    found: list[str] = []
    for label, pattern in FORBIDDEN_PATTERNS:
        if pattern.search(text):
            found.append(label)
    if _EXAMPLE_CELL.search(text):
        found.append("few_shot_cell_example")
    if _ARCH_STRING_EXAMPLE.search(text):
        found.append("few_shot_arch_string")
    return found


def assert_prompt_safe(text: str, *, source: str) -> None:
    labels = prompt_violations(text)
    if labels:
        raise Nas201PromptError(
            f"NAS prompt scan failed for {source}: {', '.join(labels)}"
        )


def assert_prompt_templates(
    system_path: Path = DEFAULT_SYSTEM_PROMPT,
    user_path: Path = DEFAULT_USER_PROMPT,
) -> dict[str, str]:
    """Load templates and refuse a live batch if the prompt scan fails."""
    system = system_path.read_text(encoding="utf-8")
    user = user_path.read_text(encoding="utf-8")
    assert_prompt_safe(system, source=str(system_path))
    assert_prompt_safe(user, source=str(user_path))
    return {"system": system, "user": user}


def assert_runtime_user_prompt(text: str, *, source: str) -> None:
    """Filled user prompt: still no metrics/datasets; parent ops JSON is allowed."""
    found = [label for label, pattern in FORBIDDEN_PATTERNS if pattern.search(text)]
    if found:
        raise Nas201PromptError(
            f"NAS prompt scan failed for {source}: {', '.join(found)}"
        )

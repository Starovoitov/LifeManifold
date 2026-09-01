"""Refuse Sokoban prompts that are not zero-shot tile grammar.

Scan committed templates before any LLM batch. A failed scan means REVISE the
prompt; do not read the batch as GO.
"""

from __future__ import annotations

import re
from pathlib import Path

from worldspace.pcg.copy_audit import readme_example_compact_jsons

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYSTEM_PROMPT = _ROOT / "prompts/pcg_sokoban_llm_emitter_system.txt"
DEFAULT_USER_PROMPT = _ROOT / "prompts/pcg_sokoban_llm_emitter_user.txt"

# MAP-Elites / prestige / published-example cues. Tile names and playability
# rules are allowed; a filled 5×5 example grid is not.
FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("fitness_or_qd", re.compile(r"\bfitness\b|qd-?score|\bcoverage\b", re.I)),
    (
        "diversity_or_control_score",
        re.compile(r"\bdiversity\b|controllability|controlability", re.I),
    ),
    (
        "leaderboard",
        re.compile(r"leaderboard|\bsota\b|state-of-the-art|state of the art", re.I),
    ),
    (
        "benchmark_name",
        re.compile(r"pcg[-_ ]?benchmark|\bfdg\b|khalifa", re.I),
    ),
    ("archive_feedback", re.compile(r"\barchive\b|\bniche\b|map-?elites", re.I)),
    (
        "few_shot_language",
        re.compile(
            r"few[- ]?shot|example (?:level|grid|puzzle|map)|microban",
            re.I,
        ),
    ),
)

_INT_ROW = r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]"
_FIVE_BY_FIVE = re.compile(
    r"\[\s*" + r"\s*,\s*".join([_INT_ROW] * 5) + r"\s*\]",
    re.I,
)


class SokobanPromptError(ValueError):
    """Raised when a Sokoban prompt template is not zero-shot tile grammar."""


def prompt_violations(text: str) -> list[str]:
    """Return human-readable violation labels (empty if the text is allowed)."""
    found: list[str] = []
    for label, pattern in FORBIDDEN_PATTERNS:
        if pattern.search(text):
            found.append(label)
    if _FIVE_BY_FIVE.search(text):
        found.append("few_shot_grid_example")
    compact = re.sub(r"\s+", "", text)
    for example in readme_example_compact_jsons():
        if example in compact:
            found.append("readme_example_grid")
            break
    return found


def assert_prompt_safe(text: str, *, source: str) -> None:
    labels = prompt_violations(text)
    if labels:
        raise SokobanPromptError(
            f"Sokoban prompt scan failed for {source}: {', '.join(labels)}"
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
    """Filled user prompt: parent grid JSON is allowed; prestige cues are not."""
    found = [label for label, pattern in FORBIDDEN_PATTERNS if pattern.search(text)]
    if found:
        raise SokobanPromptError(
            f"Sokoban prompt scan failed for {source}: {', '.join(found)}"
        )

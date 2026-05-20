"""Load static LLM prompt text from the repository ``prompts/`` directory."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = _REPO_ROOT / "prompts"

__all__ = [
    "PROMPTS_DIR",
    "default_llm_patch_system_content",
    "read_prompt",
    "read_prompt_json",
]


def read_prompt(name: str) -> str:
    """Read a UTF-8 prompt file from ``prompts/`` (``name`` may include subdirs)."""
    path = PROMPTS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path.resolve()}")
    return path.read_text(encoding="utf-8")


def read_prompt_json(name: str) -> str:
    """Read a JSON prompt fragment (returns raw text for ``json.loads``)."""
    return read_prompt(name)


def default_llm_patch_system_content() -> str:
    """Default system message for legacy LLM patch / chat calls."""
    return read_prompt("llm_patch_system.txt").strip()

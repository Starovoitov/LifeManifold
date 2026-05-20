"""MAP-Elites LLM system prompt loading and rendering."""

from __future__ import annotations

import hashlib
from pathlib import Path

from worldspace.prompt_files import PROMPTS_DIR, read_prompt

DEFAULT_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "map_elites_llm_emitter_system.txt"
DEFAULT_USER_PROMPT_PATH = PROMPTS_DIR / "map_elites_llm_emitter_user.txt"

__all__ = [
    "DEFAULT_SYSTEM_PROMPT_PATH",
    "DEFAULT_USER_PROMPT_PATH",
    "USER_PROMPT_TEMPLATE",
    "load_system_prompt_template",
    "load_user_prompt_template",
    "render_system_prompt",
    "system_prompt_version",
]


def load_system_prompt_template(path: str | Path | None = None) -> str:
    """Read the raw system prompt template from disk."""
    if path is None:
        return read_prompt("map_elites_llm_emitter_system.txt")
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"LLM system prompt not found: {src.resolve()}")
    return src.read_text(encoding="utf-8")


def load_user_prompt_template(path: str | Path | None = None) -> str:
    """Read the MAP-Elites LLM user prompt template from disk."""
    if path is None:
        return read_prompt("map_elites_llm_emitter_user.txt")
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"LLM user prompt not found: {src.resolve()}")
    return src.read_text(encoding="utf-8")


def render_system_prompt(
    grid_resolution: int, *, path: str | Path | None = None
) -> str:
    """Substitute grid size placeholders into the system prompt template."""
    if grid_resolution < 1:
        msg = f"grid_resolution must be >= 1, got {grid_resolution}"
        raise ValueError(msg)
    template = load_system_prompt_template(path)
    bin_width = 1.0 / float(grid_resolution)
    return template.format(N=grid_resolution, bin_width=f"{bin_width:.6g}")


def system_prompt_version(path: str | Path | None = None) -> str:
    """Return the first 8 hex digits of the SHA-256 hash of the prompt file."""
    src = Path(path) if path is not None else DEFAULT_SYSTEM_PROMPT_PATH
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    return digest[:8]


USER_PROMPT_TEMPLATE = load_user_prompt_template()

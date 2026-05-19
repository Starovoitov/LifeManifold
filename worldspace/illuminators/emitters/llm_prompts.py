"""MAP-Elites LLM system prompt loading and rendering."""

from __future__ import annotations

import hashlib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SYSTEM_PROMPT_PATH = (
    _REPO_ROOT / "prompts" / "map_elites_llm_emitter_system.txt"
)

_USER_PROMPT_TEMPLATE = """\
Target niche: stability ≈ {target_stability:.2f} (±0.03), diversity ≈ {target_diversity:.2f} (±0.03)
Surrogate predicts fitness ≈ {surrogate_mean:.3f}, uncertainty = {surrogate_uncertainty:.3f}

Current best elite in this cell:
{current_elite_json}

Examples of successful elites from nearby niches (with fitness):
{few_shot_examples}

Generate a new WorldSpec that:
1. Lands in the target niche for stability and diversity.
2. Maximizes fitness per the formula above (avoid early extinction).
3. Differs from existing elites.
4. Accounts for high surrogate uncertainty.

Constraints for world_spec:
{constraints}

Return JSON only:
{{
  "reasoning": "2–4 sentences",
  "world_spec": {{ ... full WorldSpec without the seed field ... }}
}}
"""

USER_PROMPT_TEMPLATE = _USER_PROMPT_TEMPLATE

__all__ = [
    "DEFAULT_SYSTEM_PROMPT_PATH",
    "USER_PROMPT_TEMPLATE",
    "load_system_prompt_template",
    "render_system_prompt",
    "system_prompt_version",
]


def load_system_prompt_template(path: str | Path | None = None) -> str:
    """Read the raw system prompt template from disk."""
    src = Path(path or DEFAULT_SYSTEM_PROMPT_PATH)
    if not src.is_file():
        raise FileNotFoundError(f"LLM system prompt not found: {src.resolve()}")
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
    src = Path(path or DEFAULT_SYSTEM_PROMPT_PATH)
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    return digest[:8]

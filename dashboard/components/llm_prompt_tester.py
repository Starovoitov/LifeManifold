"""LLM prompt template loading and MAP-Elites user-prompt preview helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from dashboard.utils.bootstrap import ensure_repo_on_path
from dashboard.utils.config import load_config, resolve_repo_path

ensure_repo_on_path()

from worldspace.illuminators.archive import ArchiveElite, GridArchive
from worldspace.illuminators.emitters.llm_emitter import (
    build_user_prompt,
    format_current_elite_json,
    format_few_shot_block,
    moore_neighbor_elites,
)
from worldspace.illuminators.emitters.llm_prompts import (
    load_system_prompt_template,
    load_user_prompt_template,
    render_system_prompt,
)
from worldspace.illuminators.scheduler import TargetBin
from worldspace.specs.world_spec_constraints import format_world_spec_constraints

DEFAULT_USER_PROMPT_FILE = "map_elites_llm_emitter_user.txt"
DEFAULT_SYSTEM_PROMPT_FILE = "map_elites_llm_emitter_system.txt"
_PLACEHOLDER_PATTERN = re.compile(r"\{(\w+)(?::[^}]*)?\}")
PREVIEW_RNG_SEED = 0


def preview_rng(seed: int = PREVIEW_RNG_SEED) -> np.random.Generator:
    """Fresh RNG for one preview step (do not reuse across ``moore_neighbor_elites`` calls)."""
    return np.random.default_rng(seed)


__all__ = [
    "DEFAULT_SYSTEM_PROMPT_FILE",
    "DEFAULT_USER_PROMPT_FILE",
    "build_user_prompt_like_emitter",
    "list_format_placeholders",
    "load_grid_archive",
    "load_system_prompt_from_config",
    "load_user_prompt_from_config",
    "minimal_user_prompt_kwargs",
    "occupied_bins",
    "parent_world_spec_dict",
    "PREVIEW_RNG_SEED",
    "preview_rng",
    "render_system_prompt_preview",
    "render_user_prompt_preview",
    "resolve_prompts_dir",
    "target_bin_for_cell",
    "user_prompt_format_kwargs",
]


def resolve_prompts_dir(cfg: dict[str, Any] | None = None) -> Path:
    """Return the configured prompts directory under the repository root."""
    config = cfg if cfg is not None else load_config()
    paths_section = config.get("paths")
    relative = "prompts"
    if isinstance(paths_section, dict):
        raw = paths_section.get("prompts_dir")
        if isinstance(raw, str) and raw.strip():
            relative = raw.strip()
    return resolve_repo_path(relative)


def load_user_prompt_from_config(
    cfg: dict[str, Any] | None = None,
    *,
    filename: str = DEFAULT_USER_PROMPT_FILE,
) -> str:
    """Load the MAP-Elites LLM user prompt template from ``prompts_dir``."""
    path = resolve_prompts_dir(cfg) / filename
    return load_user_prompt_template(path)


def load_system_prompt_from_config(
    cfg: dict[str, Any] | None = None,
    *,
    filename: str = DEFAULT_SYSTEM_PROMPT_FILE,
) -> str:
    """Load the MAP-Elites LLM system prompt template from ``prompts_dir``."""
    path = resolve_prompts_dir(cfg) / filename
    return load_system_prompt_template(path)


def list_format_placeholders(template: str) -> list[str]:
    """Return unique ``str.format`` field names found in a prompt template."""
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _PLACEHOLDER_PATTERN.finditer(template):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def load_grid_archive(archive_path: Path, *, resolution: int) -> GridArchive:
    """Load a collapsed ``GridArchive`` from a MAP-Elites JSONL path."""
    from worldspace.illuminators.archive import load_and_collapse_jsonl

    return load_and_collapse_jsonl(archive_path, resolution=resolution)


def occupied_bins(archive: GridArchive) -> list[tuple[int, int]]:
    """List occupied archive cell coordinates."""
    bins: list[tuple[int, int]] = []
    size = archive.resolution
    for i in range(size):
        for j in range(size):
            if archive.get(i, j) is not None:
                bins.append((i, j))
    return bins


def target_bin_for_cell(
    archive: GridArchive,
    bin_xy: tuple[int, int],
    *,
    target_stability: float | None = None,
    target_diversity: float | None = None,
) -> TargetBin:
    """Build a ``TargetBin`` using elite measures or explicit niche centers."""
    elite = archive.get(bin_xy[0], bin_xy[1])
    stability = target_stability
    diversity = target_diversity
    if elite is not None and elite.measures is not None:
        if stability is None and "stability" in elite.measures:
            stability = float(elite.measures["stability"])
        if diversity is None and "diversity" in elite.measures:
            diversity = float(elite.measures["diversity"])
    if stability is None:
        stability = 0.5
    if diversity is None:
        diversity = 0.5
    return TargetBin(
        bin=bin_xy,
        target_stability=float(stability),
        target_diversity=float(diversity),
    )


def user_prompt_format_kwargs(
    archive: GridArchive,
    target: TargetBin,
    surrogate_mean: float,
    surrogate_uncertainty: float,
    *,
    rng: np.random.Generator,
    max_few_shot: int = 4,
) -> dict[str, Any]:
    """Keyword arguments for ``str.format`` on the user prompt template."""
    neighbors = moore_neighbor_elites(
        archive,
        target.bin,
        rng=rng,
        max_count=max_few_shot,
    )
    current = archive.get(*target.bin)
    return {
        "target_stability": float(target.target_stability),
        "target_diversity": float(target.target_diversity),
        "surrogate_mean": float(surrogate_mean),
        "surrogate_uncertainty": float(surrogate_uncertainty),
        "current_elite_json": format_current_elite_json(current),
        "few_shot_examples": format_few_shot_block(neighbors),
        "constraints": format_world_spec_constraints(),
    }


def render_user_prompt_preview(
    template: str, kwargs: dict[str, Any]
) -> tuple[str, str | None]:
    """Format a user prompt template; return ``(text, error)`` (error set on format failure)."""
    try:
        return template.format(**kwargs), None
    except (KeyError, ValueError, IndexError) as exc:
        return "", f"Template format error: {exc}"


def build_user_prompt_like_emitter(
    archive: GridArchive,
    target: TargetBin,
    surrogate_mean: float,
    surrogate_uncertainty: float,
    *,
    rng: np.random.Generator,
) -> str:
    """Delegate to ``build_user_prompt`` for parity with the illuminator emitter."""
    return build_user_prompt(
        target=target,
        archive=archive,
        surrogate_mean=surrogate_mean,
        surrogate_uncertainty=surrogate_uncertainty,
        rng=rng,
    )


def render_system_prompt_preview(
    grid_resolution: int,
    cfg: dict[str, Any] | None = None,
) -> str:
    """Render the system prompt with grid resolution placeholders filled in."""
    path = resolve_prompts_dir(cfg) / DEFAULT_SYSTEM_PROMPT_FILE
    return render_system_prompt(grid_resolution, path=path)


def parent_world_spec_dict(elite: ArchiveElite | None) -> dict[str, Any] | None:
    """Serialize parent ``world_spec`` for surrogate prediction."""
    if elite is None or elite.world_spec is None:
        return None
    return dict(elite.world_spec.to_json_dict())


def minimal_user_prompt_kwargs(
    surrogate_mean: float,
    surrogate_uncertainty: float,
    *,
    target_stability: float = 0.5,
    target_diversity: float = 0.5,
) -> dict[str, Any]:
    """Format kwargs when no archive is loaded (surrogate placeholders only)."""
    archive = GridArchive(1)
    target = TargetBin(
        bin=(0, 0),
        target_stability=float(target_stability),
        target_diversity=float(target_diversity),
    )
    return user_prompt_format_kwargs(
        archive,
        target,
        surrogate_mean,
        surrogate_uncertainty,
        rng=preview_rng(PREVIEW_RNG_SEED),
    )

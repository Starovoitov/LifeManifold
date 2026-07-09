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
from worldspace.illuminators.emitters.archive_neighbors import neighbor_elites
from worldspace.illuminators.emitters.llm_emitter import (
    build_user_prompt,
    format_current_elite_json,
    format_few_shot_block,
)
from worldspace.illuminators.emitters.llm_prompts import (
    load_system_prompt_template,
    load_user_prompt_template,
    render_system_prompt,
    render_system_prompt_for_archive_type,
)
from worldspace.illuminators.archive_protocol import ArchiveProtocol
from worldspace.illuminators.scheduler import TargetBin, TargetCell
from worldspace.specs.world_spec_constraints import format_world_spec_constraints

DEFAULT_USER_PROMPT_FILE = "map_elites_llm_emitter_user.txt"
DEFAULT_SYSTEM_PROMPT_FILE = "map_elites_llm_emitter_system.txt"
CVT_SYSTEM_PROMPT_FILE = "map_elites_llm_emitter_system_cvt.txt"
_PLACEHOLDER_PATTERN = re.compile(r"\{(\w+)(?::[^}]*)?\}")
PREVIEW_RNG_SEED = 0


def preview_rng(seed: int = PREVIEW_RNG_SEED) -> np.random.Generator:
    """Fresh RNG for one preview step (do not reuse across neighbor sampling calls)."""
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
    "occupied_cell_ids",
    "format_cell_label",
    "PREVIEW_RNG_SEED",
    "preview_rng",
    "render_system_prompt_preview",
    "render_user_prompt_preview",
    "resolve_prompts_dir",
    "target_bin_for_cell",
    "target_for_cell_id",
    "parent_world_spec_dict",
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


def load_grid_archive(
    archive_path: Path,
    *,
    resolution: int,
    archive_type: str | None = None,
    centroids_path: Path | None = None,
) -> ArchiveProtocol:
    """Load a collapsed archive from a MAP-Elites JSONL path (grid or CVT)."""
    from dashboard.components.archive_loader import (
        detect_archive_type_from_jsonl,
        resolve_centroids_path_for_archive,
    )
    from worldspace.illuminators.archive import load_and_collapse_jsonl

    resolved_type = archive_type or detect_archive_type_from_jsonl(archive_path)
    if resolved_type == "cvt":
        resolved_centroids = centroids_path
        if resolved_centroids is None:
            resolved_centroids = resolve_centroids_path_for_archive(archive_path)
        if resolved_centroids is None:
            msg = (
                "CVT centroids file not found next to this archive and no compatible "
                "baseline fallback is available."
            )
            raise FileNotFoundError(msg)
        return load_and_collapse_jsonl(
            archive_path,
            archive_type="cvt",
            centroids_path=resolved_centroids,
        )
    return load_and_collapse_jsonl(archive_path, resolution=resolution)


def occupied_cell_ids(archive: ArchiveProtocol) -> list[int]:
    """List occupied niche indices for grid or CVT archives."""
    return [
        cell_id
        for cell_id in range(archive.n_cells)
        if archive.get_cell(cell_id) is not None
    ]


def occupied_bins(archive: ArchiveProtocol) -> list[tuple[int, int]]:
    """List occupied grid bin coordinates (grid archives only)."""
    if archive.archive_type != "grid":
        msg = "occupied_bins requires a grid archive; use occupied_cell_ids for CVT"
        raise TypeError(msg)
    if not isinstance(archive, GridArchive):
        msg = "grid archive_type requires GridArchive instance"
        raise TypeError(msg)
    bins: list[tuple[int, int]] = []
    for cell_id in occupied_cell_ids(archive):
        bins.append(archive.bin_from_cell_id(cell_id))
    return bins


def format_cell_label(archive: ArchiveProtocol, cell_id: int) -> str:
    """Short label for archive cell selectors."""
    elite = archive.get_cell(cell_id)
    fitness = elite.fitness if elite is not None else float("nan")
    if archive.archive_type == "cvt":
        stability, diversity = archive.cell_center(cell_id)
        return (
            f"cell {cell_id} · s={stability:.2f}, d={diversity:.2f} · "
            f"fitness={fitness:.4f}"
        )
    if not isinstance(archive, GridArchive):
        msg = "grid archive_type requires GridArchive instance"
        raise TypeError(msg)
    bin_xy = archive.bin_from_cell_id(cell_id)
    return f"({bin_xy[0]}, {bin_xy[1]}) · fitness={fitness:.4f}"


def target_for_cell_id(
    archive: ArchiveProtocol,
    cell_id: int,
    *,
    target_stability: float | None = None,
    target_diversity: float | None = None,
) -> TargetBin:
    """Build a ``TargetBin`` for a flat niche index using elite measures or BC center."""
    elite = archive.get_cell(cell_id)
    stability = target_stability
    diversity = target_diversity
    if elite is not None and elite.measures is not None:
        if stability is None and "stability" in elite.measures:
            stability = float(elite.measures["stability"])
        if diversity is None and "diversity" in elite.measures:
            diversity = float(elite.measures["diversity"])
    if stability is None or diversity is None:
        center_s, center_d = archive.cell_center(cell_id)
        if stability is None:
            stability = center_s
        if diversity is None:
            diversity = center_d
    return TargetBin(
        bin=archive.bin_from_cell_id(cell_id),
        target_stability=float(stability),
        target_diversity=float(diversity),
    )


def target_bin_for_cell(
    archive: ArchiveProtocol,
    bin_xy: tuple[int, int],
    *,
    target_stability: float | None = None,
    target_diversity: float | None = None,
) -> TargetBin:
    """Build a ``TargetBin`` using elite measures or explicit niche centers."""
    return target_for_cell_id(
        archive,
        archive.cell_id_from_bin(bin_xy),
        target_stability=target_stability,
        target_diversity=target_diversity,
    )


def user_prompt_format_kwargs(
    archive: ArchiveProtocol,
    target: TargetBin,
    surrogate_mean: float,
    surrogate_uncertainty: float,
    *,
    rng: np.random.Generator,
    max_few_shot: int = 4,
) -> dict[str, Any]:
    """Keyword arguments for ``str.format`` on the user prompt template."""
    cell_id = archive.cell_id_from_bin(target.bin)
    neighbors = neighbor_elites(
        archive,
        cell_id,
        rng=rng,
        max_count=max_few_shot,
    )
    current = archive.get_cell(cell_id)
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
    archive: ArchiveProtocol,
    target: TargetBin,
    surrogate_mean: float,
    surrogate_uncertainty: float,
    *,
    rng: np.random.Generator,
) -> str:
    """Delegate to ``build_user_prompt`` for parity with the illuminator emitter."""
    target_cell = TargetCell(
        cell_id=archive.cell_id_from_bin(target.bin),
        target_stability=target.target_stability,
        target_diversity=target.target_diversity,
        bin_ij=target.bin,
    )
    return build_user_prompt(
        target=target_cell,
        archive=archive,
        surrogate_mean=surrogate_mean,
        surrogate_uncertainty=surrogate_uncertainty,
        rng=rng,
    )


def render_system_prompt_preview(
    grid_resolution: int,
    cfg: dict[str, Any] | None = None,
    *,
    archive_type: str = "grid",
    n_centroids: int | None = None,
) -> str:
    """Render the system prompt with archive-type placeholders filled in."""
    prompts_dir = resolve_prompts_dir(cfg)
    if archive_type == "cvt":
        path = prompts_dir / CVT_SYSTEM_PROMPT_FILE
        count = _resolve_cvt_n_centroids(n_centroids=n_centroids, cfg=cfg)
        return render_system_prompt_for_archive_type(
            "cvt",
            grid_resolution=grid_resolution,
            n_centroids=count,
            path=path,
        )
    path = prompts_dir / DEFAULT_SYSTEM_PROMPT_FILE
    return render_system_prompt(grid_resolution, path=path)


def _resolve_cvt_n_centroids(
    *,
    n_centroids: int | None,
    cfg: dict[str, Any] | None,
) -> int:
    """Return CVT niche count for prompt preview; never infer from grid resolution."""
    if n_centroids is not None:
        if n_centroids < 1:
            msg = f"n_centroids must be >= 1, got {n_centroids}"
            raise ValueError(msg)
        return int(n_centroids)
    if cfg is not None:
        defaults = cfg.get("defaults")
        if isinstance(defaults, dict) and defaults.get("n_centroids") is not None:
            resolved = int(defaults["n_centroids"])
            if resolved < 1:
                msg = f"defaults.n_centroids must be >= 1, got {resolved}"
                raise ValueError(msg)
            return resolved
    msg = (
        "n_centroids is required when archive_type is 'cvt' "
        "(grid_resolution is not a valid fallback for CVT prompts)"
    )
    raise ValueError(msg)


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

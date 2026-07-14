"""MAP-Elites LLM system prompt loading and rendering."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from worldspace.prompt_files import PROMPTS_DIR, read_prompt
from worldspace.surrogate.types import SurrogatePrediction

ArchiveTypeLiteral = Literal["grid", "cvt"]

DEFAULT_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "map_elites_llm_emitter_system.txt"
DEFAULT_SYSTEM_PROMPT_PATH_CVT = PROMPTS_DIR / "map_elites_llm_emitter_system_cvt.txt"
DEFAULT_USER_PROMPT_PATH = PROMPTS_DIR / "map_elites_llm_emitter_user.txt"
COMPONENTS_USER_PROMPT_PATH = PROMPTS_DIR / "map_elites_llm_emitter_user_components.txt"
CVT_SYSTEM_PROMPT_FILE = "map_elites_llm_emitter_system_cvt.txt"
GRID_SYSTEM_PROMPT_FILE = "map_elites_llm_emitter_system.txt"
COMPONENTS_USER_PROMPT_FILE = "map_elites_llm_emitter_user_components.txt"

SURROGATE_USER_PROMPT_FIELD_NAMES: tuple[str, ...] = (
    "surrogate_stability",
    "surrogate_diversity",
    "surrogate_oscillation",
    "surrogate_topology_interface",
    "surrogate_topology_heterogeneity",
    "surrogate_final_density",
    "surrogate_early_extinction_prob",
    "surrogate_mean",
    "surrogate_uncertainty",
)

__all__ = [
    "COMPONENTS_USER_PROMPT_FILE",
    "COMPONENTS_USER_PROMPT_PATH",
    "CVT_SYSTEM_PROMPT_FILE",
    "DEFAULT_SYSTEM_PROMPT_PATH",
    "DEFAULT_SYSTEM_PROMPT_PATH_CVT",
    "DEFAULT_USER_PROMPT_PATH",
    "GRID_SYSTEM_PROMPT_FILE",
    "SURROGATE_USER_PROMPT_FIELD_NAMES",
    "USER_PROMPT_TEMPLATE",
    "components_user_prompt_path",
    "load_system_prompt_template",
    "load_user_prompt_template",
    "render_cvt_system_prompt",
    "render_system_prompt",
    "render_system_prompt_for_archive_type",
    "render_user_prompt",
    "surrogate_prompt_fields",
    "system_prompt_path_for_archive_type",
    "emitter_prompt_version",
    "system_prompt_version",
    "user_prompt_version",
]


def system_prompt_path_for_archive_type(archive_type: ArchiveTypeLiteral) -> Path:
    """Return the on-disk system prompt template for ``grid`` or ``cvt``."""
    if archive_type == "cvt":
        return DEFAULT_SYSTEM_PROMPT_PATH_CVT
    return DEFAULT_SYSTEM_PROMPT_PATH


def load_system_prompt_template(
    path: str | Path | None = None,
    *,
    archive_type: ArchiveTypeLiteral = "grid",
) -> str:
    """Read the raw system prompt template from disk."""
    if path is None:
        filename = (
            CVT_SYSTEM_PROMPT_FILE if archive_type == "cvt" else GRID_SYSTEM_PROMPT_FILE
        )
        return read_prompt(filename)
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"LLM system prompt not found: {src.resolve()}")
    return src.read_text(encoding="utf-8")


def components_user_prompt_path() -> Path:
    """Return the on-disk path for the component-rich MAP-Elites user prompt."""
    return COMPONENTS_USER_PROMPT_PATH


def load_user_prompt_template(path: str | Path | None = None) -> str:
    """Read the MAP-Elites LLM user prompt template from disk."""
    if path is None:
        return read_prompt("map_elites_llm_emitter_user.txt")
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"LLM user prompt not found: {src.resolve()}")
    return src.read_text(encoding="utf-8")


def surrogate_prompt_fields(prediction: SurrogatePrediction) -> dict[str, float]:
    """Map a surrogate prediction to user-prompt ``.format()`` kwargs."""
    components = prediction.components
    return {
        "surrogate_stability": float(components["stability"]),
        "surrogate_diversity": float(components["diversity"]),
        "surrogate_oscillation": float(components["oscillation_score"]),
        "surrogate_topology_interface": float(components["topology_interface_index"]),
        "surrogate_topology_heterogeneity": float(
            components["topology_window_heterogeneity"]
        ),
        "surrogate_final_density": float(components["final_density"]),
        "surrogate_early_extinction_prob": float(components["early_extinction_prob"]),
        "surrogate_mean": float(prediction.fitness),
        "surrogate_uncertainty": float(prediction.uncertainty),
    }


def render_user_prompt(
    template: str,
    *,
    required_keys: tuple[str, ...] | None = None,
    **kwargs: object,
) -> str:
    """Substitute placeholders into a user prompt template."""
    if required_keys is not None:
        missing = [key for key in required_keys if key not in kwargs]
        if missing:
            missing_text = ", ".join(missing)
            msg = f"Missing user prompt fields: {missing_text}"
            raise KeyError(msg)
    return template.format(**kwargs)


def render_system_prompt(
    grid_resolution: int, *, path: str | Path | None = None
) -> str:
    """Substitute grid size placeholders into the grid system prompt template."""
    if grid_resolution < 1:
        msg = f"grid_resolution must be >= 1, got {grid_resolution}"
        raise ValueError(msg)
    template = load_system_prompt_template(path, archive_type="grid")
    bin_width = 1.0 / float(grid_resolution)
    return template.format(N=grid_resolution, bin_width=f"{bin_width:.6g}")


def render_cvt_system_prompt(
    n_centroids: int, *, path: str | Path | None = None
) -> str:
    """Substitute CVT archive placeholders into the CVT system prompt template."""
    if n_centroids < 1:
        msg = f"n_centroids must be >= 1, got {n_centroids}"
        raise ValueError(msg)
    template = load_system_prompt_template(path, archive_type="cvt")
    return template.format(
        n_centroids=n_centroids,
        n_centroids_minus_one=n_centroids - 1,
    )


def render_system_prompt_for_archive_type(
    archive_type: ArchiveTypeLiteral,
    *,
    grid_resolution: int,
    n_centroids: int,
    path: str | Path | None = None,
) -> str:
    """Render the system prompt for the configured archive type."""
    if archive_type == "cvt":
        return render_cvt_system_prompt(n_centroids, path=path)
    return render_system_prompt(grid_resolution, path=path)


def system_prompt_version(
    path: str | Path | None = None,
    *,
    archive_type: ArchiveTypeLiteral = "grid",
) -> str:
    """Return the first 8 hex digits of the SHA-256 hash of the system prompt file."""
    src = (
        Path(path)
        if path is not None
        else system_prompt_path_for_archive_type(archive_type)
    )
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    return digest[:8]


def user_prompt_version(path: str | Path | None = None) -> str:
    """Return the first 8 hex digits of the SHA-256 hash of the user prompt file."""
    src = Path(path) if path is not None else DEFAULT_USER_PROMPT_PATH
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    return digest[:8]


def emitter_prompt_version(
    *,
    archive_type: ArchiveTypeLiteral = "grid",
    system_path: str | Path | None = None,
    user_path: str | Path | None = None,
) -> str:
    """Composite version tag for LLM emitter archive metadata (system:user hashes)."""
    system = system_prompt_version(system_path, archive_type=archive_type)
    user = user_prompt_version(user_path)
    return f"{system}:{user}"


USER_PROMPT_TEMPLATE = load_user_prompt_template()

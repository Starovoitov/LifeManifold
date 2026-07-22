"""MAP-Elites LLM system prompt loading and rendering."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from worldspace.illuminators.archive import ArchiveElite
from worldspace.illuminators.evaluation import extinction_probability
from worldspace.prompt_files import PROMPTS_DIR, read_prompt
from worldspace.specs.spec import WorldSpec
from worldspace.surrogate.direction_hints import (
    DIRECTION_HINT_EMPTY,
    direction_prompt_fields,
)
from worldspace.surrogate.types import SurrogatePrediction

if TYPE_CHECKING:
    from worldspace.surrogate.model import SurrogateModel

ArchiveTypeLiteral = Literal["grid", "cvt"]

DEFAULT_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "map_elites_llm_emitter_system.txt"
DEFAULT_SYSTEM_PROMPT_PATH_CVT = PROMPTS_DIR / "map_elites_llm_emitter_system_cvt.txt"
DEFAULT_USER_PROMPT_PATH = PROMPTS_DIR / "map_elites_llm_emitter_user.txt"
COMPONENTS_USER_PROMPT_PATH = PROMPTS_DIR / "map_elites_llm_emitter_user_components.txt"
PARENT_USER_PROMPT_PATH = PROMPTS_DIR / "map_elites_llm_emitter_user_parent_hints.txt"
DIRECTION_USER_PROMPT_PATH = PROMPTS_DIR / "map_elites_llm_emitter_user_direction.txt"
CVT_SYSTEM_PROMPT_FILE = "map_elites_llm_emitter_system_cvt.txt"
GRID_SYSTEM_PROMPT_FILE = "map_elites_llm_emitter_system.txt"
COMPONENTS_USER_PROMPT_FILE = "map_elites_llm_emitter_user_components.txt"
PARENT_USER_PROMPT_FILE = "map_elites_llm_emitter_user_parent_hints.txt"
DIRECTION_USER_PROMPT_FILE = "map_elites_llm_emitter_user_direction.txt"

PARENT_HINT_EMPTY = "Parent cell: empty (no archive elite in this niche yet)."

PARENT_USER_PROMPT_FIELD_NAMES: tuple[str, ...] = ("parent_hint_block",)
DIRECTION_USER_PROMPT_FIELD_NAMES: tuple[str, ...] = ("direction_hint_block",)

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
    "DIRECTION_USER_PROMPT_FIELD_NAMES",
    "DIRECTION_USER_PROMPT_FILE",
    "DIRECTION_USER_PROMPT_PATH",
    "GRID_SYSTEM_PROMPT_FILE",
    "PARENT_HINT_EMPTY",
    "PARENT_USER_PROMPT_FIELD_NAMES",
    "PARENT_USER_PROMPT_FILE",
    "PARENT_USER_PROMPT_PATH",
    "SURROGATE_USER_PROMPT_FIELD_NAMES",
    "USER_PROMPT_TEMPLATE",
    "components_user_prompt_path",
    "direction_user_prompt_path",
    "load_system_prompt_template",
    "load_user_prompt_template",
    "parent_prompt_fields",
    "parent_user_prompt_path",
    "resolve_direction_prompt_fields",
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


def parent_user_prompt_path() -> Path:
    """Return the on-disk path for the parent-metrics MAP-Elites user prompt."""
    return PARENT_USER_PROMPT_PATH


def direction_user_prompt_path() -> Path:
    """Return the on-disk path for the direction-hints MAP-Elites user prompt."""
    return DIRECTION_USER_PROMPT_PATH


def resolve_direction_prompt_fields(
    template: str,
    *,
    parent_world_spec: WorldSpec | None = None,
    surrogate_model: SurrogateModel | None = None,
    use_soft_extinction: bool = False,
    extinction_gate_threshold: float = 0.5,
) -> dict[str, str]:
    """Build direction hint kwargs when the user template includes the placeholder."""
    if "{direction_hint_block}" not in template:
        return {}
    if parent_world_spec is None or surrogate_model is None:
        return {"direction_hint_block": DIRECTION_HINT_EMPTY}
    return direction_prompt_fields(
        parent_world_spec,
        surrogate_model,
        use_soft_extinction=use_soft_extinction,
        extinction_gate_threshold=extinction_gate_threshold,
    )


def load_user_prompt_template(path: str | Path | None = None) -> str:
    """Read the MAP-Elites LLM user prompt template from disk."""
    if path is None:
        return read_prompt("map_elites_llm_emitter_user.txt")
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"LLM user prompt not found: {src.resolve()}")
    return src.read_text(encoding="utf-8")


def _fmt_prompt_metric(value: float, *, precision: int = 3) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{precision}f}"


def _format_parent_hint_block(
    *,
    fitness: float,
    stability: float,
    diversity: float,
    final_density: float,
    early_extinction_prob: float,
    oscillation: float,
    topology_interface: float,
    topology_heterogeneity: float,
) -> str:
    return (
        "Parent cell (observed from simulation):\n"
        f"  fitness: {_fmt_prompt_metric(fitness)}\n"
        f"  stability: {_fmt_prompt_metric(stability)}\n"
        f"  diversity: {_fmt_prompt_metric(diversity)}\n"
        f"  final_density: {_fmt_prompt_metric(final_density)}\n"
        f"  early_extinction_prob: {_fmt_prompt_metric(early_extinction_prob)}\n"
        f"  oscillation_score: {_fmt_prompt_metric(oscillation)}\n"
        f"  topology_interface_index: {_fmt_prompt_metric(topology_interface)}\n"
        f"  topology_window_heterogeneity: "
        f"{_fmt_prompt_metric(topology_heterogeneity)}"
    )


def parent_prompt_fields(elite: ArchiveElite | None) -> dict[str, str]:
    """Map the current archive elite to parent hint-block ``.format()`` kwargs."""
    if elite is None or elite.world_spec is None:
        return {"parent_hint_block": PARENT_HINT_EMPTY}
    measures = elite.measures or {}
    stability = float(measures.get("stability", float("nan")))
    diversity = float(measures.get("diversity", float("nan")))
    fitness = float(elite.fitness)
    metrics = elite.metrics
    if metrics is not None:
        final_density = float(metrics.density_mean)
        oscillation = float(metrics.oscillation_score)
        topology_interface = float(metrics.topology_interface_index)
        topology_heterogeneity = float(metrics.topology_window_heterogeneity)
    else:
        final_density = float("nan")
        oscillation = float("nan")
        topology_interface = float("nan")
        topology_heterogeneity = float("nan")
    early_extinction_prob = (
        extinction_probability(final_density)
        if math.isfinite(final_density)
        else float("nan")
    )
    return {
        "parent_hint_block": _format_parent_hint_block(
            fitness=fitness,
            stability=stability,
            diversity=diversity,
            final_density=final_density,
            early_extinction_prob=early_extinction_prob,
            oscillation=oscillation,
            topology_interface=topology_interface,
            topology_heterogeneity=topology_heterogeneity,
        )
    }


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

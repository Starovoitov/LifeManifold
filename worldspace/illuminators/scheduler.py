"""MAP-Elites scheduler YAML and target-bin selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from worldspace.illuminators.archive import GridArchive
from worldspace.illuminators.evaluation import bin_center

EmitterKind = Literal["random", "genetic", "llm"]
_SCHEDULER_SCHEMA_VERSION = "1.2"
_DEFAULT_SPECS_DIR = Path(__file__).resolve().parent.parent / "specs"
DEFAULT_SCHEDULER_PATH = _DEFAULT_SPECS_DIR / "map_elites_scheduler.yaml"
DEFAULT_MINI_SCHEDULER_PATH = _DEFAULT_SPECS_DIR / "map_elites_scheduler_mini.yaml"

__all__ = [
    "DEFAULT_MINI_SCHEDULER_PATH",
    "DEFAULT_SCHEDULER_PATH",
    "EmitterKind",
    "RunCounters",
    "SchedulerConfig",
    "TargetBin",
    "load_scheduler",
    "resolve_emitter_for_slot",
    "resolve_emitter_kind",
    "select_target_bin",
    "slot_emitter_for_candidate",
]


@dataclass(frozen=True)
class SchedulerConfig:
    """Validated MAP-Elites scheduler settings."""

    schema_version: str
    iterations: int
    batch_size: int
    grid_resolution: int
    early_extinction_step: int
    min_steps: int
    batch_emitters: tuple[EmitterKind, ...]
    initial_random_candidates: int
    llm_enabled: bool
    surrogate_enabled: bool
    surrogate_stub_mean: float
    surrogate_stub_uncertainty: float
    genetic_mutation_scale: float


@dataclass(frozen=True)
class TargetBin:
    """Archive cell chosen for the next candidate and its BC niche center."""

    bin: tuple[int, int]
    target_stability: float
    target_diversity: float


@dataclass
class RunCounters:
    """Global illuminator counters persisted across iterations."""

    candidates_evaluated: int = 0

    def record_evaluation(self) -> None:
        """Increment after each candidate is evaluated (accepted or rejected)."""
        self.candidates_evaluated += 1


def load_scheduler(
    path: str | Path,
    *,
    iterations_override: int | None = None,
) -> SchedulerConfig:
    """Load and validate scheduler YAML from ``path``."""
    src = Path(path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"Scheduler YAML not found: {src.resolve()}")
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {src}")
    try:
        doc = _MapElitesSchedulerYaml.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid scheduler YAML at {src}:\n{exc}") from exc
    if doc.schema_version != _SCHEDULER_SCHEMA_VERSION:
        msg = (
            f"schema_version must be {_SCHEDULER_SCHEMA_VERSION!r}, "
            f"got {doc.schema_version!r}"
        )
        raise ValueError(msg)
    if len(doc.batch_emitters) != doc.batch_size:
        msg = (
            f"batch_emitters length ({len(doc.batch_emitters)}) "
            f"must equal batch_size ({doc.batch_size})"
        )
        raise ValueError(msg)
    iterations = doc.iterations if iterations_override is None else iterations_override
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    return SchedulerConfig(
        schema_version=doc.schema_version,
        iterations=iterations,
        batch_size=doc.batch_size,
        grid_resolution=doc.grid_resolution,
        early_extinction_step=doc.early_extinction_step,
        min_steps=doc.min_steps,
        batch_emitters=tuple(doc.batch_emitters),
        initial_random_candidates=doc.initial_random_candidates,
        llm_enabled=doc.llm.enabled,
        surrogate_enabled=doc.surrogate.enabled,
        surrogate_stub_mean=doc.surrogate.stub_mean,
        surrogate_stub_uncertainty=doc.surrogate.stub_uncertainty,
        genetic_mutation_scale=doc.genetic.mutation_scale,
    )


def select_target_bin(
    archive: GridArchive,
    rng: np.random.Generator,
) -> TargetBin:
    """Choose a target archive cell and BC niche center for the next candidate."""
    resolution = archive.resolution
    if archive.filled_count() == 0:
        flat_index = int(rng.integers(0, resolution * resolution))
        i, j = divmod(flat_index, resolution)
    else:
        boundary = _boundary_bins(archive)
        if boundary:
            i, j = _min_fitness_bin(boundary, archive)
        else:
            flat_index = int(rng.integers(0, resolution * resolution))
            i, j = divmod(flat_index, resolution)
    stability, diversity = bin_center(i, j, resolution)
    return TargetBin(bin=(i, j), target_stability=stability, target_diversity=diversity)


def slot_emitter_for_candidate(
    config: SchedulerConfig,
    candidate_id: int,
) -> EmitterKind:
    """Return the YAML emitter for batch slot ``candidate_id``."""
    if candidate_id < 0 or candidate_id >= config.batch_size:
        msg = f"candidate_id must be in [0, {config.batch_size}), got {candidate_id}"
        raise ValueError(msg)
    return config.batch_emitters[candidate_id]


def resolve_emitter_kind(
    config: SchedulerConfig,
    *,
    slot_emitter: EmitterKind,
    candidates_evaluated: int,
) -> EmitterKind:
    """Apply initial-fill and disabled-LLM overrides before using the YAML slot."""
    if candidates_evaluated < config.initial_random_candidates:
        return "random"
    if slot_emitter == "llm" and not config.llm_enabled:
        return "random"
    return slot_emitter


def resolve_emitter_for_slot(
    config: SchedulerConfig,
    *,
    candidate_id: int,
    candidates_evaluated: int,
) -> EmitterKind:
    """Resolve the emitter for one batch slot (YAML slot + initial random phase)."""
    slot_emitter = slot_emitter_for_candidate(config, candidate_id)
    return resolve_emitter_kind(
        config,
        slot_emitter=slot_emitter,
        candidates_evaluated=candidates_evaluated,
    )


class _LlmSchedulerBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


class _SurrogateSchedulerBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    stub_mean: float = Field(..., ge=0.0, le=1.0)
    stub_uncertainty: float = Field(..., ge=0.0)


class _GeneticSchedulerBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mutation_scale: float = Field(default=0.02, ge=0.0)


class _MapElitesSchedulerYaml(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    iterations: int = Field(..., ge=1)
    batch_size: int = Field(..., ge=1)
    grid_resolution: int = Field(..., ge=1)
    early_extinction_step: int = Field(..., ge=1)
    min_steps: int = Field(..., ge=200)
    batch_emitters: list[EmitterKind]
    initial_random_candidates: int = Field(..., ge=0)
    llm: _LlmSchedulerBlock
    surrogate: _SurrogateSchedulerBlock
    genetic: _GeneticSchedulerBlock = Field(default_factory=_GeneticSchedulerBlock)


def _boundary_bins(archive: GridArchive) -> list[tuple[int, int]]:
    resolution = archive.resolution
    boundary: list[tuple[int, int]] = []
    for i in range(resolution):
        for j in range(resolution):
            if archive.is_empty(i, j):
                continue
            for ni, nj in _cardinal_neighbors(i, j, resolution):
                if archive.is_empty(ni, nj):
                    boundary.append((i, j))
                    break
    return boundary


def _min_fitness_bin(
    bins: list[tuple[int, int]],
    archive: GridArchive,
) -> tuple[int, int]:
    best: tuple[int, int] | None = None
    best_fitness = float("inf")
    for i, j in bins:
        elite = archive.get(i, j)
        if elite is None:
            continue
        if elite.fitness < best_fitness or (
            elite.fitness == best_fitness and (best is None or (i, j) < best)
        ):
            best = (i, j)
            best_fitness = elite.fitness
    if best is None:
        msg = "boundary bins must contain elites"
        raise RuntimeError(msg)
    return best


def _cardinal_neighbors(i: int, j: int, resolution: int) -> tuple[tuple[int, int], ...]:
    neighbors: list[tuple[int, int]] = []
    if i > 0:
        neighbors.append((i - 1, j))
    if i + 1 < resolution:
        neighbors.append((i + 1, j))
    if j > 0:
        neighbors.append((i, j - 1))
    if j + 1 < resolution:
        neighbors.append((i, j + 1))
    return tuple(neighbors)

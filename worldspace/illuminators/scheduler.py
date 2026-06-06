"""MAP-Elites scheduler YAML and target-bin selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from worldspace.illuminators.archive import GridArchive
from worldspace.illuminators.evaluation import bin_center
from worldspace.illuminators.grid_neighbors import cardinal_neighbors_bounded
from worldspace.specs.spec import WorldSpec
from worldspace.surrogate.acquisition_config import (
    AcquisitionConfig,
    AcquisitionPolicyName,
    DEFAULT_SURROGATE_ARCHIVE_PATH,
    RetrainConfig,
)
from worldspace.surrogate.types import SurrogateConfig, SurrogateProtocol

logger = logging.getLogger(__name__)

EmitterKind = Literal["random", "genetic", "llm"]
_SCHEDULER_SCHEMA_VERSION = "1.2"
_DEFAULT_SPECS_DIR = Path(__file__).resolve().parent.parent / "specs"
DEFAULT_SCHEDULER_PATH = _DEFAULT_SPECS_DIR / "map_elites_scheduler.yaml"
DEFAULT_MINI_SCHEDULER_PATH = _DEFAULT_SPECS_DIR / "map_elites_scheduler_mini.yaml"
DEFAULT_NIGHTLY_SCHEDULER_PATH = (
    _DEFAULT_SPECS_DIR / "map_elites_scheduler_nightly.yaml"
)
DEFAULT_NIGHTLY_SURROGATE_SCHEDULER_PATH = (
    _DEFAULT_SPECS_DIR / "map_elites_scheduler_nightly_surrogate.yaml"
)
DEFAULT_GITHUB_LLM_SCHEDULER_PATH = (
    _DEFAULT_SPECS_DIR / "map_elites_scheduler_github_llm.yaml"
)
DEFAULT_QWEN_LLM_SPEC_PATH = _DEFAULT_SPECS_DIR / "llm_world_generator_qwen.yaml"

__all__ = [
    "AcquisitionConfig",
    "DEFAULT_MINI_SCHEDULER_PATH",
    "DEFAULT_NIGHTLY_SCHEDULER_PATH",
    "DEFAULT_GITHUB_LLM_SCHEDULER_PATH",
    "DEFAULT_NIGHTLY_SURROGATE_SCHEDULER_PATH",
    "DEFAULT_QWEN_LLM_SPEC_PATH",
    "DEFAULT_SCHEDULER_PATH",
    "EmitterKind",
    "RetrainConfig",
    "RunCounters",
    "SchedulerConfig",
    "TargetBin",
    "load_scheduler",
    "resolve_emitter_for_slot",
    "resolve_emitter_kind",
    "resolve_surrogate_stub",
    "select_target_bin",
    "slot_emitter_for_candidate",
    "surrogate_config_from_scheduler",
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
    surrogate_model_type: str
    surrogate_checkpoint: str | None
    surrogate_buffer_path: str
    surrogate_stub_mean: float
    surrogate_stub_uncertainty: float
    genetic_mutation_scale: float
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    surrogate_archive_path: str = field(
        default=DEFAULT_SURROGATE_ARCHIVE_PATH,
    )
    retrain: RetrainConfig = field(default_factory=RetrainConfig)
    surrogate_calibration: str | None = None


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
        """Increment after each completed real simulation (skipped slots excluded)."""
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
    acquisition = _acquisition_config_from_yaml(doc.surrogate.acquisition)
    retrain = _retrain_config_from_yaml(doc.surrogate.retrain)
    archive_path = (
        doc.surrogate.surrogate_archive_path or DEFAULT_SURROGATE_ARCHIVE_PATH
    )
    config = SchedulerConfig(
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
        surrogate_model_type=doc.surrogate.model_type,
        surrogate_checkpoint=doc.surrogate.checkpoint,
        surrogate_buffer_path=doc.surrogate.buffer_path,
        surrogate_stub_mean=doc.surrogate.stub_mean,
        surrogate_stub_uncertainty=doc.surrogate.stub_uncertainty,
        genetic_mutation_scale=doc.genetic.mutation_scale,
        acquisition=acquisition,
        surrogate_archive_path=archive_path,
        retrain=retrain,
        surrogate_calibration=_normalize_calibration_path(doc.surrogate.calibration),
    )
    return _normalize_acquisition_config(config)


def _normalize_calibration_path(value: str | None) -> str | None:
    """Return a non-empty calibration path or None when calibration is disabled."""
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


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


def surrogate_config_from_scheduler(
    config: SchedulerConfig,
    *,
    require_quality_gate: bool = False,
) -> SurrogateConfig:
    """Build runtime surrogate settings from scheduler YAML fields."""
    from worldspace.surrogate.types import ModelType

    model_type: ModelType
    if config.surrogate_model_type == "mlp":
        model_type = "mlp"
    else:
        model_type = "lightgbm"
    return SurrogateConfig(
        enabled=config.surrogate_enabled,
        model_type=model_type,
        checkpoint=config.surrogate_checkpoint,
        stub_mean=config.surrogate_stub_mean,
        stub_uncertainty=config.surrogate_stub_uncertainty,
        calibration=config.surrogate_calibration,
        require_quality_gate=require_quality_gate,
    )


def resolve_surrogate_stub(
    config: SchedulerConfig,
    surrogate: SurrogateProtocol,
    world_spec: WorldSpec,
) -> tuple[float, float]:
    """Return surrogate fitness and uncertainty for LLM user prompts."""
    if not config.surrogate_enabled:
        return (config.surrogate_stub_mean, config.surrogate_stub_uncertainty)
    prediction = surrogate.predict(world_spec)
    return (float(prediction.fitness), float(prediction.uncertainty))


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


class _AcquisitionYamlBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["off", "shadow", "filter"] = "off"
    policy: Literal["threshold_gate", "ucb_promote"] = "threshold_gate"
    min_predicted_fitness: float = Field(default=0.25, ge=0.0, le=1.0)
    max_uncertainty_to_skip: float = Field(default=0.40, ge=0.0)
    never_skip_empty_bin: bool = True
    exploration_weight: float = Field(default=0.15, ge=0.0)


class _RetrainYamlBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    every_iterations: int = Field(default=50, ge=1)
    min_new_buffer_rows: int = Field(default=500, ge=0)


class _SurrogateSchedulerBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    model_type: str = Field(default="lightgbm")
    checkpoint: str | None = Field(default="artifacts/surrogate/checkpoints/latest.pkl")
    buffer_path: str = Field(default="artifacts/surrogate/buffer.jsonl")
    stub_mean: float = Field(..., ge=0.0, le=1.0)
    stub_uncertainty: float = Field(..., ge=0.0)
    calibration: str | None = None
    acquisition: _AcquisitionYamlBlock | None = None
    retrain: _RetrainYamlBlock | None = None
    surrogate_archive_path: str | None = None

    @field_validator("calibration", mode="before")
    @classmethod
    def _normalize_calibration_field(cls, value: object) -> str | None:
        """Treat false, null, and blank as disabled (raw ensemble uncertainty)."""
        if value is None or value is False:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value  # type: ignore[return-value]


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


def _acquisition_config_from_yaml(
    block: _AcquisitionYamlBlock | None,
) -> AcquisitionConfig:
    if block is None:
        return AcquisitionConfig()
    return AcquisitionConfig(
        mode=block.mode,
        policy=block.policy,
        min_predicted_fitness=block.min_predicted_fitness,
        max_uncertainty_to_skip=block.max_uncertainty_to_skip,
        never_skip_empty_bin=block.never_skip_empty_bin,
        exploration_weight=block.exploration_weight,
    )


def _retrain_config_from_yaml(block: _RetrainYamlBlock | None) -> RetrainConfig:
    if block is None:
        return RetrainConfig()
    return RetrainConfig(
        enabled=block.enabled,
        every_iterations=block.every_iterations,
        min_new_buffer_rows=block.min_new_buffer_rows,
    )


def _normalize_acquisition_config(config: SchedulerConfig) -> SchedulerConfig:
    """Apply safe defaults when acquisition settings are inconsistent."""
    acquisition = config.acquisition
    policy: AcquisitionPolicyName = acquisition.policy
    mode = acquisition.mode

    if policy == "ucb_promote":
        logger.warning(
            "acquisition policy %r is not implemented yet; using threshold_gate",
            policy,
        )
        policy = "threshold_gate"

    if mode == "filter" and not config.surrogate_enabled:
        logger.warning(
            "acquisition.mode filter requires surrogate.enabled; forcing mode off",
        )
        mode = "off"

    if policy == acquisition.policy and mode == acquisition.mode:
        return config
    return replace(
        config,
        acquisition=replace(acquisition, policy=policy, mode=mode),
    )


def _boundary_bins(archive: GridArchive) -> list[tuple[int, int]]:
    resolution = archive.resolution
    boundary: list[tuple[int, int]] = []
    for i in range(resolution):
        for j in range(resolution):
            if archive.is_empty(i, j):
                continue
            for ni, nj in cardinal_neighbors_bounded(i, j, resolution):
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

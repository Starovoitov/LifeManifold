"""MAP-Elites scheduler YAML and target-bin selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from worldspace.illuminators.archive import DEFAULT_GRID_RESOLUTION, GridArchive
from worldspace.illuminators.archive_protocol import ArchiveProtocol
from worldspace.illuminators.cvt import DEFAULT_LLOYD_ITERATIONS
from worldspace.simulator_perf import (
    DEFAULT_SIMULATOR_PERFORMANCE,
    SimulatorPerformanceOptions,
    resolve_simulator_performance,
    validate_simulator_performance,
)
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
ArchiveType = Literal["grid", "cvt"]
TargetSelectionStrategy = Literal["min_fitness_frontier", "uniform_frontier"]
DEFAULT_TARGET_SELECTION: TargetSelectionStrategy = "min_fitness_frontier"
_SCHEDULER_SCHEMA_VERSION = "1.2"
_SCHEDULER_SCHEMA_VERSIONS = ("1.2", "1.3")
_DEFAULT_SPECS_DIR = Path(__file__).resolve().parent.parent / "specs"
DEFAULT_SCHEDULER_PATH = _DEFAULT_SPECS_DIR / "map_elites_scheduler.yaml"
DEFAULT_MINI_SCHEDULER_PATH = _DEFAULT_SPECS_DIR / "map_elites_scheduler_mini.yaml"
DEFAULT_MINI_CVT_SCHEDULER_PATH = (
    _DEFAULT_SPECS_DIR / "map_elites_scheduler_mini_cvt.yaml"
)
DEFAULT_NIGHTLY_CVT_SCHEDULER_PATH = (
    _DEFAULT_SPECS_DIR / "map_elites_scheduler_nightly_cvt.yaml"
)
DEFAULT_NIGHTLY_SURROGATE_CVT_SCHEDULER_PATH = (
    _DEFAULT_SPECS_DIR / "map_elites_scheduler_nightly_surrogate_cvt.yaml"
)
DEFAULT_NIGHTLY_SCHEDULER_PATH = (
    _DEFAULT_SPECS_DIR / "map_elites_scheduler_nightly.yaml"
)
DEFAULT_NIGHTLY_SURROGATE_SCHEDULER_PATH = (
    _DEFAULT_SPECS_DIR / "map_elites_scheduler_nightly_surrogate.yaml"
)
DEFAULT_GITHUB_LLM_SCHEDULER_PATH = (
    _DEFAULT_SPECS_DIR / "map_elites_scheduler_github_llm_cvt.yaml"
)
DEFAULT_GITHUB_LLM_GRID_SCHEDULER_PATH = (
    _DEFAULT_SPECS_DIR / "map_elites_scheduler_github_llm.yaml"
)
DEFAULT_QWEN_LLM_SPEC_PATH = _DEFAULT_SPECS_DIR / "llm_world_generator_qwen.yaml"

__all__ = [
    "AcquisitionConfig",
    "DEFAULT_MINI_CVT_SCHEDULER_PATH",
    "DEFAULT_MINI_SCHEDULER_PATH",
    "DEFAULT_NIGHTLY_CVT_SCHEDULER_PATH",
    "DEFAULT_NIGHTLY_SCHEDULER_PATH",
    "DEFAULT_GITHUB_LLM_GRID_SCHEDULER_PATH",
    "DEFAULT_GITHUB_LLM_SCHEDULER_PATH",
    "DEFAULT_NIGHTLY_SURROGATE_CVT_SCHEDULER_PATH",
    "DEFAULT_NIGHTLY_SURROGATE_SCHEDULER_PATH",
    "DEFAULT_QWEN_LLM_SPEC_PATH",
    "DEFAULT_SCHEDULER_PATH",
    "EmitterKind",
    "RetrainConfig",
    "RunCounters",
    "SchedulerConfig",
    "DEFAULT_TARGET_SELECTION",
    "TargetBin",
    "TargetCell",
    "TargetSelectionStrategy",
    "load_scheduler",
    "resolve_emitter_for_slot",
    "resolve_emitter_kind",
    "resolve_surrogate_stub",
    "select_target_bin",
    "select_target_cell",
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
    surrogate_use_soft_extinction: bool = False
    surrogate_extinction_gate_threshold: float = 0.5
    archive_type: ArchiveType = "grid"
    n_centroids: int = DEFAULT_GRID_RESOLUTION * DEFAULT_GRID_RESOLUTION
    cvt_seed: int = 0
    lloyd_iterations: int = DEFAULT_LLOYD_ITERATIONS
    performance: SimulatorPerformanceOptions = field(
        default_factory=lambda: DEFAULT_SIMULATOR_PERFORMANCE
    )
    target_selection: TargetSelectionStrategy = DEFAULT_TARGET_SELECTION

    @property
    def n_cells(self) -> int:
        """Number of behavioral niches for the configured archive."""
        if self.archive_type == "grid":
            return self.grid_resolution * self.grid_resolution
        return self.n_centroids


@dataclass(frozen=True)
class TargetCell:
    """Archive niche chosen for the next candidate and its BC center."""

    cell_id: int
    target_stability: float
    target_diversity: float
    bin_ij: tuple[int, int]


@dataclass(frozen=True)
class TargetBin:
    """Archive cell chosen for the next candidate and its BC niche center."""

    bin: tuple[int, int]
    target_stability: float
    target_diversity: float

    @classmethod
    def from_target_cell(cls, cell: TargetCell) -> TargetBin:
        """Bridge ``TargetCell`` to acquisition and logging APIs that use bins."""
        return cls(
            bin=cell.bin_ij,
            target_stability=cell.target_stability,
            target_diversity=cell.target_diversity,
        )


@dataclass
class RunCounters:
    """Global illuminator counters persisted across iterations."""

    candidates_evaluated: int = 0
    llm_emit_attempts: int = 0
    llm_emit_fallbacks: int = 0
    emit_llm_seconds: float = 0.0
    eval_seconds: float = 0.0

    def record_evaluation(self) -> None:
        """Increment after each completed real simulation (skipped slots excluded)."""
        self.candidates_evaluated += 1

    def record_llm_emit(self, *, fallback: bool) -> None:
        """Increment after each LLM emit slot (including parse/API fallbacks)."""
        self.llm_emit_attempts += 1
        if fallback:
            self.llm_emit_fallbacks += 1


@dataclass(frozen=True)
class _ArchiveSettings:
    archive_type: ArchiveType
    resolution: int
    n_centroids: int
    cvt_seed: int
    lloyd_iterations: int


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
    if doc.schema_version not in _SCHEDULER_SCHEMA_VERSIONS:
        msg = (
            f"schema_version must be one of {_SCHEDULER_SCHEMA_VERSIONS!r}, "
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
    archive_settings = _archive_settings_from_yaml(doc)
    performance = resolve_simulator_performance(
        doc.performance.model_dump() if doc.performance is not None else None
    )
    validate_simulator_performance(performance)
    config = SchedulerConfig(
        schema_version=doc.schema_version,
        iterations=iterations,
        batch_size=doc.batch_size,
        grid_resolution=archive_settings.resolution,
        early_extinction_step=doc.early_extinction_step,
        min_steps=doc.min_steps,
        target_selection=doc.target_selection,
        batch_emitters=tuple(doc.batch_emitters),
        initial_random_candidates=doc.initial_random_candidates,
        llm_enabled=doc.llm.enabled,
        surrogate_enabled=doc.surrogate.enabled,
        surrogate_model_type=doc.surrogate.model_type,
        surrogate_checkpoint=doc.surrogate.checkpoint,
        surrogate_buffer_path=doc.surrogate.buffer_path,
        surrogate_stub_mean=doc.surrogate.stub_mean,
        surrogate_stub_uncertainty=doc.surrogate.stub_uncertainty,
        surrogate_use_soft_extinction=doc.surrogate.use_soft_extinction,
        surrogate_extinction_gate_threshold=doc.surrogate.extinction_gate_threshold,
        genetic_mutation_scale=doc.genetic.mutation_scale,
        acquisition=acquisition,
        surrogate_archive_path=archive_path,
        retrain=retrain,
        surrogate_calibration=_normalize_calibration_path(doc.surrogate.calibration),
        archive_type=archive_settings.archive_type,
        n_centroids=archive_settings.n_centroids,
        cvt_seed=archive_settings.cvt_seed,
        lloyd_iterations=archive_settings.lloyd_iterations,
        performance=performance,
    )
    return _normalize_acquisition_config(config)


def select_target_cell(
    archive: ArchiveProtocol,
    rng: np.random.Generator,
    *,
    target_selection: TargetSelectionStrategy = DEFAULT_TARGET_SELECTION,
) -> TargetCell:
    """Choose a target archive niche and BC center for the next candidate."""
    if archive.filled_count() == 0:
        cell_id = int(rng.integers(0, archive.n_cells))
    else:
        frontier = _frontier_cell_ids(archive)
        if frontier:
            cell_id = _select_frontier_cell(
                frontier,
                archive,
                rng,
                target_selection=target_selection,
            )
        else:
            cell_id = int(rng.integers(0, archive.n_cells))
    stability, diversity = archive.cell_center(cell_id)
    bin_ij = archive.bin_from_cell_id(cell_id)
    return TargetCell(
        cell_id=cell_id,
        target_stability=stability,
        target_diversity=diversity,
        bin_ij=bin_ij,
    )


def select_target_bin(
    archive: GridArchive,
    rng: np.random.Generator,
    *,
    target_selection: TargetSelectionStrategy = DEFAULT_TARGET_SELECTION,
) -> TargetBin:
    """Choose a target archive cell and BC niche center for the next candidate."""
    target = select_target_cell(
        archive,
        rng,
        target_selection=target_selection,
    )
    i, j = target.bin_ij
    return TargetBin(
        bin=(i, j),
        target_stability=target.target_stability,
        target_diversity=target.target_diversity,
    )


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
        use_soft_extinction=config.surrogate_use_soft_extinction,
        extinction_gate_threshold=config.surrogate_extinction_gate_threshold,
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
    model_type: str = Field(default="mlp")
    checkpoint: str | None = Field(
        default="artifacts/surrogate/checkpoints/nightly_v3_mc_d005.pkl"
    )
    buffer_path: str = Field(default="artifacts/surrogate/buffer.jsonl")
    stub_mean: float = Field(..., ge=0.0, le=1.0)
    stub_uncertainty: float = Field(..., ge=0.0)
    use_soft_extinction: bool = False
    extinction_gate_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
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


class _PerformanceYamlBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    numba_simulator: bool = False
    numba_cache: bool = True
    parallel_eval: bool = False
    parallel_workers: int = Field(
        default=0,
        ge=0,
        description="0 = os.cpu_count() (auto), capped by batch_size when parallel_eval is on",
    )
    verify_against_reference: bool = False
    llm_parallel_emit: bool = False
    log_iteration_timing: bool = False


class _GridArchiveYamlBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["grid"]
    resolution: int = Field(..., ge=1)


class _CvtArchiveYamlBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["cvt"]
    n_centroids: int = Field(..., ge=1)
    cvt_seed: int = 0
    lloyd_iterations: int = Field(default=DEFAULT_LLOYD_ITERATIONS, ge=1)


_ArchiveYamlBlock = Annotated[
    _GridArchiveYamlBlock | _CvtArchiveYamlBlock,
    Field(discriminator="type"),
]


class _MapElitesSchedulerYaml(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    iterations: int = Field(..., ge=1)
    batch_size: int = Field(..., ge=1)
    grid_resolution: int | None = Field(default=None, ge=1)
    archive: _ArchiveYamlBlock | None = None
    early_extinction_step: int = Field(..., ge=1)
    min_steps: int = Field(..., ge=200)
    target_selection: TargetSelectionStrategy = DEFAULT_TARGET_SELECTION
    batch_emitters: list[EmitterKind]
    initial_random_candidates: int = Field(..., ge=0)
    llm: _LlmSchedulerBlock
    surrogate: _SurrogateSchedulerBlock
    genetic: _GeneticSchedulerBlock = Field(default_factory=_GeneticSchedulerBlock)
    performance: _PerformanceYamlBlock | None = None

    @model_validator(mode="after")
    def _validate_archive_fields(self) -> _MapElitesSchedulerYaml:
        if self.schema_version == _SCHEDULER_SCHEMA_VERSION:
            if self.grid_resolution is None:
                msg = "grid_resolution is required for schema_version 1.2"
                raise ValueError(msg)
            if self.archive is not None:
                msg = "archive block is not supported for schema_version 1.2"
                raise ValueError(msg)
            return self
        if self.schema_version == "1.3":
            if self.archive is None and self.grid_resolution is None:
                msg = (
                    "schema_version 1.3 requires archive block or grid_resolution "
                    "for implicit grid archive"
                )
                raise ValueError(msg)
            if isinstance(self.archive, _GridArchiveYamlBlock):
                if self.grid_resolution is not None and (
                    self.grid_resolution != self.archive.resolution
                ):
                    msg = (
                        "grid_resolution conflicts with archive.resolution; "
                        "use one source of truth"
                    )
                    raise ValueError(msg)
            return self
        return self


def _archive_settings_from_yaml(doc: _MapElitesSchedulerYaml) -> _ArchiveSettings:
    if doc.schema_version == _SCHEDULER_SCHEMA_VERSION:
        assert doc.grid_resolution is not None
        return _ArchiveSettings(
            archive_type="grid",
            resolution=doc.grid_resolution,
            n_centroids=doc.grid_resolution * doc.grid_resolution,
            cvt_seed=0,
            lloyd_iterations=DEFAULT_LLOYD_ITERATIONS,
        )

    if isinstance(doc.archive, _CvtArchiveYamlBlock):
        return _ArchiveSettings(
            archive_type="cvt",
            resolution=DEFAULT_GRID_RESOLUTION,
            n_centroids=doc.archive.n_centroids,
            cvt_seed=doc.archive.cvt_seed,
            lloyd_iterations=doc.archive.lloyd_iterations,
        )

    if isinstance(doc.archive, _GridArchiveYamlBlock):
        return _ArchiveSettings(
            archive_type="grid",
            resolution=doc.archive.resolution,
            n_centroids=doc.archive.resolution * doc.archive.resolution,
            cvt_seed=0,
            lloyd_iterations=DEFAULT_LLOYD_ITERATIONS,
        )

    assert doc.grid_resolution is not None
    return _ArchiveSettings(
        archive_type="grid",
        resolution=doc.grid_resolution,
        n_centroids=doc.grid_resolution * doc.grid_resolution,
        cvt_seed=0,
        lloyd_iterations=DEFAULT_LLOYD_ITERATIONS,
    )


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


def _normalize_calibration_path(value: str | None) -> str | None:
    """Return a non-empty calibration path or None when calibration is disabled."""
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


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


def _select_frontier_cell(
    frontier: list[int],
    archive: ArchiveProtocol,
    rng: np.random.Generator,
    *,
    target_selection: TargetSelectionStrategy,
) -> int:
    if not frontier:
        msg = "frontier must be non-empty"
        raise ValueError(msg)
    if target_selection == "uniform_frontier":
        return int(frontier[int(rng.integers(0, len(frontier)))])
    return _min_fitness_cell(frontier, archive)


def _frontier_cell_ids(archive: ArchiveProtocol) -> list[int]:
    frontier: list[int] = []
    for cell_id in range(archive.n_cells):
        if archive.is_empty_cell(cell_id):
            continue
        for neighbor in archive.neighbors(cell_id):
            if archive.is_empty_cell(neighbor):
                frontier.append(cell_id)
                break
    return frontier


def _min_fitness_cell(
    cell_ids: list[int],
    archive: ArchiveProtocol,
) -> int:
    best: int | None = None
    best_fitness = float("inf")
    for cell_id in cell_ids:
        elite = archive.get_cell(cell_id)
        if elite is None:
            continue
        if elite.fitness < best_fitness or (
            elite.fitness == best_fitness and (best is None or cell_id < best)
        ):
            best = cell_id
            best_fitness = elite.fitness
    if best is None:
        msg = "frontier cells must contain elites"
        raise RuntimeError(msg)
    return best

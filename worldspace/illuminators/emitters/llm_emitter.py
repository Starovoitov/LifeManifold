"""MAP-Elites LLM emitter: prompts, API call, parse, and random-walk fallback."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from worldspace.generators import RandomWalkWorldGenerator
from worldspace.generators.llm_config import LlmTextCaller, load_llm_config
from worldspace.illuminators.archive import (
    ArchiveElite,
    GridArchive,
    new_elite_metadata,
)
from worldspace.illuminators.archive_protocol import ArchiveProtocol
from worldspace.illuminators.emitters.base import EmitterOutput, strip_seed
from worldspace.illuminators.emitters.llm_prompts import (
    render_system_prompt,
    system_prompt_version,
)
from worldspace.illuminators.emitters.random_emitter import RandomEmitter
from worldspace.illuminators.grid_neighbors import moore_neighbors_bounded
from worldspace.illuminators.scheduler import (
    SchedulerConfig,
    TargetBin,
    resolve_surrogate_stub,
)
from worldspace.surrogate.types import SurrogateProtocol
from worldspace.specs.spec import WorldSpec
from worldspace.specs.world_spec_constraints import format_world_spec_constraints
from worldspace.specs.world_spec_from_llm import (
    extract_json_object_from_text,
    world_spec_from_llm_payload,
)

_DEFAULT_FEW_SHOT = 4
_EMPTY_FEW_SHOT_TEXT = "(no occupied neighboring niches)"
_DEFAULT_LLM_SPEC = (
    Path(__file__).resolve().parents[2] / "specs" / "llm_world_generator.yaml"
)
_EMITTER_TYPE_LLM = "llm"
_EMITTER_TYPE_LLM_FALLBACK = "llm_fallback"

__all__ = [
    "LlmEmitter",
    "build_user_prompt",
    "format_current_elite_json",
    "format_few_shot_block",
    "moore_neighbor_elites",
]


class LlmEmitter:
    """Generate candidates via LLM JSON or one random-walk step from parent 1."""

    def __init__(
        self,
        *,
        grid_resolution: int,
        scheduler: SchedulerConfig | None = None,
        surrogate: SurrogateProtocol | None = None,
        surrogate_mean: float = 0.5,
        surrogate_uncertainty: float = 1.0,
        fallback_scale: float = 0.02,
        llm_spec_path: str | Path | None = None,
        call_llm_text: LlmTextCaller | None = None,
        random_emitter: RandomEmitter | None = None,
    ) -> None:
        self._grid_resolution = int(grid_resolution)
        self._scheduler = scheduler
        self._surrogate = surrogate
        self._surrogate_mean = float(surrogate_mean)
        self._surrogate_uncertainty = float(surrogate_uncertainty)
        self._fallback_scale = float(fallback_scale)
        self._llm_config = load_llm_config(llm_spec_path or _DEFAULT_LLM_SPEC)
        self._random = random_emitter or RandomEmitter()
        self._call_llm_text = call_llm_text
        self._prompt_version = system_prompt_version()

    def emit(
        self,
        *,
        target: TargetBin,
        archive: ArchiveProtocol,
        rng: np.random.Generator,
        grid_size: int,
        steps: int,
    ) -> EmitterOutput:
        if not isinstance(archive, GridArchive):
            return self._random.emit(
                target=target,
                archive=archive,
                rng=rng,
                grid_size=grid_size,
                steps=steps,
            )
        parent_spec, parent_id = self._resolve_parent_one(
            target=target,
            archive=archive,
            rng=rng,
            grid_size=grid_size,
            steps=steps,
        )
        prepared_parent = replace(parent_spec, grid_size=grid_size, steps=steps)
        surrogate_mean, surrogate_uncertainty = self._resolve_surrogate_values(
            prepared_parent
        )
        system_prompt = render_system_prompt(self._grid_resolution)
        user_prompt = build_user_prompt(
            target=target,
            archive=archive,
            surrogate_mean=surrogate_mean,
            surrogate_uncertainty=surrogate_uncertainty,
            rng=rng,
        )
        try:
            response = self._request_llm(system_prompt, user_prompt)
        except (RuntimeError, ValueError):
            response = ""
        parsed = extract_json_object_from_text(response)
        if parsed is not None:
            spec = world_spec_from_llm_payload(
                parsed,
                grid_size=grid_size,
                steps=steps,
                base=parent_spec,
            )
            if spec is not None:
                return EmitterOutput(
                    world_spec=strip_seed(spec),
                    metadata=new_elite_metadata(
                        generated_by="llm",
                        emitter_type=_EMITTER_TYPE_LLM,
                        parent_id=parent_id,
                        prompt_version=self._prompt_version,
                    ),
                )
        fallback_spec = _random_walk_step(
            parent_spec,
            scale=self._fallback_scale,
            rng=rng,
            grid_size=grid_size,
            steps=steps,
        )
        return EmitterOutput(
            world_spec=fallback_spec,
            metadata=new_elite_metadata(
                generated_by="llm",
                emitter_type=_EMITTER_TYPE_LLM_FALLBACK,
                parent_id=parent_id,
                prompt_version=self._prompt_version,
            ),
        )

    def _resolve_parent_one(
        self,
        *,
        target: TargetBin,
        archive: GridArchive,
        rng: np.random.Generator,
        grid_size: int,
        steps: int,
    ) -> tuple[WorldSpec, str | None]:
        elite = archive.get(*target.bin)
        if elite is not None and elite.world_spec is not None:
            parent_id = elite.metadata.id if elite.metadata is not None else None
            return (
                replace(elite.world_spec, grid_size=grid_size, steps=steps),
                parent_id,
            )
        random_out = self._random.emit(
            target=target,
            archive=archive,
            rng=rng,
            grid_size=grid_size,
            steps=steps,
        )
        return random_out.world_spec, None

    def _resolve_surrogate_values(self, world_spec: WorldSpec) -> tuple[float, float]:
        if self._scheduler is not None and self._surrogate is not None:
            return resolve_surrogate_stub(self._scheduler, self._surrogate, world_spec)
        return (self._surrogate_mean, self._surrogate_uncertainty)

    def _request_llm(self, system_prompt: str, user_prompt: str) -> str:
        if self._call_llm_text is None:
            from worldspace.generators import call_llm

            caller: LlmTextCaller = call_llm
        else:
            caller = self._call_llm_text
        cfg = self._llm_config
        return caller(
            mode=cfg.mode,
            provider_name=cfg.active_provider,
            providers=cfg.providers,
            prompt=user_prompt,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            system_content=system_prompt,
        )


def build_user_prompt(
    *,
    target: TargetBin,
    archive: GridArchive,
    surrogate_mean: float,
    surrogate_uncertainty: float,
    rng: np.random.Generator,
    max_few_shot: int = _DEFAULT_FEW_SHOT,
) -> str:
    """Build the LLM user prompt for one emitter slot."""
    from worldspace.illuminators.emitters.llm_prompts import USER_PROMPT_TEMPLATE

    neighbors = moore_neighbor_elites(
        archive, target.bin, rng=rng, max_count=max_few_shot
    )
    current = archive.get(*target.bin)
    return USER_PROMPT_TEMPLATE.format(
        target_stability=target.target_stability,
        target_diversity=target.target_diversity,
        surrogate_mean=surrogate_mean,
        surrogate_uncertainty=surrogate_uncertainty,
        current_elite_json=format_current_elite_json(current),
        few_shot_examples=format_few_shot_block(neighbors),
        constraints=format_world_spec_constraints(),
    )


def moore_neighbor_elites(
    archive: GridArchive,
    bin_coord: tuple[int, int],
    *,
    rng: np.random.Generator,
    max_count: int = _DEFAULT_FEW_SHOT,
) -> list[ArchiveElite]:
    """Return up to ``max_count`` elites from occupied Moore neighbors of ``bin_coord``."""
    if max_count < 1:
        return []
    i, j = bin_coord
    resolution = archive.resolution
    neighbors: list[ArchiveElite] = []
    for ni, nj in moore_neighbors_bounded(i, j, resolution):
        elite = archive.get(ni, nj)
        if elite is not None:
            neighbors.append(elite)
    if len(neighbors) <= max_count:
        return neighbors
    indices = rng.choice(len(neighbors), size=max_count, replace=False)
    return [neighbors[int(idx)] for idx in sorted(indices)]


def format_current_elite_json(elite: ArchiveElite | None) -> str:
    """Serialize the current cell elite for the user prompt (or ``null``)."""
    if elite is None:
        return "null"
    return json.dumps(_elite_prompt_record(elite), ensure_ascii=True, indent=2)


def format_few_shot_block(elites: list[ArchiveElite]) -> str:
    """Format few-shot neighbor examples for the user prompt."""
    if not elites:
        return _EMPTY_FEW_SHOT_TEXT
    records = [_elite_prompt_record(elite) for elite in elites]
    return json.dumps(records, ensure_ascii=True, indent=2)


def _random_walk_step(
    parent: WorldSpec,
    *,
    scale: float,
    rng: np.random.Generator,
    grid_size: int,
    steps: int,
) -> WorldSpec:
    walker = RandomWalkWorldGenerator(
        start_world=replace(parent, seed=0, grid_size=grid_size, steps=steps),
        scale=scale,
    )
    step_seed = int(rng.integers(0, 2**31))
    child = walker._step(
        replace(parent, seed=0, grid_size=grid_size, steps=steps), step_seed
    )
    return strip_seed(replace(child, grid_size=grid_size, steps=steps))


def _elite_prompt_record(elite: ArchiveElite) -> dict:
    if elite.world_spec is None:
        msg = "elite.world_spec is required for prompt serialization"
        raise ValueError(msg)
    spec_dict = elite.world_spec.to_canonical_dict()
    record: dict = {
        "bin": [elite.bin[0], elite.bin[1]],
        "fitness": elite.fitness,
        "measures": dict(elite.measures) if elite.measures is not None else {},
        "world_spec": spec_dict,
    }
    return record

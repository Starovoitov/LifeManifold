"""MAP-Elites LLM emitter: prompts, API call, parse, and random-walk fallback."""

from __future__ import annotations

import http.client
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

import numpy as np

from worldspace.generators import RandomWalkWorldGenerator
from worldspace.generators.llm_config import LlmTextCaller, load_llm_config
from worldspace.illuminators.archive import ArchiveElite, new_elite_metadata
from worldspace.illuminators.archive_protocol import ArchiveProtocol
from worldspace.illuminators.emitters.archive_neighbors import neighbor_elites
from worldspace.illuminators.emitters.base import EmitterOutput, strip_seed
from worldspace.illuminators.emitters.llm_prompts import (
    emitter_prompt_version,
    load_user_prompt_template,
    parent_prompt_fields,
    render_system_prompt_for_archive_type,
    surrogate_prompt_fields,
)
from worldspace.illuminators.emitters.random_emitter import RandomEmitter
from worldspace.illuminators.scheduler import (
    SchedulerConfig,
    TargetCell,
    resolve_surrogate_prediction,
)
from worldspace.surrogate import StubSurrogate
from worldspace.surrogate.types import SurrogatePrediction, SurrogateProtocol
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
_LLM_RESPONSE_PREVIEW_CHARS = 160

logger = logging.getLogger(__name__)

__all__ = [
    "LlmEmitter",
    "LlmPreparedSlot",
    "build_user_prompt",
    "format_current_elite_json",
    "format_few_shot_block",
]


@dataclass(frozen=True)
class LlmPreparedSlot:
    """Prompts and lineage for one LLM slot after prepare, before HTTP."""

    target: TargetCell
    parent_spec: WorldSpec
    parent_id: str | None
    system_prompt: str
    user_prompt: str
    prompt_version: str
    grid_size: int
    steps: int
    surrogate_prediction: SurrogatePrediction | None = None


class LlmEmitter:
    """Generate candidates via LLM JSON or one random-walk step from parent 1."""

    def __init__(
        self,
        *,
        grid_resolution: int | None = None,
        scheduler: SchedulerConfig | None = None,
        surrogate: SurrogateProtocol | None = None,
        surrogate_mean: float = 0.5,
        surrogate_uncertainty: float = 1.0,
        fallback_scale: float = 0.02,
        llm_spec_path: str | Path | None = None,
        call_llm_text: LlmTextCaller | None = None,
        random_emitter: RandomEmitter | None = None,
    ) -> None:
        """Configure grid/CVT resolution, surrogate hints, and LLM provider settings."""
        if scheduler is not None:
            self._grid_resolution = scheduler.grid_resolution
            self._n_centroids = scheduler.n_centroids
        elif grid_resolution is not None:
            self._grid_resolution = int(grid_resolution)
            self._n_centroids = self._grid_resolution * self._grid_resolution
        else:
            self._grid_resolution = 50
            self._n_centroids = 50 * 50
        self._scheduler = scheduler
        self._surrogate = surrogate
        self._surrogate_mean = float(surrogate_mean)
        self._surrogate_uncertainty = float(surrogate_uncertainty)
        self._fallback_scale = float(fallback_scale)
        self._llm_config = load_llm_config(llm_spec_path or _DEFAULT_LLM_SPEC)
        self._random = random_emitter or RandomEmitter()
        self._call_llm_text = call_llm_text

    def emit(
        self,
        *,
        target: TargetCell,
        archive: ArchiveProtocol,
        rng: np.random.Generator,
        grid_size: int,
        steps: int,
    ) -> EmitterOutput:
        """Prepare prompts, call the LLM once, and parse or random-walk fallback."""
        prepared = self.prepare_emit(
            target=target,
            archive=archive,
            rng=rng,
            grid_size=grid_size,
            steps=steps,
        )
        try:
            response = self.request_llm(prepared)
        except (RuntimeError, ValueError, OSError, http.client.HTTPException) as exc:
            return self.finalize_emit(
                prepared,
                response="",
                rng=rng,
                request_error=exc,
            )
        return self.finalize_emit(prepared, response=response, rng=rng)

    def prepare_emit(
        self,
        *,
        target: TargetCell,
        archive: ArchiveProtocol,
        rng: np.random.Generator,
        grid_size: int,
        steps: int,
    ) -> LlmPreparedSlot:
        """Build parent, surrogate hints, and prompts without an HTTP call."""
        parent_spec, parent_id = self._resolve_parent_one(
            target=target,
            archive=archive,
            rng=rng,
            grid_size=grid_size,
            steps=steps,
        )
        prepared_parent = replace(parent_spec, grid_size=grid_size, steps=steps)
        prediction = self._resolve_surrogate_prediction(prepared_parent)
        archive_type = cast(
            Literal["grid", "cvt"],
            archive.archive_type,
        )
        if archive_type not in {"grid", "cvt"}:
            msg = f"unsupported archive_type for LLM emitter: {archive.archive_type!r}"
            raise ValueError(msg)
        prompt_kind = self._resolve_system_prompt_kind(archive_type)
        system_prompt = render_system_prompt_for_archive_type(
            prompt_kind,
            grid_resolution=self._grid_resolution,
            n_centroids=archive.n_cells,
        )
        user_prompt_path = (
            self._scheduler.llm_user_prompt_path if self._scheduler is not None else None
        )
        user_prompt_template = load_user_prompt_template(user_prompt_path)
        prompt_version = emitter_prompt_version(
            archive_type=prompt_kind,
            user_path=user_prompt_path,
        )
        user_prompt = build_user_prompt(
            target=target,
            archive=archive,
            prediction=prediction,
            user_prompt_template=user_prompt_template,
            rng=rng,
        )
        return LlmPreparedSlot(
            target=target,
            parent_spec=parent_spec,
            parent_id=parent_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_version=prompt_version,
            grid_size=grid_size,
            steps=steps,
            surrogate_prediction=prediction,
        )

    def request_llm(self, prepared: LlmPreparedSlot) -> str:
        """POST chat completions for a prepared slot; raise on empty content."""
        response = self._request_llm(prepared.system_prompt, prepared.user_prompt)
        if not response.strip():
            msg = "empty LLM response"
            raise RuntimeError(msg)
        return response

    def finalize_emit(
        self,
        prepared: LlmPreparedSlot,
        *,
        response: str,
        rng: np.random.Generator,
        request_error: BaseException | None = None,
    ) -> EmitterOutput:
        """Parse LLM JSON into a child ``WorldSpec`` or apply random-walk fallback."""
        fallback_reason: str | None = None
        if request_error is not None:
            fallback_reason = (
                f"request_failed:{type(request_error).__name__}:{request_error}"
            )
            response = ""
        elif not response.strip():
            fallback_reason = "empty_response"

        parsed = extract_json_object_from_text(response)
        if parsed is not None:
            spec = world_spec_from_llm_payload(
                parsed,
                grid_size=prepared.grid_size,
                steps=prepared.steps,
                base=prepared.parent_spec,
            )
            if spec is not None:
                return EmitterOutput(
                    world_spec=strip_seed(spec),
                    metadata=new_elite_metadata(
                        generated_by="llm",
                        emitter_type=_EMITTER_TYPE_LLM,
                        parent_id=prepared.parent_id,
                        prompt_version=prepared.prompt_version,
                    ),
                )
            fallback_reason = "invalid_world_spec"
        elif fallback_reason is None:
            preview = response.strip().replace("\n", " ")[:_LLM_RESPONSE_PREVIEW_CHARS]
            fallback_reason = (
                "no_json_in_response" if preview else "no_json_in_empty_response"
            )

        logger.warning(
            "LLM emitter fallback cell_id=%s parent_id=%s reason=%s response_preview=%r",
            prepared.target.cell_id,
            prepared.parent_id,
            fallback_reason,
            response.strip().replace("\n", " ")[:_LLM_RESPONSE_PREVIEW_CHARS],
        )
        fallback_spec = _random_walk_step(
            prepared.parent_spec,
            scale=self._fallback_scale,
            rng=rng,
            grid_size=prepared.grid_size,
            steps=prepared.steps,
        )
        return EmitterOutput(
            world_spec=fallback_spec,
            metadata=new_elite_metadata(
                generated_by="llm",
                emitter_type=_EMITTER_TYPE_LLM_FALLBACK,
                parent_id=prepared.parent_id,
                prompt_version=prepared.prompt_version,
            ),
        )

    def _resolve_parent_one(
        self,
        *,
        target: TargetCell,
        archive: ArchiveProtocol,
        rng: np.random.Generator,
        grid_size: int,
        steps: int,
    ) -> tuple[WorldSpec, str | None]:
        """Return the target-cell elite as parent, or one random world if empty."""
        elite = archive.get_cell(target.cell_id)
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

    def _resolve_system_prompt_kind(
        self, archive_type: Literal["grid", "cvt"]
    ) -> Literal["grid", "cvt"]:
        """Return which system prompt template to render (may differ from archive)."""
        if self._scheduler is None:
            return archive_type
        kind = self._scheduler.llm_system_prompt_kind
        if kind == "auto":
            return archive_type
        return kind

    def _resolve_surrogate_prediction(self, world_spec: WorldSpec) -> SurrogatePrediction:
        """Return surrogate prediction for the user prompt."""
        if self._scheduler is not None and self._surrogate is not None:
            return resolve_surrogate_prediction(
                self._scheduler,
                self._surrogate,
                world_spec,
            )
        return StubSurrogate(
            self._surrogate_mean,
            self._surrogate_uncertainty,
        ).predict(world_spec)

    def _request_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Invoke the configured text caller (live API or test mock)."""
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
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            system_content=system_prompt,
        )


def build_user_prompt(
    *,
    target: TargetCell,
    archive: ArchiveProtocol,
    rng: np.random.Generator,
    prediction: SurrogatePrediction | None = None,
    surrogate_mean: float = 0.5,
    surrogate_uncertainty: float = 1.0,
    user_prompt_template: str | None = None,
    max_few_shot: int = _DEFAULT_FEW_SHOT,
) -> str:
    """Build the LLM user prompt for one emitter slot."""
    if prediction is None:
        prediction = StubSurrogate(surrogate_mean, surrogate_uncertainty).predict(
            _PROMPT_STUB_WORLD_SPEC
        )
    template = (
        user_prompt_template
        if user_prompt_template is not None
        else load_user_prompt_template()
    )
    neighbors = neighbor_elites(
        archive,
        target.cell_id,
        rng=rng,
        max_count=max_few_shot,
    )
    current = archive.get_cell(target.cell_id)
    return template.format(
        target_stability=target.target_stability,
        target_diversity=target.target_diversity,
        **surrogate_prompt_fields(prediction),
        **parent_prompt_fields(current),
        current_elite_json=format_current_elite_json(current),
        few_shot_examples=format_few_shot_block(neighbors),
        constraints=format_world_spec_constraints(),
    )


_PROMPT_STUB_WORLD_SPEC = WorldSpec(
    birth=[1],
    survival=[2, 3],
    noise=0.0,
    resource_regen=0.0,
    predation=0.0,
    cell_types=["life", "food"],
    grid_size=8,
    steps=200,
    seed=0,
)


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
    """Perturb ``parent`` by one scaled random-walk step when LLM output fails."""
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
    """Build a JSON-serializable elite record for few-shot / current-cell blocks."""
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

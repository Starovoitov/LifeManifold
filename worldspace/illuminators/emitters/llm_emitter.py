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
    resolve_direction_prompt_fields,
    surrogate_prompt_fields,
)
from worldspace.illuminators.emitters.random_emitter import RandomEmitter
from worldspace.illuminators.scheduler import (
    ChildRewriteConfig,
    SchedulerConfig,
    TargetCell,
    resolve_surrogate_prediction,
)
from worldspace.surrogate import StubSurrogate
from worldspace.surrogate.model import SurrogateModel
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
_EMITTER_TYPE_LLM_REWRITE = "llm_rewrite"
_LLM_RESPONSE_PREVIEW_CHARS = 160
_MISSING_PARENT_TRUE = float("nan")

logger = logging.getLogger(__name__)

__all__ = [
    "LlmEmitter",
    "LlmPreparedSlot",
    "apply_batch_hint_placebo",
    "build_rewrite_user_prompt",
    "build_user_prompt",
    "format_current_elite_json",
    "format_few_shot_block",
    "remap_prepared_slot_prediction",
    "should_rewrite_child",
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


def _surrogate_line(prediction: SurrogatePrediction) -> str:
    return (
        f"Surrogate predicts fitness ≈ {prediction.fitness:.3f}, "
        f"uncertainty = {prediction.uncertainty:.3f}"
    )


def remap_prepared_slot_prediction(
    slot: LlmPreparedSlot,
    prediction: SurrogatePrediction,
) -> LlmPreparedSlot:
    """Swap the surrogate scalars in a prepared slot without re-sampling few-shot."""
    old = slot.surrogate_prediction
    if old is None:
        msg = "cannot remap placebo prediction onto a slot without surrogate_prediction"
        raise ValueError(msg)
    old_line = _surrogate_line(old)
    new_line = _surrogate_line(prediction)
    if old_line not in slot.user_prompt:
        msg = (
            "prepared user_prompt is missing the expected surrogate line; "
            "hint_placebo=shuffle_batch requires the default fitness/uncertainty template"
        )
        raise ValueError(msg)
    return replace(
        slot,
        user_prompt=slot.user_prompt.replace(old_line, new_line, 1),
        surrogate_prediction=prediction,
    )


def apply_batch_hint_placebo(
    prepared_slots: list[LlmPreparedSlot],
    rng: np.random.Generator,
) -> list[LlmPreparedSlot]:
    """Permute intact SurrogatePrediction objects across LLM slots in one batch.

    Parent JSON / few-shot text stay with the slot; only the (fitness, uncertainty)
    pair (and attached prediction payload) is reassigned. The multiset of pairs is
    preserved so the control is distribution-matched to live MLP hints.
    """
    n = len(prepared_slots)
    if n <= 1:
        return list(prepared_slots)
    predictions = [slot.surrogate_prediction for slot in prepared_slots]
    if any(pred is None for pred in predictions):
        msg = (
            "hint_placebo=shuffle_batch requires surrogate_prediction on every LLM slot"
        )
        raise ValueError(msg)
    order = rng.permutation(n)
    return [
        remap_prepared_slot_prediction(slot, predictions[int(order[i])])  # type: ignore[arg-type]
        for i, slot in enumerate(prepared_slots)
    ]


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
        """Prepare prompts, call the LLM, optionally rewrite draft child, parse."""
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
        draft = self.finalize_emit(prepared, response=response, rng=rng)
        return self.maybe_apply_child_rewrite(
            prepared,
            draft,
            archive=archive,
            rng=rng,
        )

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
            self._scheduler.llm_user_prompt_path
            if self._scheduler is not None
            else None
        )
        user_prompt_template = load_user_prompt_template(user_prompt_path)
        prompt_version = emitter_prompt_version(
            archive_type=prompt_kind,
            user_path=user_prompt_path,
        )
        surrogate_model = (
            getattr(self._surrogate, "model", None)
            if self._surrogate is not None
            else None
        )
        use_soft_extinction = (
            self._scheduler.surrogate_use_soft_extinction
            if self._scheduler is not None
            else False
        )
        extinction_gate_threshold = (
            self._scheduler.surrogate_extinction_gate_threshold
            if self._scheduler is not None
            else 0.5
        )
        user_prompt = build_user_prompt(
            target=target,
            archive=archive,
            prediction=prediction,
            user_prompt_template=user_prompt_template,
            rng=rng,
            direction_parent_spec=prepared_parent,
            direction_surrogate_model=cast(SurrogateModel | None, surrogate_model),
            direction_use_soft_extinction=use_soft_extinction,
            direction_extinction_gate_threshold=extinction_gate_threshold,
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

    def child_rewrite_config(self) -> ChildRewriteConfig:
        """Return child-rewrite settings (disabled when scheduler absent)."""
        if self._scheduler is None:
            return ChildRewriteConfig()
        return self._scheduler.llm_child_rewrite

    def prepare_child_rewrite(
        self,
        prepared: LlmPreparedSlot,
        draft: EmitterOutput,
        *,
        archive: ArchiveProtocol,
        rng: np.random.Generator,
    ) -> LlmPreparedSlot | None:
        """Build a rewrite prompt slot, or None if rewrite is not triggered."""
        cfg = self.child_rewrite_config()
        if not cfg.enabled:
            return None
        if draft.metadata.emitter_type != _EMITTER_TYPE_LLM:
            return None

        child_pred = self._resolve_surrogate_prediction(draft.world_spec)
        parent_true = _parent_true_fitness(archive, prepared.target.cell_id)
        parent_pred = prepared.surrogate_prediction
        if not should_rewrite_child(
            cfg,
            child_pred_fitness=float(child_pred.fitness),
            parent_true_fitness=parent_true,
            parent_pred_fitness=(
                float(parent_pred.fitness) if parent_pred is not None else None
            ),
        ):
            return None

        rewrite_prompt = build_rewrite_user_prompt(
            target=prepared.target,
            archive=archive,
            rng=rng,
            draft_spec=draft.world_spec,
            child_prediction=child_pred,
            parent_true_fitness=parent_true,
            user_prompt_path=cfg.user_prompt_path,
        )
        return LlmPreparedSlot(
            target=prepared.target,
            parent_spec=prepared.parent_spec,
            parent_id=prepared.parent_id,
            system_prompt=prepared.system_prompt,
            user_prompt=rewrite_prompt,
            prompt_version=prepared.prompt_version,
            grid_size=prepared.grid_size,
            steps=prepared.steps,
            surrogate_prediction=child_pred,
        )

    def commit_child_rewrite(
        self,
        *,
        draft: EmitterOutput,
        rewrite_prepared: LlmPreparedSlot,
        rewrite_response: str,
        rng: np.random.Generator,
        request_error: BaseException | None = None,
    ) -> EmitterOutput:
        """Parse rewrite response; keep draft on failure when configured."""
        cfg = self.child_rewrite_config()
        if request_error is not None:
            logger.warning(
                "LLM child rewrite request failed cell_id=%s reason=%s:%s",
                rewrite_prepared.target.cell_id,
                type(request_error).__name__,
                request_error,
            )
            return draft
        rewritten = self.finalize_emit(
            rewrite_prepared,
            response=rewrite_response,
            rng=rng,
        )
        if rewritten.metadata.emitter_type == _EMITTER_TYPE_LLM:
            return EmitterOutput(
                world_spec=rewritten.world_spec,
                metadata=new_elite_metadata(
                    generated_by="llm",
                    emitter_type=_EMITTER_TYPE_LLM_REWRITE,
                    parent_id=rewrite_prepared.parent_id,
                    prompt_version=rewrite_prepared.prompt_version,
                ),
            )
        if cfg.keep_draft_on_rewrite_fail:
            return draft
        return rewritten

    def maybe_apply_child_rewrite(
        self,
        prepared: LlmPreparedSlot,
        draft: EmitterOutput,
        *,
        archive: ArchiveProtocol,
        rng: np.random.Generator,
    ) -> EmitterOutput:
        """Optionally second-pass rewrite a parsed draft using predict(child)."""
        rewrite_prepared = self.prepare_child_rewrite(
            prepared, draft, archive=archive, rng=rng
        )
        if rewrite_prepared is None:
            return draft
        try:
            rewrite_response = self.request_llm(rewrite_prepared)
        except (RuntimeError, ValueError, OSError, http.client.HTTPException) as exc:
            return self.commit_child_rewrite(
                draft=draft,
                rewrite_prepared=rewrite_prepared,
                rewrite_response="",
                rng=rng,
                request_error=exc,
            )
        return self.commit_child_rewrite(
            draft=draft,
            rewrite_prepared=rewrite_prepared,
            rewrite_response=rewrite_response,
            rng=rng,
        )

    def _resolve_surrogate_prediction(
        self, world_spec: WorldSpec
    ) -> SurrogatePrediction:
        """Return surrogate prediction for the user prompt.

        When ``llm.stub_hints_only`` is set, prompt scalars stay at YAML stubs
        even if a live facade is loaded for after-generation filtering.
        """
        if self._scheduler is not None and self._scheduler.llm_stub_hints_only:
            return StubSurrogate(
                self._scheduler.surrogate_stub_mean,
                self._scheduler.surrogate_stub_uncertainty,
            ).predict(world_spec)
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
    direction_parent_spec: WorldSpec | None = None,
    direction_surrogate_model: SurrogateModel | None = None,
    direction_use_soft_extinction: bool = False,
    direction_extinction_gate_threshold: float = 0.5,
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
        **resolve_direction_prompt_fields(
            template,
            parent_world_spec=direction_parent_spec,
            surrogate_model=direction_surrogate_model,
            use_soft_extinction=direction_use_soft_extinction,
            extinction_gate_threshold=direction_extinction_gate_threshold,
        ),
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


def should_rewrite_child(
    cfg: ChildRewriteConfig,
    *,
    child_pred_fitness: float,
    parent_true_fitness: float,
    parent_pred_fitness: float | None = None,
) -> bool:
    """Return whether a draft child should receive a rewrite LLM call."""
    if not cfg.enabled:
        return False
    if cfg.trigger == "always":
        return True
    if cfg.trigger == "below_tau":
        return child_pred_fitness < float(cfg.min_predicted_fitness)
    if cfg.trigger == "below_parent_pred":
        if parent_pred_fitness is None:
            return True
        return child_pred_fitness < float(parent_pred_fitness)
    # below_parent_true (default): empty niche → always rewrite parsed drafts
    if parent_true_fitness != parent_true_fitness:  # NaN
        return True
    return child_pred_fitness < float(parent_true_fitness)


def _parent_true_fitness(archive: ArchiveProtocol, cell_id: int) -> float:
    elite = archive.get_cell(cell_id)
    if elite is None:
        return _MISSING_PARENT_TRUE
    return float(elite.fitness)


def build_rewrite_user_prompt(
    *,
    target: TargetCell,
    archive: ArchiveProtocol,
    rng: np.random.Generator,
    draft_spec: WorldSpec,
    child_prediction: SurrogatePrediction,
    parent_true_fitness: float,
    user_prompt_path: str | None,
    max_few_shot: int = _DEFAULT_FEW_SHOT,
) -> str:
    """Build the second-pass rewrite user prompt for a draft child."""
    template = load_user_prompt_template(user_prompt_path)
    neighbors = neighbor_elites(
        archive,
        target.cell_id,
        rng=rng,
        max_count=max_few_shot,
    )
    current = archive.get_cell(target.cell_id)
    parent_fit = (
        parent_true_fitness if parent_true_fitness == parent_true_fitness else 0.0
    )
    return template.format(
        target_stability=target.target_stability,
        target_diversity=target.target_diversity,
        parent_true_fitness=parent_fit,
        child_surrogate_mean=float(child_prediction.fitness),
        child_surrogate_uncertainty=float(child_prediction.uncertainty),
        draft_world_spec_json=json.dumps(
            strip_seed(draft_spec).to_json_dict(),
            ensure_ascii=True,
            indent=2,
        ),
        current_elite_json=format_current_elite_json(current),
        few_shot_examples=format_few_shot_block(neighbors),
        constraints=format_world_spec_constraints(),
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

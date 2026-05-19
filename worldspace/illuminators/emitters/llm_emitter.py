"""MAP-Elites LLM user prompt assembly and few-shot selection."""

from __future__ import annotations

import json

import numpy as np

from worldspace.illuminators.archive import ArchiveElite, GridArchive
from worldspace.illuminators.emitters.llm_prompts import USER_PROMPT_TEMPLATE
from worldspace.illuminators.grid_neighbors import moore_neighbors_bounded
from worldspace.illuminators.scheduler import TargetBin
from worldspace.specs.world_spec_constraints import format_world_spec_constraints

_DEFAULT_FEW_SHOT = 4
_EMPTY_FEW_SHOT_TEXT = "(no occupied neighboring niches)"

__all__ = [
    "build_user_prompt",
    "format_current_elite_json",
    "format_few_shot_block",
    "moore_neighbor_elites",
]


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

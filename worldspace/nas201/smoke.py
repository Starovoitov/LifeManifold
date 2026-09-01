"""NAS-Bench-201 feasibility smoke: random/genetic, no LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from worldspace.nas201.archive import Nas201Archive, Nas201Elite
from worldspace.nas201.descriptors import Nas201BinEdges
from worldspace.nas201.emitters import (
    Nas201TargetSelection,
    emit_genetic,
    emit_random,
    select_target_cell,
)
from worldspace.nas201.emitters import Nas201EmitterResult
from worldspace.nas201.evaluation import evaluate_spec
from worldspace.nas201.spec import Nas201Spec
from worldspace.nas201.table import Nas201Lookup

Nas201Generator = Literal["random", "genetic"]

SMOKE_EVALUATIONS = 200
SMOKE_INITIAL_RANDOM = 20
RESERVED_SMOKE_SEED = 201_001


@dataclass(frozen=True)
class Nas201SmokeResult:
    generator: Nas201Generator
    selector: Nas201TargetSelection
    seed: int
    initial_random: int
    genetic_or_random_steps: int
    proposals: int
    evaluations: int
    lookup_hits: int
    lookup_misses: int
    structurally_invalid: int
    filled_cells: int
    coverage: float
    qd_score: float
    occupied_bins: tuple[tuple[int, int], ...]


def niche_jaccard(
    first: frozenset[tuple[int, int]],
    second: frozenset[tuple[int, int]],
) -> float:
    if not first and not second:
        return 1.0
    union = first | second
    return len(first & second) / float(len(union))


def run_nas201_smoke(
    table: Nas201Lookup,
    edges: Nas201BinEdges,
    *,
    generator: Nas201Generator,
    selector: Nas201TargetSelection,
    seed: int = RESERVED_SMOKE_SEED,
    evaluations: int = SMOKE_EVALUATIONS,
    initial_random: int = SMOKE_INITIAL_RANDOM,
    initial_archive: Nas201Archive | None = None,
) -> tuple[Nas201SmokeResult, Nas201Archive]:
    rng = np.random.default_rng(seed)
    archive = (
        initial_archive.clone() if initial_archive is not None else Nas201Archive(edges)
    )
    proposals = 0
    eval_count = 0
    hits = 0
    misses = 0
    invalid = 0
    if initial_archive is None:
        for _ in range(initial_random):
            emitted = emit_random(rng)
            proposals += 1
            outcome = _insert(emitted.spec, emitted, archive, table, edges, proposals)
            eval_count += 1
            hits += int(outcome == "hit")
            misses += int(outcome == "miss")
            invalid += int(outcome == "invalid")
    steps = evaluations
    for _ in range(steps):
        if generator == "random":
            emitted = emit_random(rng)
        else:
            target = select_target_cell(archive, rng, target_selection=selector)
            emitted = emit_genetic(target, rng)
        proposals += 1
        outcome = _insert(emitted.spec, emitted, archive, table, edges, proposals)
        eval_count += 1
        hits += int(outcome == "hit")
        misses += int(outcome == "miss")
        invalid += int(outcome == "invalid")
    result = Nas201SmokeResult(
        generator=generator,
        selector=selector,
        seed=seed,
        initial_random=0 if initial_archive is not None else initial_random,
        genetic_or_random_steps=steps,
        proposals=proposals,
        evaluations=eval_count,
        lookup_hits=hits,
        lookup_misses=misses,
        structurally_invalid=invalid,
        filled_cells=archive.filled_count(),
        coverage=archive.coverage(),
        qd_score=archive.qd_score(),
        occupied_bins=tuple(sorted(archive.occupied_bins())),
    )
    return result, archive


def seeded_initial_archive(
    table: Nas201Lookup,
    edges: Nas201BinEdges,
    *,
    seed: int,
    n_random: int = SMOKE_INITIAL_RANDOM,
) -> Nas201Archive:
    rng = np.random.default_rng(seed)
    archive = Nas201Archive(edges)
    for index in range(n_random):
        emitted = emit_random(rng)
        _insert(emitted.spec, emitted, archive, table, edges, index + 1)
    return archive


def _insert(
    spec: Nas201Spec,
    emitted: Nas201EmitterResult,
    archive: Nas201Archive,
    table: Nas201Lookup,
    edges: Nas201BinEdges,
    proposal_index: int,
) -> str:
    evaluation = evaluate_spec(spec, table, edges)
    if not evaluation.structurally_valid:
        return "invalid"
    if not evaluation.lookup_hit or evaluation.record is None or evaluation.bin is None:
        return "miss"
    if evaluation.fitness is None or evaluation.measures is None:
        return "miss"
    elite = Nas201Elite(
        bin=evaluation.bin,
        fitness=evaluation.fitness,
        measures=evaluation.measures,
        spec=spec,
        candidate_id=f"{spec.candidate_hash()}-{proposal_index}",
        parent_id=emitted.parent_id,
        emitter_type=emitted.emitter_type,
        architecture_index=evaluation.record.index,
    )
    archive.try_insert(elite)
    return "hit"

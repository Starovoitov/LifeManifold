"""PCG random/genetic smoke: no LLM. Repair is a named factor, default identity."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from worldspace.pcg.archive import PcgArchive, PcgElite
from worldspace.pcg.descriptors import PcgBinEdges
from worldspace.pcg.emitters import (
    PcgEmitterResult,
    PcgTargetSelection,
    emit_genetic,
    emit_random,
    select_target_cell,
)
from worldspace.pcg.evaluation import PcgEnvLike, evaluate_spec
from worldspace.pcg.repair import RepairKind, RepairMeta, apply_repair
from worldspace.pcg.spec import PcgSpec, PcgTask

PcgGenerator = Literal["random", "genetic"]

SMOKE_EVALUATIONS = 200
SMOKE_INITIAL_RANDOM = 20
RANDOM_EDGE_SAMPLES = 256
RESERVED_SOKOBAN_SEED = 201_301
RESERVED_ZELDA_SEED = 201_351
RESERVED_PAIR_SOKOBAN_SEED = 201_501
RESERVED_PAIR_ZELDA_SEED = 201_551
MAX_SELECTOR_JACCARD = 0.80
MAX_SMOKE_COVERAGE = 0.95
DETERMINISM_CONTENTS = 20
DETERMINISM_REPEATS = 10


@dataclass(frozen=True)
class PcgSmokeResult:
    problem_name: str
    generator: PcgGenerator
    selector: PcgTargetSelection
    seed: int
    initial_random: int
    steps: int
    proposals: int
    evaluations: int
    structurally_invalid: int
    playable: int
    filled_cells: int
    coverage: float
    qd_score: float
    occupied_bins: tuple[tuple[int, int], ...]
    repair_kind: RepairKind
    tiles_changed_mean: float
    astar_eligible: int
    quality_min: float | None
    quality_max: float | None
    measure0_min: float | None
    measure0_max: float | None


def niche_jaccard(
    first: frozenset[tuple[int, int]],
    second: frozenset[tuple[int, int]],
) -> float:
    if not first and not second:
        return 1.0
    union = first | second
    return len(first & second) / float(len(union))


def run_pcg_smoke(
    env: PcgEnvLike,
    task: PcgTask,
    edges: PcgBinEdges,
    *,
    generator: PcgGenerator,
    selector: PcgTargetSelection,
    seed: int,
    evaluations: int = SMOKE_EVALUATIONS,
    initial_random: int = SMOKE_INITIAL_RANDOM,
    initial_archive: PcgArchive | None = None,
    sample_from_env: bool = False,
    repair_kind: RepairKind = "identity",
) -> tuple[PcgSmokeResult, PcgArchive]:
    rng = np.random.default_rng(seed)
    archive = (
        initial_archive.clone() if initial_archive is not None else PcgArchive(edges)
    )
    proposals = 0
    eval_count = 0
    invalid = 0
    playable = 0
    tiles_changed = 0
    astar_eligible = 0
    qualities: list[float] = []
    measure0: list[float] = []
    if initial_archive is None:
        for _ in range(initial_random):
            sampled = _maybe_sample(env, sample_from_env)
            emitted = emit_random(task, rng, sampled=sampled)
            emitted, meta = _apply_named_repair(emitted, repair_kind)
            proposals += 1
            outcome, fitness, first_measure = _insert(
                emitted.spec, emitted, archive, env, edges, task, proposals
            )
            eval_count += 1
            invalid += int(outcome == "invalid")
            playable += int(outcome == "playable")
            tiles_changed += meta["tiles_changed"]
            astar_eligible += int(bool(meta.get("astar_eligible")))
            if fitness is not None:
                qualities.append(fitness)
            if first_measure is not None:
                measure0.append(first_measure)
    for _ in range(evaluations):
        if generator == "random":
            sampled = _maybe_sample(env, sample_from_env)
            emitted = emit_random(task, rng, sampled=sampled)
        else:
            target = select_target_cell(archive, rng, target_selection=selector)
            emitted = emit_genetic(target, rng, task)
        emitted, meta = _apply_named_repair(emitted, repair_kind)
        proposals += 1
        outcome, fitness, first_measure = _insert(
            emitted.spec, emitted, archive, env, edges, task, proposals
        )
        eval_count += 1
        invalid += int(outcome == "invalid")
        playable += int(outcome == "playable")
        tiles_changed += meta["tiles_changed"]
        astar_eligible += int(bool(meta.get("astar_eligible")))
        if fitness is not None:
            qualities.append(fitness)
        if first_measure is not None:
            measure0.append(first_measure)
    result = PcgSmokeResult(
        problem_name=task.problem_name,
        generator=generator,
        selector=selector,
        seed=seed,
        initial_random=0 if initial_archive is not None else initial_random,
        steps=evaluations,
        proposals=proposals,
        evaluations=eval_count,
        structurally_invalid=invalid,
        playable=playable,
        filled_cells=archive.filled_count(),
        coverage=archive.coverage(),
        qd_score=archive.qd_score(),
        occupied_bins=tuple(sorted(archive.occupied_bins())),
        repair_kind=repair_kind,
        tiles_changed_mean=(tiles_changed / float(proposals) if proposals else 0.0),
        astar_eligible=astar_eligible,
        quality_min=min(qualities) if qualities else None,
        quality_max=max(qualities) if qualities else None,
        measure0_min=min(measure0) if measure0 else None,
        measure0_max=max(measure0) if measure0 else None,
    )
    return result, archive


def seeded_initial_archive(
    env: PcgEnvLike,
    task: PcgTask,
    edges: PcgBinEdges,
    *,
    seed: int,
    n_random: int = SMOKE_INITIAL_RANDOM,
    sample_from_env: bool = False,
    repair_kind: RepairKind = "identity",
) -> PcgArchive:
    rng = np.random.default_rng(seed)
    archive = PcgArchive(edges)
    for index in range(n_random):
        sampled = _maybe_sample(env, sample_from_env)
        emitted = emit_random(task, rng, sampled=sampled)
        emitted, _meta = _apply_named_repair(emitted, repair_kind)
        _insert(emitted.spec, emitted, archive, env, edges, task, index + 1)
    return archive


def _maybe_sample(env: PcgEnvLike, sample_from_env: bool) -> object | None:
    if not sample_from_env:
        return None
    sampler = getattr(env, "sample_content", None)
    if sampler is None:
        raise TypeError("env must provide sample_content when sample_from_env=True")
    return sampler()


def _insert(
    spec: PcgSpec,
    emitted: PcgEmitterResult,
    archive: PcgArchive,
    env: PcgEnvLike,
    edges: PcgBinEdges,
    task: PcgTask,
    proposal_index: int,
) -> tuple[str, float | None, float | None]:
    evaluation = evaluate_spec(spec, env, edges, task)
    if not evaluation.structurally_valid:
        return "invalid", None, None
    if (
        evaluation.fitness is None
        or evaluation.measures is None
        or evaluation.bin is None
    ):
        return "invalid", None, None
    elite = PcgElite(
        bin=evaluation.bin,
        fitness=evaluation.fitness,
        measures=evaluation.measures,
        spec=spec,
        candidate_id=f"{spec.candidate_hash()}-{proposal_index}",
        parent_id=emitted.parent_id,
        emitter_type=emitted.emitter_type,
        playable=evaluation.playable,
    )
    archive.try_insert(elite)
    outcome = "playable" if evaluation.playable else "hit"
    return outcome, evaluation.fitness, evaluation.measures[0]


def _apply_named_repair(
    emitted: PcgEmitterResult, repair_kind: RepairKind
) -> tuple[PcgEmitterResult, RepairMeta]:
    spec, meta = apply_repair(emitted.spec, repair_kind)
    return replace(emitted, spec=spec), meta

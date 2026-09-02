"""Native PCG sokoban-v0 confirmatory loop (matched structural_counts)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from worldspace.attribution.public_loop import (
    SUMMARY_FILENAME,
    SUMMARY_SCHEMA,
    TRACE_FILENAME,
    PublicRunConfig,
    append_jsonl,
    anytime_auc,
    mean_or_none,
    should_use_llm,
    write_json,
)
from worldspace.pcg.archive import PcgArchive, PcgElite
from worldspace.pcg.descriptors import PcgBinEdges
from worldspace.pcg.emitters import (
    PcgTargetSelection,
    emit_genetic,
    emit_random,
    select_target_cell,
)
from worldspace.pcg.evaluation import PcgEnvLike, evaluate_spec
from worldspace.pcg.repair import RepairKind, apply_repair
from worldspace.pcg.spec import SOKOBAN_V0, PcgSpec

ARCHIVE_FILENAME = "pcg_archive.jsonl"

LlmEmitFn = Callable[
    [PcgSpec, np.random.Generator, int],
    tuple[PcgSpec, dict[str, Any]],
]


@dataclass(frozen=True)
class PcgRunResult:
    output_dir: Path
    summary: dict[str, Any]
    filled_cells: int
    coverage: float
    qd_score: float
    anytime_auc: float


def run_pcg_sokoban_qd(
    env: PcgEnvLike,
    edges: PcgBinEdges,
    config: PublicRunConfig,
    *,
    output_dir: Path,
    llm_emit: LlmEmitFn | None = None,
) -> PcgRunResult:
    """Run one sokoban MAP-Elites job with named repair on every emit."""
    if edges.problem_name != SOKOBAN_V0.problem_name:
        raise ValueError(
            f"edges problem {edges.problem_name!r} is not {SOKOBAN_V0.problem_name}"
        )
    if config.generator == "llm" and llm_emit is None:
        raise ValueError("llm generator requires llm_emit callback")
    if config.prompt_channel == "live" and config.generator != "llm":
        raise ValueError("live prompt channel requires llm generator")
    repair_kind: RepairKind
    if config.repair_kind not in {"identity", "structural_counts", "genetic_fallback"}:
        raise ValueError(f"unsupported PCG repair {config.repair_kind!r}")
    repair_kind = (
        "identity" if config.repair_kind == "genetic_fallback" else config.repair_kind  # type: ignore[assignment]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (SUMMARY_FILENAME, ARCHIVE_FILENAME, TRACE_FILENAME):
        path = output_dir / name
        if path.exists():
            path.unlink()

    rng = np.random.default_rng(config.seed)
    archive = PcgArchive(edges)
    proposals = 0
    evaluations = 0
    valid = 0
    playable = 0
    llm_attempted = 0
    llm_completed = 0
    llm_forfeited = 0
    fallbacks = 0
    repair_tiles = 0
    wall_started = time.perf_counter()
    qd_curve: list[float] = []

    def _trace() -> None:
        append_jsonl(
            output_dir / TRACE_FILENAME,
            {
                "proposals": proposals,
                "evaluations": evaluations,
                "filled_cells": archive.filled_count(),
                "coverage": archive.coverage(),
                "qd_score": archive.qd_score(),
                "mean_best_fitness": mean_or_none(
                    [elite.fitness for elite in archive.elites()]
                ),
                "llm_calls_completed": llm_completed,
            },
        )

    def _emit_random() -> tuple[PcgSpec, str | None, str, int]:
        emitted = emit_random(SOKOBAN_V0, rng)
        repaired, meta = apply_repair(emitted.spec, repair_kind)
        return repaired, None, "random", int(meta["tiles_changed"])

    def _insert(spec: PcgSpec, parent_id: str | None, emitter_type: str) -> bool:
        nonlocal evaluations, valid, playable
        evaluations += 1
        result = evaluate_spec(spec, env, edges, SOKOBAN_V0)
        if (
            not result.structurally_valid
            or result.fitness is None
            or result.measures is None
            or result.bin is None
        ):
            return False
        valid += 1
        playable += int(result.playable)
        elite = PcgElite(
            bin=result.bin,
            fitness=result.fitness,
            measures=result.measures,
            spec=spec,
            candidate_id=f"{spec.candidate_hash()}-{proposals}",
            parent_id=parent_id,
            emitter_type=emitter_type,
            playable=result.playable,
        )
        archive.try_insert(elite)
        return True

    for _ in range(config.floor_random):
        proposals += 1
        spec, _parent, _etype, tiles = _emit_random()
        repair_tiles += tiles
        _insert(spec, None, "random_floor")
        qd_curve.append(archive.qd_score())
        _trace()

    selector: PcgTargetSelection = config.selector  # type: ignore[assignment]
    for _slot in range(config.search_horizon):
        proposals += 1
        target = select_target_cell(archive, rng, target_selection=selector)
        fitnesses = [elite.fitness for elite in archive.elites()]
        target_fitness = None if target.parent is None else target.parent.fitness
        use_llm = config.generator == "llm" and should_use_llm(
            allocation=config.allocation,
            archive_fitnesses=fitnesses,
            target_empty=target.parent is None,
            target_fitness=target_fitness,
            completed_llm_calls=llm_completed,
            llm_call_cap=config.llm_call_cap,
        )
        if config.generator == "llm" and not use_llm:
            llm_forfeited += 1
        if config.generator == "random" or target.parent is None:
            spec, parent_id, emitter_type, tiles = _emit_random()
            repair_tiles += tiles
            _insert(spec, parent_id, emitter_type)
        elif config.generator == "genetic" or not use_llm:
            emitted = emit_genetic(target, rng, SOKOBAN_V0)
            repaired, meta = apply_repair(emitted.spec, repair_kind)
            repair_tiles += int(meta["tiles_changed"])
            _insert(repaired, emitted.parent_id, "genetic")
        else:
            assert llm_emit is not None
            assert target.parent is not None
            llm_attempted += 1
            child, meta = llm_emit(target.parent.spec, rng, proposals)
            if bool(meta.get("used_fallback")):
                fallbacks += 1
                emitter_type = "llm_fallback_genetic"
            else:
                llm_completed += 1
                emitter_type = "llm"
            repaired, repair_meta = apply_repair(child, repair_kind)
            repair_tiles += int(repair_meta["tiles_changed"])
            _insert(repaired, target.parent.candidate_id, emitter_type)
        qd_curve.append(archive.qd_score())
        _trace()

    wall_s = time.perf_counter() - wall_started
    search_curve = qd_curve[config.floor_random :]
    auc = anytime_auc(search_curve)
    for elite in archive.elites():
        append_jsonl(
            output_dir / ARCHIVE_FILENAME,
            {
                "bin": list(elite.bin),
                "fitness": elite.fitness,
                "measures": list(elite.measures),
                "grid": elite.spec.to_nested_list(),
                "candidate_id": elite.candidate_id,
                "parent_id": elite.parent_id,
                "emitter_type": elite.emitter_type,
                "playable": elite.playable,
                "genotype_hash": elite.spec.genotype_sha256(),
            },
        )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "domain": "pcg_sokoban",
        "benchmark": "pcg_sokoban",
        "problem_name": SOKOBAN_V0.problem_name,
        "seed": config.seed,
        "generator": config.generator,
        "target_selection": config.selector,
        "allocation": config.allocation,
        "prompt_channel": config.prompt_channel,
        "repair": (
            config.repair_kind
            if config.repair_kind != "genetic_fallback"
            else repair_kind
        ),
        "archive_type": "grid",
        "n_cells": archive.n_cells,
        "floor_random": config.floor_random,
        "search_horizon": config.search_horizon,
        "proposals": proposals,
        "evaluations": evaluations,
        "valid_proposals": valid,
        "playable": playable,
        "filled_cells": archive.filled_count(),
        "coverage": archive.coverage(),
        "qd_score": archive.qd_score(),
        "mean_best_fitness": mean_or_none(
            [elite.fitness for elite in archive.elites()]
        ),
        "qd_score_anytime_auc": auc,
        "mean_tiles_changed_by_repair": (
            repair_tiles / float(proposals) if proposals else 0.0
        ),
        "llm_enabled": config.generator == "llm",
        "llm_calls_attempted": llm_attempted,
        "llm_calls_completed": llm_completed,
        "llm_calls_forfeited": llm_forfeited,
        "fallbacks": fallbacks,
        "wall_seconds": wall_s,
        "completed": True,
    }
    write_json(output_dir / SUMMARY_FILENAME, summary)
    return PcgRunResult(
        output_dir=output_dir,
        summary=summary,
        filled_cells=archive.filled_count(),
        coverage=archive.coverage(),
        qd_score=archive.qd_score(),
        anytime_auc=auc,
    )

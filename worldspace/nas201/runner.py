"""Native NAS-Bench-201 confirmatory loop (sidecar artifacts, no universal QD)."""

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
    LlmCallLedger,
    PublicRunConfig,
    append_jsonl,
    llm_summary_fields,
    mean_or_none,
    occupied_genotype_hash,
    record_llm_meta,
    search_curve_auc,
    should_use_llm,
    write_json,
)
from worldspace.nas201.archive import Nas201Archive, Nas201Elite
from worldspace.nas201.descriptors import Nas201BinEdges
from worldspace.nas201.emitters import (
    Nas201TargetSelection,
    emit_genetic,
    emit_random,
    mutate_one_edge,
    random_spec,
    select_target_cell,
)
from worldspace.nas201.evaluation import evaluate_spec
from worldspace.nas201.spec import Nas201Spec
from worldspace.generators.llm_call_log import private_llm_call_log
from worldspace.nas201.table import Nas201Lookup

ARCHIVE_FILENAME = "nas201_archive.jsonl"

LlmEmitFn = Callable[
    [Nas201Spec, np.random.Generator, int, Nas201Elite],
    tuple[Nas201Spec, dict[str, Any]],
]


@dataclass(frozen=True)
class Nas201RunResult:
    output_dir: Path
    summary: dict[str, Any]
    filled_cells: int
    coverage: float
    qd_score: float
    anytime_auc: float


def run_nas201_qd(
    table: Nas201Lookup,
    edges: Nas201BinEdges,
    config: PublicRunConfig,
    *,
    output_dir: Path,
    llm_emit: LlmEmitFn | None = None,
) -> Nas201RunResult:
    """Run one NAS MAP-Elites job and write normalizable artifacts."""
    if config.floor_random < 0 or config.floor_genetic < 0:
        raise ValueError("floor lengths cannot be negative")
    if config.floor_random and config.floor_genetic:
        raise ValueError("a NAS run has one floor")
    if config.generator == "llm" and llm_emit is None:
        raise ValueError("llm generator requires llm_emit callback")
    if config.prompt_channel == "live" and config.generator != "llm":
        raise ValueError("live prompt channel requires llm generator")
    if config.repair_kind != "identity":
        raise ValueError(f"unsupported NAS repair {config.repair_kind!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (SUMMARY_FILENAME, ARCHIVE_FILENAME, TRACE_FILENAME):
        path = output_dir / name
        if path.exists():
            path.unlink()

    with private_llm_call_log(
        output_dir / "llm_call_log.jsonl",
        enabled=config.generator == "llm",
    ):
        return _run_nas201_qd_logged(
            table,
            edges,
            config,
            output_dir=output_dir,
            llm_emit=llm_emit,
        )


def _run_nas201_qd_logged(
    table: Nas201Lookup,
    edges: Nas201BinEdges,
    config: PublicRunConfig,
    *,
    output_dir: Path,
    llm_emit: LlmEmitFn | None,
) -> Nas201RunResult:
    rng = np.random.default_rng(config.seed)
    archive = Nas201Archive(edges)
    proposals = 0
    evaluations = 0
    valid = 0
    llm_attempted = 0
    llm_completed = 0
    llm_forfeited = 0
    fallbacks = 0
    ledger = LlmCallLedger()
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

    def _insert(spec: Nas201Spec, parent_id: str | None, emitter_type: str) -> bool:
        nonlocal evaluations, valid
        evaluations += 1
        result = evaluate_spec(spec, table, edges)
        if (
            not result.lookup_hit
            or result.fitness is None
            or result.measures is None
            or result.bin is None
            or result.record is None
        ):
            return False
        valid += 1
        elite = Nas201Elite(
            bin=result.bin,
            fitness=result.fitness,
            measures=result.measures,
            spec=spec,
            candidate_id=f"{spec.candidate_hash()}-{proposals}",
            parent_id=parent_id,
            emitter_type=emitter_type,
            architecture_index=result.record.index,
        )
        archive.try_insert(elite)
        return True

    if config.floor_genetic:
        for _ in range(config.floor_genetic):
            proposals += 1
            # Parent is drawn and discarded. The floor proposal is the one-edge mutant.
            child = mutate_one_edge(random_spec(rng), rng)
            _insert(child, None, "genetic_floor")
            qd_curve.append(archive.qd_score())
            _trace()
    else:
        for _ in range(config.floor_random):
            proposals += 1
            emitted = emit_random(rng)
            _insert(emitted.spec, None, "random_floor")
            qd_curve.append(archive.qd_score())
            _trace()

    initial_archive_hash = occupied_genotype_hash(
        [elite.spec.genotype_sha256() for elite in archive.elites()]
    )
    selector: Nas201TargetSelection = config.selector  # type: ignore[assignment]
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
            emitted = emit_random(rng)
            _insert(emitted.spec, None, emitted.emitter_type)
        elif config.generator == "genetic" or not use_llm:
            emitted = emit_genetic(target, rng)
            _insert(emitted.spec, emitted.parent_id, emitted.emitter_type)
        else:
            assert llm_emit is not None
            assert target.parent is not None
            llm_attempted += 1
            child, meta = llm_emit(target.parent.spec, rng, proposals, target.parent)
            record_llm_meta(ledger, meta)
            if bool(meta.get("used_fallback")):
                fallbacks += 1
                emitter_type = "llm_fallback_genetic"
            else:
                llm_completed += 1
                emitter_type = "llm"
            _insert(child, target.parent.candidate_id, emitter_type)
        qd_curve.append(archive.qd_score())
        _trace()

    wall_s = time.perf_counter() - wall_started
    auc = search_curve_auc(qd_curve, config.floor_random + config.floor_genetic)
    for elite in archive.elites():
        append_jsonl(
            output_dir / ARCHIVE_FILENAME,
            {
                "bin": list(elite.bin),
                "fitness": elite.fitness,
                "measures": list(elite.measures),
                "arch": elite.spec.arch_str,
                "ops": list(elite.spec.ops),
                "candidate_id": elite.candidate_id,
                "parent_id": elite.parent_id,
                "emitter_type": elite.emitter_type,
                "architecture_index": elite.architecture_index,
                "genotype_hash": elite.spec.genotype_sha256(),
            },
        )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "domain": "nas201",
        "benchmark": "nas201",
        "seed": config.seed,
        "generator": config.generator,
        "target_selection": config.selector,
        "allocation": config.allocation,
        "prompt_channel": config.prompt_channel,
        "repair": config.repair_kind,
        "initial_archive_hash": initial_archive_hash,
        "archive_type": "grid",
        "n_cells": archive.n_cells,
        "floor_random": config.floor_random,
        "floor_genetic": config.floor_genetic,
        "search_horizon": config.search_horizon,
        "proposals": proposals,
        "evaluations": evaluations,
        "valid_proposals": valid,
        "filled_cells": archive.filled_count(),
        "coverage": archive.coverage(),
        "qd_score": archive.qd_score(),
        "mean_best_fitness": mean_or_none(
            [elite.fitness for elite in archive.elites()]
        ),
        "qd_score_anytime_auc": auc,
        "llm_enabled": config.generator == "llm",
        "llm_calls_attempted": llm_attempted,
        "llm_calls_completed": llm_completed,
        "llm_calls_forfeited": llm_forfeited,
        "fallbacks": fallbacks,
        "wall_seconds": wall_s,
        "completed": True,
    }
    if config.generator == "llm":
        summary.update(llm_summary_fields(ledger, config))
    write_json(output_dir / SUMMARY_FILENAME, summary)
    return Nas201RunResult(
        output_dir=output_dir,
        summary=summary,
        filled_cells=archive.filled_count(),
        coverage=archive.coverage(),
        qd_score=archive.qd_score(),
        anytime_auc=auc,
    )

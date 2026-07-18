"""Domain-specific MAP-Elites loop for the B4 dungeon factorial."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TextIO

import numpy as np
import yaml

from worldspace.dungeons.archive import DungeonArchive, DungeonElite
from worldspace.dungeons.emitters import (
    DungeonEmitterResult,
    DungeonTarget,
    emit_genetic,
    emit_random,
    select_uniform_frontier,
)
from worldspace.dungeons.evaluation import evaluate_dungeon
from worldspace.dungeons.spec import DungeonSpec
from worldspace.dungeons.surrogate import DungeonPrediction
from worldspace.illuminators.archive_trace import write_archive_trace_line
from worldspace.illuminators.scheduler import TargetBin
from worldspace.surrogate.acquisition import decide, effective_action
from worldspace.surrogate.acquisition_config import AcquisitionConfig

DungeonEmitterKind = Literal["random", "genetic", "llm"]
DungeonCondition = Literal[
    "random",
    "genetic",
    "genetic_filter",
    "llm_stub",
    "llm_hints",
    "llm_hints_filter",
]


class DungeonPredictor(Protocol):
    def predict(self, spec: DungeonSpec) -> DungeonPrediction: ...


class DungeonLlmEmitterProtocol(Protocol):
    def emit(
        self,
        *,
        target: DungeonTarget,
        archive: DungeonArchive,
        rng: np.random.Generator,
        prediction: DungeonPrediction | None,
    ) -> DungeonEmitterResult: ...

    def emit_batch(
        self,
        jobs: list[
            tuple[
                DungeonTarget,
                DungeonArchive,
                np.random.Generator,
                DungeonPrediction | None,
            ]
        ],
        *,
        max_workers: int = 4,
    ) -> list[DungeonEmitterResult]: ...


@dataclass(frozen=True)
class DungeonSchedulerConfig:
    condition: DungeonCondition
    iterations: int = 650
    batch_size: int = 50
    archive_resolution: int = 30
    initial_random_candidates: int = 100
    emitters: tuple[DungeonEmitterKind, ...] = (
        *("random" for _ in range(20)),
        *("genetic" for _ in range(30)),
    )
    acquisition: AcquisitionConfig = AcquisitionConfig()
    surrogate_checkpoint: str | None = None
    llm_prompt_mode: Literal["off", "stub", "hints"] = "off"

    def validate(self) -> None:
        if self.iterations < 1 or self.batch_size < 1:
            raise ValueError("iterations and batch_size must be positive")
        if len(self.emitters) != self.batch_size:
            raise ValueError("emitters length must equal batch_size")
        if self.archive_resolution < 1:
            raise ValueError("archive_resolution must be positive")
        if self.acquisition.mode != "off" and not self.surrogate_checkpoint:
            raise ValueError("acquisition mode requires surrogate_checkpoint")
        if "llm" in self.emitters and self.llm_prompt_mode == "off":
            raise ValueError("LLM emitters require stub or hints prompt mode")


@dataclass(frozen=True)
class DungeonRunResult:
    condition: DungeonCondition
    seed: int
    proposals: int
    evaluations: int
    skipped: int
    filled_cells: int
    coverage: float
    mean_best_fitness: float | None
    qd_score: float
    elapsed_seconds: float


def load_dungeon_scheduler(path: Path) -> DungeonSchedulerConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("dungeon scheduler root must be a mapping")
    acquisition_raw = raw.get("acquisition") or {}
    condition = _yaml_enum(raw["condition"])
    batch_size = int(raw.get("batch_size", 50))
    emitters_raw = raw.get("emitters")
    emitters = (
        tuple(emitters_raw)
        if isinstance(emitters_raw, list)
        else _default_emitters(condition, batch_size)
    )
    acquisition = AcquisitionConfig(
        mode=_yaml_enum(acquisition_raw.get("mode", "off")),  # type: ignore[arg-type]
        policy=str(acquisition_raw.get("policy", "threshold_gate")),  # type: ignore[arg-type]
        min_predicted_fitness=float(acquisition_raw.get("min_predicted_fitness", 0.45)),
        max_uncertainty_to_skip=float(
            acquisition_raw.get("max_uncertainty_to_skip", 1.0)
        ),
        never_skip_empty_bin=bool(acquisition_raw.get("never_skip_empty_bin", True)),
    )
    config = DungeonSchedulerConfig(
        condition=condition,  # type: ignore[arg-type]
        iterations=int(raw.get("iterations", 650)),
        batch_size=batch_size,
        archive_resolution=int(raw.get("archive_resolution", 30)),
        initial_random_candidates=int(raw.get("initial_random_candidates", 100)),
        emitters=emitters,  # type: ignore[arg-type]
        acquisition=acquisition,
        surrogate_checkpoint=raw.get("surrogate_checkpoint"),
        llm_prompt_mode=_yaml_enum(raw.get("llm_prompt_mode", "off")),  # type: ignore[arg-type]
    )
    config.validate()
    return config


def _yaml_enum(value: object) -> str:
    """Normalize YAML 1.1's unquoted off/false coercion."""
    if value is False or value is None:
        return "off"
    return str(value).strip().lower()


def _default_emitters(
    condition: str,
    batch_size: int,
) -> tuple[DungeonEmitterKind, ...]:
    if batch_size != 50:
        raise ValueError("implicit emitter layout requires batch_size 50")
    if condition == "random":
        return ("random",) * 50
    if condition.startswith("llm_"):
        return ("random",) * 20 + ("llm",) * 30
    return ("random",) * 20 + ("genetic",) * 30


def run_dungeon_qd(
    config: DungeonSchedulerConfig,
    *,
    seed: int,
    output_dir: Path,
    predictor: DungeonPredictor | None = None,
    llm_emitter: DungeonLlmEmitterProtocol | None = None,
) -> DungeonRunResult:
    """Run one exact-proposal dungeon seed and write the standard artifacts."""
    config.validate()
    summary_path = output_dir / "nightly_run_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"completed run already exists: {summary_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = DungeonArchive(config.archive_resolution)
    rng = np.random.default_rng(seed)
    trace_path = output_dir / "archive_trace.jsonl"
    surrogate_path = output_dir / "surrogate_archive.jsonl"
    proposals = evaluations = skipped = 0
    started = time.perf_counter()
    with (
        trace_path.open("w", encoding="utf-8") as trace_file,
        surrogate_path.open("w", encoding="utf-8") as surrogate_file,
    ):
        _write_trace(trace_file, archive, iteration=0, evaluations=0, proposals=0)
        for iteration in range(config.iterations):
            plans: list[
                tuple[
                    int,
                    DungeonTarget,
                    DungeonEmitterKind,
                    np.random.Generator,
                    DungeonPrediction | None,
                ]
            ] = []
            emitted_batch: list[DungeonEmitterResult | None] = [
                None
            ] * config.batch_size
            llm_jobs: list[
                tuple[
                    DungeonTarget,
                    DungeonArchive,
                    np.random.Generator,
                    DungeonPrediction | None,
                ]
            ] = []
            llm_slots: list[int] = []
            for slot, configured_kind in enumerate(config.emitters):
                target = select_uniform_frontier(archive, rng)
                kind: DungeonEmitterKind = (
                    "random"
                    if proposals + slot < config.initial_random_candidates
                    else configured_kind
                )
                slot_rng = np.random.default_rng(
                    int(rng.integers(0, np.iinfo(np.int64).max))
                )
                parent_prediction = (
                    predictor.predict(target.parent.spec)
                    if kind == "llm"
                    and predictor is not None
                    and target.parent is not None
                    else None
                )
                plans.append((slot, target, kind, slot_rng, parent_prediction))
                if kind == "llm":
                    if llm_emitter is None:
                        raise ValueError("LLM condition requires llm_emitter")
                    llm_slots.append(slot)
                    llm_jobs.append((target, archive, slot_rng, parent_prediction))
                elif kind == "genetic":
                    emitted_batch[slot] = emit_genetic(target, archive, slot_rng)
                else:
                    emitted_batch[slot] = emit_random(slot_rng)
            if llm_jobs:
                assert llm_emitter is not None
                llm_results = llm_emitter.emit_batch(llm_jobs, max_workers=4)
                for slot, emitted in zip(llm_slots, llm_results, strict=True):
                    emitted_batch[slot] = emitted

            for slot, target, _, _, parent_prediction in plans:
                emitted = emitted_batch[slot]
                assert emitted is not None
                prediction = parent_prediction
                if predictor is not None:
                    prediction = predictor.predict(emitted.spec)
                action = "eval"
                decision_payload: dict[str, object] | None = None
                if config.acquisition.mode != "off":
                    if prediction is None:
                        raise ValueError("acquisition requires predictor")
                    target_bin = TargetBin(
                        bin=target.bin,
                        target_stability=target.center[0],
                        target_diversity=target.center[1],
                    )
                    decision = decide(
                        config.acquisition,
                        prediction,  # type: ignore[arg-type]
                        target_bin,
                        archive,  # type: ignore[arg-type]
                    )
                    action = effective_action(config.acquisition.mode, decision)
                    decision_payload = {
                        "action": action,
                        "recommended_action": decision.action,
                        "reason": decision.reason,
                        "policy_version": decision.policy_version,
                    }
                proposals += 1
                surrogate_file.write(
                    json.dumps(
                        {
                            "proposal": proposals,
                            "iteration": iteration,
                            "slot": slot,
                            "target_bin": list(target.bin),
                            "target_was_empty": archive.is_empty_cell(target.cell_id),
                            "candidate_hash": emitted.spec.candidate_hash(),
                            "prediction": (
                                {
                                    "fitness": prediction.fitness,
                                    "uncertainty": prediction.uncertainty,
                                    "measures": prediction.measures,
                                }
                                if prediction is not None
                                else None
                            ),
                            "decision": decision_payload,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                if action == "skip":
                    skipped += 1
                    continue
                evaluation = evaluate_dungeon(
                    emitted.spec,
                    seed=seed * 1_000_000 + proposals,
                )
                evaluations += 1
                elite = DungeonElite(
                    bin=archive.bin_for_measures(evaluation.measures),
                    fitness=evaluation.fitness,
                    measures=evaluation.measures,
                    spec=emitted.spec,
                    candidate_id=f"{emitted.spec.candidate_hash()}-{proposals}",
                    parent_id=emitted.parent_id,
                    emitter_type=emitted.emitter_type,
                )
                archive.try_insert(elite)
            _write_trace(
                trace_file,
                archive,
                iteration=iteration + 1,
                evaluations=evaluations,
                proposals=proposals,
            )
    elapsed = time.perf_counter() - started
    filled, coverage, mean_fitness, qd_score = _metrics(archive)
    archive_path = output_dir / "dungeon_archive.jsonl"
    archive.write_jsonl(archive_path)
    result = DungeonRunResult(
        condition=config.condition,
        seed=seed,
        proposals=proposals,
        evaluations=evaluations,
        skipped=skipped,
        filled_cells=filled,
        coverage=coverage,
        mean_best_fitness=mean_fitness,
        qd_score=qd_score,
        elapsed_seconds=round(elapsed, 3),
    )
    payload = {
        "schema_version": "dungeon-1.0",
        "scheduler": f"dungeon:{config.condition}",
        "condition": config.condition,
        "benchmark": "dungeon",
        "dungeon_benchmark": True,
        "seed": seed,
        "iterations": config.iterations,
        "proposals": proposals,
        "evaluations": evaluations,
        "skipped": skipped,
        "skip_rate": skipped / proposals if proposals else 0.0,
        "filled_cells": filled,
        "coverage": coverage,
        "mean_best_fitness": mean_fitness,
        "qd_score": qd_score,
        "elapsed_seconds": result.elapsed_seconds,
        "archive_type": "grid",
        "grid_resolution": config.archive_resolution,
        "n_cells": archive.n_cells,
        "llm_enabled": "llm" in config.emitters,
        "surrogate_enabled": predictor is not None,
        "archive_jsonl": str(archive_path.resolve()),
        "archive_trace": str(trace_path.resolve()),
        "surrogate_archive": str(surrogate_path.resolve()),
    }
    llm_audit = getattr(llm_emitter, "audit", None)
    if llm_audit is not None and hasattr(llm_audit, "to_dict"):
        audit_payload = llm_audit.to_dict()
        payload["llm_audit"] = audit_payload
        payload["llm_calls"] = audit_payload.get(
            "api_calls",
            audit_payload["attempts"],
        )
        payload["llm_fallback_rate"] = audit_payload["fallback_rate"]
        payload["llm_fallback_rate_pct"] = float(audit_payload["fallback_rate"]) * 100.0
        payload["llm_parse_success_rate"] = audit_payload["parse_success_rate"]
        payload["llm_mean_tile_distance"] = audit_payload["mean_tile_distance"]
    prompt_version = getattr(llm_emitter, "prompt_version", None)
    if prompt_version is not None:
        payload["prompt_version"] = prompt_version
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return result


def _metrics(
    archive: DungeonArchive,
) -> tuple[int, float, float | None, float]:
    elites = archive.elites()
    filled = len(elites)
    score = sum(elite.fitness for elite in elites)
    return (
        filled,
        filled / archive.n_cells,
        score / filled if filled else None,
        score,
    )


def _write_trace(
    handle: TextIO,
    archive: DungeonArchive,
    *,
    iteration: int,
    evaluations: int,
    proposals: int,
) -> None:
    filled, coverage, mean_fitness, qd_score = _metrics(archive)
    write_archive_trace_line(
        handle,
        {
            "iteration": iteration,
            "proposals": proposals,
            "evaluations": evaluations,
            "filled_cells": filled,
            "coverage": round(coverage, 6),
            "mean_best_fitness": (
                round(mean_fitness, 6) if mean_fitness is not None else None
            ),
            "qd_score": round(qd_score, 6),
        },
    )

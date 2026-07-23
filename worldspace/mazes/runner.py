"""Domain-specific MAP-Elites loop for the maze symmetry arm."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

import numpy as np
import yaml

from worldspace.illuminators.archive_trace import write_archive_trace_line
from worldspace.mazes.archive import MazeArchive, MazeElite
from worldspace.mazes.emitters import emit_genetic, emit_random, select_uniform_frontier
from worldspace.mazes.evaluation import evaluate_maze
from worldspace.surrogate.acquisition_config import AcquisitionConfig

MazeEmitterKind = Literal["random", "genetic"]
MazeCondition = Literal[
    "random",
    "genetic",
    "genetic_filter",
    "llm_stub",
    "llm_hints",
    "llm_hints_filter",
]


@dataclass(frozen=True)
class MazeSchedulerConfig:
    condition: MazeCondition
    iterations: int = 650
    batch_size: int = 50
    archive_resolution: int = 30
    initial_random_candidates: int = 100
    emitters: tuple[MazeEmitterKind, ...] = (
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
        if self.acquisition.mode != "off":
            raise ValueError(
                "maze filter/surrogate arms are not implemented yet; "
                "use acquisition.mode=off"
            )
        if any(kind not in ("random", "genetic") for kind in self.emitters):
            raise ValueError("maze emitters currently support only random and genetic")


@dataclass(frozen=True)
class MazeRunResult:
    condition: MazeCondition
    seed: int
    proposals: int
    evaluations: int
    skipped: int
    filled_cells: int
    coverage: float
    mean_best_fitness: float | None
    qd_score: float
    elapsed_seconds: float


def load_maze_scheduler(path: Path) -> MazeSchedulerConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("maze scheduler root must be a mapping")
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
    config = MazeSchedulerConfig(
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
) -> tuple[MazeEmitterKind, ...]:
    if batch_size != 50:
        raise ValueError("implicit emitter layout requires batch_size 50")
    if condition == "random":
        return ("random",) * 50
    return ("random",) * 20 + ("genetic",) * 30


def run_maze_qd(
    config: MazeSchedulerConfig,
    *,
    seed: int,
    output_dir: Path,
) -> MazeRunResult:
    """Run one exact-proposal maze seed and write the standard artifacts."""
    config.validate()
    summary_path = output_dir / "nightly_run_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"completed run already exists: {summary_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = MazeArchive(config.archive_resolution)
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
            for slot, configured_kind in enumerate(config.emitters):
                target = select_uniform_frontier(archive, rng)
                kind: MazeEmitterKind = (
                    "random"
                    if proposals + slot < config.initial_random_candidates
                    else configured_kind
                )
                slot_rng = np.random.default_rng(
                    int(rng.integers(0, np.iinfo(np.int64).max))
                )
                if kind == "genetic":
                    emitted = emit_genetic(target, archive, slot_rng)
                else:
                    emitted = emit_random(slot_rng)
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
                            "prediction": None,
                            "decision": None,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                evaluation = evaluate_maze(emitted.spec)
                evaluations += 1
                elite = MazeElite(
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
    archive_path = output_dir / "maze_archive.jsonl"
    archive.write_jsonl(archive_path)
    result = MazeRunResult(
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
        "schema_version": "maze-1.0",
        "scheduler": f"maze:{config.condition}",
        "condition": config.condition,
        "benchmark": "maze",
        "maze_benchmark": True,
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
        "llm_enabled": False,
        "surrogate_enabled": False,
        "archive_jsonl": str(archive_path.resolve()),
        "archive_trace": str(trace_path.resolve()),
        "surrogate_archive": str(surrogate_path.resolve()),
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return result


def _metrics(
    archive: MazeArchive,
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
    archive: MazeArchive,
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

"""Sphere RQ1 / H1: MAP-Elites with minfit vs uniform and optional LLM scalars.

Literature Fontaine Sphere (D=20, 100×100). Not Holm. Does not amend Sphere H2,
CA H1, or maze empty 2×2. Protocol: artifacts/Q1_RQ1_SPHERE_DOMAIN.md
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TextIO

import joblib
import numpy as np
import yaml
from numpy.typing import NDArray
from ribs.archives import GridArchive
from sklearn.neural_network import MLPRegressor

from worldspace.benchmarks.qd_sphere import (
    CLIP_BOUND,
    DEFAULT_ARCHIVE_DIMS,
    DEFAULT_SOLUTION_DIM,
    SPHERE_SHIFT,
    archive_ranges,
    clip_solution,
    linear_projection_measures,
    sphere_objective,
)
from worldspace.illuminators.archive_trace import write_archive_trace_line

FloatArray = NDArray[np.float64]
SphereEmitterKind = Literal["random", "genetic", "llm"]
SphereCondition = Literal[
    "genetic",
    "genetic_minfit",
    "llm_stub_minfit",
    "llm_stub_uniform",
    "llm_hints_minfit",
    "llm_hints_uniform",
]
SphereTargetSelection = Literal[
    "min_fitness_frontier",
    "uniform_frontier",
    "max_fitness_frontier",
]
DEFAULT_SPHERE_TARGET_SELECTION: SphereTargetSelection = "uniform_frontier"
DEFAULT_SIGMA = 0.5
DEFAULT_H1_CHECKPOINT = Path("artifacts/surrogate/sphere_h1_mlp.joblib")
STUB_FITNESS = 0.5
STUB_UNCERTAINTY = 1.0
ARCHIVE_TRACE_FILENAME = "archive_trace.jsonl"

__all__ = [
    "DEFAULT_H1_CHECKPOINT",
    "DEFAULT_SIGMA",
    "STUB_FITNESS",
    "STUB_UNCERTAINTY",
    "SphereElite",
    "SphereEmitterResult",
    "SphereH1Surrogate",
    "SpherePrediction",
    "SphereRunResult",
    "SphereSchedulerConfig",
    "SphereTarget",
    "emit_genetic",
    "emit_random",
    "load_sphere_h1_surrogate",
    "load_sphere_scheduler",
    "run_sphere_qd",
    "save_sphere_h1_surrogate",
    "select_target_cell",
    "train_sphere_h1_surrogate",
]


@dataclass(frozen=True)
class SpherePrediction:
    fitness: float
    uncertainty: float


@dataclass(frozen=True)
class SphereElite:
    cell_id: int
    bin: tuple[int, int]
    center: tuple[float, float]
    solution: tuple[float, ...]
    objective: float
    measures: tuple[float, float]
    candidate_id: str


@dataclass(frozen=True)
class SphereTarget:
    cell_id: int
    bin: tuple[int, int]
    center: tuple[float, float]
    parent: SphereElite | None


@dataclass(frozen=True)
class SphereEmitterResult:
    solution: FloatArray
    parent_id: str | None
    emitter_type: str


@dataclass
class SphereH1Surrogate:
    """Ensemble MLP: clipped θ → predicted sphere objective in [0, 100]."""

    models: list[MLPRegressor]
    train_mae: float
    n_train: int
    n_members: int

    def predict(self, solution: FloatArray) -> SpherePrediction:
        clipped = clip_solution(solution)
        batch = clipped if clipped.ndim == 2 else clipped[np.newaxis, :]
        members = np.stack(
            [np.asarray(model.predict(batch), dtype=np.float64) for model in self.models]
        )
        mean = float(np.mean(members[:, 0]))
        std = float(np.std(members[:, 0], ddof=0))
        return SpherePrediction(
            fitness=float(np.clip(mean / 100.0, 0.0, 1.0)),
            uncertainty=float(np.clip(std / 100.0, 0.0, 1.0)),
        )


class SpherePredictor(Protocol):
    def predict(self, solution: FloatArray) -> SpherePrediction: ...


class SphereLlmEmitterProtocol(Protocol):
    def emit(
        self,
        *,
        target: SphereTarget,
        rng: np.random.Generator,
        prediction: SpherePrediction | None,
    ) -> SphereEmitterResult: ...

    def emit_batch(
        self,
        jobs: list[tuple[SphereTarget, np.random.Generator, SpherePrediction | None]],
        *,
        max_workers: int = 4,
    ) -> list[SphereEmitterResult]: ...


@dataclass(frozen=True)
class SphereSchedulerConfig:
    condition: SphereCondition
    iterations: int = 100
    batch_size: int = 50
    archive_dims: tuple[int, int] = DEFAULT_ARCHIVE_DIMS
    solution_dim: int = DEFAULT_SOLUTION_DIM
    sigma: float = DEFAULT_SIGMA
    initial_random_candidates: int = 100
    emitters: tuple[SphereEmitterKind, ...] = (
        *("random" for _ in range(20)),
        *("genetic" for _ in range(30)),
    )
    surrogate_checkpoint: str | None = None
    llm_prompt_mode: Literal["off", "stub", "hints"] = "off"
    target_selection: SphereTargetSelection = DEFAULT_SPHERE_TARGET_SELECTION

    def validate(self) -> None:
        if self.iterations < 1 or self.batch_size < 1:
            raise ValueError("iterations and batch_size must be positive")
        if len(self.emitters) != self.batch_size:
            raise ValueError("emitters length must equal batch_size")
        if self.solution_dim < 2 or self.solution_dim % 2:
            raise ValueError("solution_dim must be a positive even integer")
        if any(dim < 1 for dim in self.archive_dims):
            raise ValueError("archive_dims must be positive")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if "llm" in self.emitters and self.llm_prompt_mode == "off":
            raise ValueError("LLM emitters require stub or hints prompt mode")
        if self.llm_prompt_mode == "hints" and not self.surrogate_checkpoint:
            raise ValueError("hints mode requires surrogate_checkpoint")
        if self.target_selection not in (
            "min_fitness_frontier",
            "uniform_frontier",
            "max_fitness_frontier",
        ):
            raise ValueError(f"unknown target_selection {self.target_selection!r}")


@dataclass(frozen=True)
class SphereRunResult:
    condition: SphereCondition
    seed: int
    proposals: int
    evaluations: int
    filled_cells: int
    coverage: float
    mean_best_fitness: float | None
    qd_score: float
    elapsed_seconds: float


def train_sphere_h1_surrogate(
    *,
    seed: int = 0,
    n_train: int = 20_000,
    n_members: int = 3,
    solution_dim: int = DEFAULT_SOLUTION_DIM,
) -> SphereH1Surrogate:
    """Fit a small ensemble on random box samples with analytic labels."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-CLIP_BOUND, CLIP_BOUND, size=(n_train, solution_dim))
    y = np.asarray(sphere_objective(x), dtype=np.float64)
    models: list[MLPRegressor] = []
    for member in range(n_members):
        model = MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="adam",
            max_iter=200,
            random_state=seed + member,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        model.fit(x, y)
        models.append(model)
    stacked = np.stack(
        [np.asarray(model.predict(x), dtype=np.float64) for model in models]
    )
    mean = np.mean(stacked, axis=0)
    mae = float(np.mean(np.abs(mean - y)))
    return SphereH1Surrogate(
        models=models,
        train_mae=mae,
        n_train=n_train,
        n_members=n_members,
    )


def load_sphere_h1_surrogate(path: Path) -> SphereH1Surrogate:
    blob = joblib.load(path)
    if isinstance(blob, SphereH1Surrogate):
        return blob
    if not isinstance(blob, dict) or "models" not in blob:
        raise TypeError(f"invalid Sphere H1 surrogate checkpoint: {path}")
    return SphereH1Surrogate(
        models=list(blob["models"]),
        train_mae=float(blob.get("train_mae", 0.0)),
        n_train=int(blob.get("n_train", 0)),
        n_members=int(blob.get("n_members", len(blob["models"]))),
    )


def save_sphere_h1_surrogate(surrogate: SphereH1Surrogate, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": surrogate.models,
            "train_mae": surrogate.train_mae,
            "n_train": surrogate.n_train,
            "n_members": surrogate.n_members,
            "schema": "sphere-h1-ensemble-v1",
        },
        path,
    )


def load_sphere_scheduler(path: Path) -> SphereSchedulerConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("sphere scheduler root must be a mapping")
    condition = _yaml_enum(raw["condition"])
    batch_size = int(raw.get("batch_size", 50))
    emitters_raw = raw.get("emitters")
    emitters = (
        tuple(emitters_raw)
        if isinstance(emitters_raw, list)
        else _default_emitters(condition, batch_size)
    )
    dims_raw = raw.get("archive_dims", list(DEFAULT_ARCHIVE_DIMS))
    if not isinstance(dims_raw, list) or len(dims_raw) != 2:
        raise ValueError("archive_dims must be a two-item list")
    config = SphereSchedulerConfig(
        condition=condition,  # type: ignore[arg-type]
        iterations=int(raw.get("iterations", 100)),
        batch_size=batch_size,
        archive_dims=(int(dims_raw[0]), int(dims_raw[1])),
        solution_dim=int(raw.get("solution_dim", DEFAULT_SOLUTION_DIM)),
        sigma=float(raw.get("sigma", DEFAULT_SIGMA)),
        initial_random_candidates=int(raw.get("initial_random_candidates", 100)),
        emitters=emitters,  # type: ignore[arg-type]
        surrogate_checkpoint=raw.get("surrogate_checkpoint"),
        llm_prompt_mode=_yaml_enum(raw.get("llm_prompt_mode", "off")),  # type: ignore[arg-type]
        target_selection=_yaml_enum(  # type: ignore[arg-type]
            raw.get("target_selection", DEFAULT_SPHERE_TARGET_SELECTION)
        ),
    )
    config.validate()
    return config


def select_target_cell(
    archive: GridArchive,
    rng: np.random.Generator,
    *,
    target_selection: SphereTargetSelection = DEFAULT_SPHERE_TARGET_SELECTION,
    occupied: dict[int, SphereElite] | None = None,
    frontier: set[int] | None = None,
) -> SphereTarget:
    """Select a Sphere archive target under min / uniform / max frontier policy."""
    n_cells = int(np.prod(archive.dims))
    if occupied is None:
        snapshot = _elite_snapshot(archive)
        occupied = {} if snapshot is None else snapshot.elites
    if not occupied:
        cell_id = int(rng.integers(0, n_cells))
        return SphereTarget(
            cell_id=cell_id,
            bin=_bin_from_cell_id(archive, cell_id),
            center=_cell_center(archive, cell_id),
            parent=None,
        )
    if frontier is None:
        occupied_set = set(occupied)
        frontier_ids = [
            cell_id
            for cell_id in occupied
            if _has_empty_cardinal_neighbor(archive, cell_id, occupied_set)
        ]
        pool = frontier_ids or list(occupied)
    else:
        pool = list(frontier) or list(occupied)
    if target_selection == "uniform_frontier":
        cell_id = int(pool[int(rng.integers(0, len(pool)))])
    elif target_selection == "min_fitness_frontier":
        cell_id = _extremal_occupied_cell(pool, occupied, maximize=False)
    elif target_selection == "max_fitness_frontier":
        cell_id = _extremal_occupied_cell(pool, occupied, maximize=True)
    else:
        raise ValueError(f"unknown target_selection {target_selection!r}")
    return SphereTarget(
        cell_id=cell_id,
        bin=_bin_from_cell_id(archive, cell_id),
        center=_cell_center(archive, cell_id),
        parent=occupied[cell_id],
    )


def emit_random(
    rng: np.random.Generator,
    *,
    solution_dim: int = DEFAULT_SOLUTION_DIM,
) -> SphereEmitterResult:
    solution = rng.uniform(-CLIP_BOUND, CLIP_BOUND, size=solution_dim)
    return SphereEmitterResult(
        solution=clip_solution(solution),
        parent_id=None,
        emitter_type="random",
    )


def emit_genetic(
    target: SphereTarget,
    rng: np.random.Generator,
    *,
    sigma: float = DEFAULT_SIGMA,
    solution_dim: int = DEFAULT_SOLUTION_DIM,
) -> SphereEmitterResult:
    if target.parent is None:
        return emit_random(rng, solution_dim=solution_dim)
    parent = np.asarray(target.parent.solution, dtype=np.float64)
    child = clip_solution(parent + rng.normal(0.0, sigma, size=parent.shape))
    return SphereEmitterResult(
        solution=child,
        parent_id=target.parent.candidate_id,
        emitter_type="genetic",
    )


def run_sphere_qd(
    config: SphereSchedulerConfig,
    *,
    seed: int,
    output_dir: Path,
    predictor: SpherePredictor | None = None,
    llm_emitter: SphereLlmEmitterProtocol | None = None,
) -> SphereRunResult:
    """Run one exact-proposal Sphere seed and write the standard artifacts."""
    config.validate()
    if predictor is None and config.surrogate_checkpoint:
        predictor = load_sphere_h1_surrogate(Path(config.surrogate_checkpoint))
    summary_path = output_dir / "nightly_run_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"completed run already exists: {summary_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = GridArchive(
        solution_dim=config.solution_dim,
        dims=config.archive_dims,
        ranges=archive_ranges(config.solution_dim),
        seed=seed,
        learning_rate=1.0,
    )
    rng = np.random.default_rng(seed)
    n_cells = int(np.prod(config.archive_dims))
    occupied: dict[int, SphereElite] = {}
    frontier: set[int] = set()
    proposals = evaluations = 0
    started = time.perf_counter()
    trace_path = output_dir / ARCHIVE_TRACE_FILENAME
    surrogate_path = output_dir / "surrogate_archive.jsonl"
    with (
        trace_path.open("w", encoding="utf-8") as trace_file,
        surrogate_path.open("w", encoding="utf-8") as surrogate_file,
    ):
        _write_trace(
            trace_file,
            archive,
            n_cells=n_cells,
            iteration=0,
            evaluations=0,
            proposals=0,
        )
        for iteration in range(config.iterations):
            plans: list[
                tuple[
                    int,
                    SphereTarget,
                    SphereEmitterKind,
                    np.random.Generator,
                    SpherePrediction | None,
                ]
            ] = []
            emitted_batch: list[SphereEmitterResult | None] = [None] * config.batch_size
            llm_jobs: list[
                tuple[SphereTarget, np.random.Generator, SpherePrediction | None]
            ] = []
            llm_slots: list[int] = []
            for slot, configured_kind in enumerate(config.emitters):
                target = select_target_cell(
                    archive,
                    rng,
                    target_selection=config.target_selection,
                    occupied=occupied,
                    frontier=frontier,
                )
                kind: SphereEmitterKind = (
                    "random"
                    if proposals + slot < config.initial_random_candidates
                    else configured_kind
                )
                slot_rng = np.random.default_rng(
                    int(rng.integers(0, np.iinfo(np.int64).max))
                )
                parent_prediction = None
                if (
                    kind == "llm"
                    and predictor is not None
                    and target.parent is not None
                ):
                    parent_prediction = predictor.predict(
                        np.asarray(target.parent.solution, dtype=np.float64)
                    )
                plans.append((slot, target, kind, slot_rng, parent_prediction))
                if kind == "llm":
                    if llm_emitter is None:
                        raise ValueError("LLM condition requires llm_emitter")
                    llm_slots.append(slot)
                    llm_jobs.append((target, slot_rng, parent_prediction))
                elif kind == "genetic":
                    emitted_batch[slot] = emit_genetic(
                        target,
                        slot_rng,
                        sigma=config.sigma,
                        solution_dim=config.solution_dim,
                    )
                else:
                    emitted_batch[slot] = emit_random(
                        slot_rng, solution_dim=config.solution_dim
                    )
            if llm_jobs:
                assert llm_emitter is not None
                llm_results = llm_emitter.emit_batch(
                    llm_jobs, max_workers=_llm_max_workers()
                )
                for slot, emitted in zip(llm_slots, llm_results, strict=True):
                    emitted_batch[slot] = emitted

            for slot, target, _, _, parent_prediction in plans:
                emitted = emitted_batch[slot]
                assert emitted is not None
                child = clip_solution(emitted.solution)
                objective = float(sphere_objective(child))
                measures = linear_projection_measures(child)
                proposals += 1
                evaluations += 1
                surrogate_file.write(
                    json.dumps(
                        {
                            "proposal": proposals,
                            "iteration": iteration,
                            "slot": slot,
                            "target_cell": target.cell_id,
                            "target_bin": list(target.bin),
                            "emitter_type": emitted.emitter_type,
                            "parent_id": emitted.parent_id,
                            "prediction": (
                                {
                                    "fitness": parent_prediction.fitness,
                                    "uncertainty": parent_prediction.uncertainty,
                                }
                                if parent_prediction is not None
                                else None
                            ),
                            "objective": objective,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                add_info = archive.add(
                    child[np.newaxis, :],
                    np.asarray([objective], dtype=np.float64),
                    measures[np.newaxis, :],
                )
                status = int(np.asarray(add_info["status"]).reshape(-1)[0])
                if status > 0:
                    cell_id = int(archive.index_of(measures[np.newaxis, :])[0])
                    occupied[cell_id] = SphereElite(
                        cell_id=cell_id,
                        bin=_bin_from_cell_id(archive, cell_id),
                        center=_cell_center(archive, cell_id),
                        solution=tuple(float(x) for x in child),
                        objective=objective,
                        measures=(float(measures[0]), float(measures[1])),
                        candidate_id=f"cell-{cell_id}",
                    )
                    _refresh_frontier(archive, occupied, frontier, cell_id)
            _write_trace(
                trace_file,
                archive,
                n_cells=n_cells,
                iteration=iteration + 1,
                evaluations=evaluations,
                proposals=proposals,
            )
    elapsed = time.perf_counter() - started
    filled, coverage, mean_fitness, qd_score = _archive_metrics(archive, n_cells)
    result = SphereRunResult(
        condition=config.condition,
        seed=seed,
        proposals=proposals,
        evaluations=evaluations,
        filled_cells=filled,
        coverage=coverage,
        mean_best_fitness=mean_fitness,
        qd_score=qd_score,
        elapsed_seconds=round(elapsed, 3),
    )
    archive_arrays = {
        str(key): np.asarray(value) for key, value in archive.data().items()
    }
    np.savez_compressed(
        str(output_dir / "pyribs_archive.npz"),
        allow_pickle=False,
        **archive_arrays,
    )
    payload: dict[str, object] = {
        "schema_version": "sphere-rq1-1.0",
        "scheduler": f"sphere:{config.condition}",
        "condition": config.condition,
        "benchmark": "sphere",
        "study": "sphere_rq1_h1",
        "seed": seed,
        "iterations": config.iterations,
        "proposals": proposals,
        "evaluations": evaluations,
        "skipped": 0,
        "skip_rate": 0.0,
        "filled_cells": filled,
        "coverage": coverage,
        "mean_best_fitness": mean_fitness,
        "qd_score": qd_score,
        "elapsed_seconds": result.elapsed_seconds,
        "archive_type": "grid",
        "archive_dims": list(config.archive_dims),
        "n_cells": n_cells,
        "solution_dim": config.solution_dim,
        "sigma": config.sigma,
        "llm_enabled": "llm" in config.emitters,
        "surrogate_enabled": predictor is not None,
        "archive_trace": str(trace_path.resolve()),
        "surrogate_archive": str(surrogate_path.resolve()),
        "target_selection": config.target_selection,
        "llm_prompt_mode": config.llm_prompt_mode,
        "surrogate_checkpoint": config.surrogate_checkpoint,
    }
    llm_config = getattr(llm_emitter, "config", None)
    if llm_config is not None:
        provider = getattr(llm_config, "active_provider", None)
        payload["llm_provider"] = provider
        spec_path = getattr(llm_emitter, "llm_spec_path", None)
        if spec_path is not None:
            payload["llm_spec"] = str(spec_path)
        providers = getattr(llm_config, "providers", None) or {}
        if provider and isinstance(providers, dict):
            active = providers.get(provider) or {}
            if isinstance(active, dict) and active.get("model"):
                payload["llm_model"] = active["model"]
    llm_audit = getattr(llm_emitter, "audit", None)
    if llm_audit is not None and hasattr(llm_audit, "to_dict"):
        audit_payload = llm_audit.to_dict()
        payload["llm_audit"] = audit_payload
        payload["llm_calls"] = audit_payload.get(
            "api_calls",
            audit_payload["attempts"],
        )
        payload["llm_fallback_rate"] = audit_payload["fallback_rate"]
        payload["llm_parse_success_rate"] = audit_payload["parse_success_rate"]
        payload["llm_mean_l2"] = audit_payload.get("mean_l2", 0.0)
    prompt_version = getattr(llm_emitter, "prompt_version", None)
    if prompt_version is not None:
        payload["prompt_version"] = prompt_version
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return result


@dataclass
class _EliteSnapshot:
    cell_ids: list[int]
    elites: dict[int, SphereElite]


def _elite_snapshot(archive: GridArchive) -> _EliteSnapshot | None:
    if archive.stats.num_elites == 0:
        return None
    data = archive.data()
    indices = np.asarray(data["index"], dtype=np.int64)
    objectives = np.asarray(data["objective"], dtype=np.float64)
    solutions = np.asarray(data["solution"], dtype=np.float64)
    measures = np.asarray(data["measures"], dtype=np.float64)
    elites: dict[int, SphereElite] = {}
    cell_ids: list[int] = []
    for row, cell_id in enumerate(indices.tolist()):
        cid = int(cell_id)
        cell_ids.append(cid)
        elites[cid] = SphereElite(
            cell_id=cid,
            bin=_bin_from_cell_id(archive, cid),
            center=_cell_center(archive, cid),
            solution=tuple(float(x) for x in solutions[row]),
            objective=float(objectives[row]),
            measures=(float(measures[row, 0]), float(measures[row, 1])),
            candidate_id=f"cell-{cid}",
        )
    return _EliteSnapshot(cell_ids=cell_ids, elites=elites)


def _refresh_frontier(
    archive: GridArchive,
    occupied: dict[int, SphereElite],
    frontier: set[int],
    cell_id: int,
) -> None:
    """Update the frontier after ``cell_id`` is added or improved."""
    i, j = _bin_from_cell_id(archive, cell_id)
    d0, d1 = (int(archive.dims[0]), int(archive.dims[1]))
    to_check = [cell_id]
    for ni, nj in _cardinal_neighbors(i, j, d0, d1):
        to_check.append(_index_from_bin(archive, ni, nj))
    for cid in to_check:
        if cid not in occupied:
            frontier.discard(cid)
            continue
        if _has_empty_cardinal_neighbor(archive, cid, occupied):
            frontier.add(cid)
        else:
            frontier.discard(cid)


def _has_empty_cardinal_neighbor(
    archive: GridArchive,
    cell_id: int,
    occupied: dict[int, SphereElite] | set[int],
) -> bool:
    i, j = _bin_from_cell_id(archive, cell_id)
    d0, d1 = (int(archive.dims[0]), int(archive.dims[1]))
    for ni, nj in _cardinal_neighbors(i, j, d0, d1):
        neighbor = _index_from_bin(archive, ni, nj)
        if neighbor not in occupied:
            return True
    return False


def _cardinal_neighbors(
    i: int, j: int, d0: int, d1: int
) -> tuple[tuple[int, int], ...]:
    neighbors: list[tuple[int, int]] = []
    if i > 0:
        neighbors.append((i - 1, j))
    if i + 1 < d0:
        neighbors.append((i + 1, j))
    if j > 0:
        neighbors.append((i, j - 1))
    if j + 1 < d1:
        neighbors.append((i, j + 1))
    return tuple(neighbors)


def _bin_from_cell_id(archive: GridArchive, cell_id: int) -> tuple[int, int]:
    grid = np.asarray(archive.int_to_grid_index(np.asarray([cell_id], dtype=np.int32)))
    return int(grid[0, 0]), int(grid[0, 1])


def _index_from_bin(archive: GridArchive, i: int, j: int) -> int:
    return int(
        archive.grid_to_int_index(np.asarray([[i, j]], dtype=np.int32))[0]
    )


def _cell_center(archive: GridArchive, cell_id: int) -> tuple[float, float]:
    i, j = _bin_from_cell_id(archive, cell_id)
    b0, b1 = archive.boundaries
    return (
        0.5 * (float(b0[i]) + float(b0[i + 1])),
        0.5 * (float(b1[j]) + float(b1[j + 1])),
    )


def _extremal_occupied_cell(
    pool: list[int],
    occupied: dict[int, SphereElite],
    *,
    maximize: bool,
) -> int:
    best: int | None = None
    best_fitness = float("-inf") if maximize else float("inf")
    for cell_id in pool:
        elite = occupied[cell_id]
        better = (
            elite.objective > best_fitness
            if maximize
            else elite.objective < best_fitness
        )
        tied = elite.objective == best_fitness and (best is None or cell_id < best)
        if better or tied:
            best = cell_id
            best_fitness = elite.objective
    if best is None:
        raise RuntimeError("frontier cells must contain elites")
    return best


def _archive_metrics(
    archive: GridArchive,
    n_cells: int,
) -> tuple[int, float, float | None, float]:
    objectives = np.asarray(archive.data("objective"), dtype=np.float64)
    filled = int(objectives.size)
    coverage = float(filled) / float(n_cells)
    score = float(np.sum(objectives)) if filled else 0.0
    mean_fitness = float(np.mean(objectives)) if filled else None
    return filled, coverage, mean_fitness, score


def _write_trace(
    handle: TextIO,
    archive: GridArchive,
    *,
    n_cells: int,
    iteration: int,
    evaluations: int,
    proposals: int,
) -> None:
    filled, coverage, mean_fitness, qd_score = _archive_metrics(archive, n_cells)
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


def _default_emitters(
    condition: str,
    batch_size: int,
) -> tuple[SphereEmitterKind, ...]:
    if batch_size != 50:
        raise ValueError("implicit emitter layout requires batch_size 50")
    if condition.startswith("llm_"):
        return ("random",) * 20 + ("llm",) * 30
    return ("random",) * 20 + ("genetic",) * 30


def _yaml_enum(value: object) -> str:
    if value is False or value is None:
        return "off"
    return str(value).strip().lower()


def _llm_max_workers() -> int:
    raw = os.environ.get("LIFEMANIFOLD_LLM_PARALLEL_WORKERS", "4")
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        value = 4
    return max(1, value)

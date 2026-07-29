"""Sphere H2: after-generation threshold gate on Fontaine linear-projection Sphere.

Supplementary defense package (not a confirmatory Holm family). Matches the
primary-grid H2 logic---predict, skip low predicted objective, evaluate the
rest---on a literature-standard continuous QD benchmark.

H1 (prompt-side scalars) is intentionally out of scope: genotypes are anonymous
``R^20`` vectors, not named structured fields.

Design notes
------------
- Fixed **proposal** budget (default 32{,}500), same as B3 ``me_random``.
- Uniform arm evaluates every proposal; filter arm skips when
  ``pred_objective < tau``.
- Sampling mirrors pyribs ``GaussianEmitter`` (``sigma=0.5``, parent = random
  elite or ``x0``), so skips never call ``archive.add`` and cannot fill empty
  cells with sentinel objectives.
- Surrogate is a sklearn MLP fit offline on random box samples; production
  LifeManifold MLP/WorldSpec checkpoint is not reused.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
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
from worldspace.illuminators.archive_trace import (
    ARCHIVE_TRACE_FILENAME,
    write_archive_trace_line,
)

FloatArray = NDArray[np.float64]
ArmName = Literal["me_uniform", "me_filter"]

DEFAULT_PROPOSALS = 32_500
DEFAULT_SIGMA = 0.5
DEFAULT_TAU = 55.0  # calibrated for ~30% skip on random box proposals
DEFAULT_TRAIN_N = 50_000
TRACE_EVERY = 250  # proposals between trace rows

__all__ = [
    "DEFAULT_PROPOSALS",
    "DEFAULT_TAU",
    "SphereH2Config",
    "SphereH2Result",
    "SphereSurrogate",
    "calibrate_tau",
    "run_sphere_h2",
    "train_sphere_surrogate",
]


@dataclass(frozen=True)
class SphereH2Config:
    """One matched Sphere H2 seed."""

    arm: ArmName
    seed: int
    proposals: int = DEFAULT_PROPOSALS
    solution_dim: int = DEFAULT_SOLUTION_DIM
    archive_dims: tuple[int, int] = DEFAULT_ARCHIVE_DIMS
    sigma: float = DEFAULT_SIGMA
    tau: float = DEFAULT_TAU
    surrogate_path: Path | None = None


@dataclass(frozen=True)
class SphereH2Result:
    """Terminal metrics for one Sphere H2 arm/seed."""

    arm: ArmName
    seed: int
    proposals: int
    true_evaluations: int
    skips: int
    skip_rate: float
    filled_cells: int
    coverage: float
    mean_best_fitness: float | None
    qd_score: float
    elapsed_seconds: float
    tau: float


@dataclass
class SphereSurrogate:
    """Offline MLP regressor: clipped θ → predicted sphere objective."""

    model: MLPRegressor
    train_mae: float
    train_r2: float
    n_train: int

    def predict(self, solutions: FloatArray) -> FloatArray:
        clipped = clip_solution(solutions)
        batch = clipped if clipped.ndim == 2 else clipped[np.newaxis, :]
        pred = np.asarray(self.model.predict(batch), dtype=np.float64)
        return pred


def train_sphere_surrogate(
    *,
    seed: int = 0,
    n_train: int = DEFAULT_TRAIN_N,
    solution_dim: int = DEFAULT_SOLUTION_DIM,
) -> SphereSurrogate:
    """Fit an MLP on random clipped-box samples with analytic labels."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-CLIP_BOUND, CLIP_BOUND, size=(n_train, solution_dim))
    y = np.asarray(sphere_objective(x), dtype=np.float64)
    model = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        max_iter=200,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    model.fit(x, y)
    pred = np.asarray(model.predict(x), dtype=np.float64)
    mae = float(np.mean(np.abs(pred - y)))
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return SphereSurrogate(model=model, train_mae=mae, train_r2=r2, n_train=n_train)


def calibrate_tau(
    surrogate: SphereSurrogate,
    *,
    target_skip: float = 0.30,
    n_probe: int = 20_000,
    seed: int = 1,
    solution_dim: int = DEFAULT_SOLUTION_DIM,
    mode: str = "me_like",
    sigma: float = DEFAULT_SIGMA,
    me_warmup: int = 2_000,
) -> float:
    """Return objective threshold yielding ≈ ``target_skip`` skip rate.

    ``mode="box"`` uses uniform samples in the search box (too easy: ME quickly
    leaves that mass). ``mode="me_like"`` (default) gathers predictions on
    GaussianEmitter-style proposals after a short true-eval warmup, matching
    the live skip band under MAP-Elites search.
    """
    if mode == "box":
        rng = np.random.default_rng(seed)
        x = rng.uniform(-CLIP_BOUND, CLIP_BOUND, size=(n_probe, solution_dim))
        pred = surrogate.predict(x)
        return float(np.quantile(pred, target_skip))

    if mode != "me_like":
        raise ValueError(f"unknown calibrate mode: {mode}")

    # Score ME-like children across a short true-eval ME trajectory so the
    # quantile tracks live proposal quality (not just the final elite cloud).
    archive = GridArchive(
        solution_dim=solution_dim,
        dims=DEFAULT_ARCHIVE_DIMS,
        ranges=archive_ranges(solution_dim),
        seed=seed,
        learning_rate=1.0,
    )
    rng = np.random.default_rng(seed)
    x0 = np.full(solution_dim, SPHERE_SHIFT, dtype=np.float64)
    preds: list[float] = []
    for _ in range(max(me_warmup, n_probe)):
        parent = _sample_parent(archive, x0=x0, rng=rng)
        child = clip_solution(parent + rng.normal(0.0, sigma, size=parent.shape))
        preds.append(float(surrogate.predict(child)[0]))
        objective = float(sphere_objective(child))
        measures = linear_projection_measures(child)
        archive.add(
            child[np.newaxis, :],
            np.asarray([objective], dtype=np.float64),
            measures[np.newaxis, :],
        )
    return float(np.quantile(np.asarray(preds, dtype=np.float64), target_skip))


def run_sphere_h2(
    config: SphereH2Config,
    *,
    output_dir: Path,
    surrogate: SphereSurrogate | None = None,
) -> SphereH2Result:
    """Run one Sphere H2 arm with fixed proposal budget."""
    if config.proposals <= 0:
        raise ValueError("proposals must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.arm == "me_filter":
        if surrogate is None:
            if config.surrogate_path is None or not config.surrogate_path.is_file():
                raise FileNotFoundError("me_filter requires a trained surrogate path")
            surrogate = joblib.load(config.surrogate_path)

    archive = GridArchive(
        solution_dim=config.solution_dim,
        dims=config.archive_dims,
        ranges=archive_ranges(config.solution_dim),
        seed=config.seed,
        learning_rate=1.0,
    )
    rng = np.random.default_rng(config.seed)
    x0 = np.full(config.solution_dim, SPHERE_SHIFT, dtype=np.float64)
    n_cells = int(np.prod(config.archive_dims))

    true_evaluations = 0
    skips = 0
    started = time.perf_counter()
    trace_path = output_dir / ARCHIVE_TRACE_FILENAME
    occupied: set[int] = set()

    with trace_path.open("w", encoding="utf-8") as trace_file:
        _write_trace(
            trace_file,
            archive=archive,
            n_cells=n_cells,
            proposal=0,
            proposals_total=config.proposals,
            true_evaluations=0,
            skips=0,
        )
        for proposal in range(1, config.proposals + 1):
            parent = _sample_parent(archive, x0=x0, rng=rng)
            child = clip_solution(
                parent + rng.normal(0.0, config.sigma, size=parent.shape)
            )
            measures = linear_projection_measures(child)
            cell_idx = int(archive.index_of(measures[np.newaxis, :])[0])
            if config.arm == "me_filter":
                assert surrogate is not None
                pred = float(surrogate.predict(child)[0])
                empty_cell = cell_idx not in occupied
                if (not empty_cell) and pred < config.tau:
                    skips += 1
                    if proposal % TRACE_EVERY == 0 or proposal == config.proposals:
                        _write_trace(
                            trace_file,
                            archive=archive,
                            n_cells=n_cells,
                            proposal=proposal,
                            proposals_total=config.proposals,
                            true_evaluations=true_evaluations,
                            skips=skips,
                        )
                    continue

            objective = float(sphere_objective(child))
            add_info = archive.add(
                child[np.newaxis, :],
                np.asarray([objective], dtype=np.float64),
                measures[np.newaxis, :],
            )
            occupied.add(cell_idx)
            true_evaluations += 1
            if proposal % TRACE_EVERY == 0 or proposal == config.proposals:
                _write_trace(
                    trace_file,
                    archive=archive,
                    n_cells=n_cells,
                    proposal=proposal,
                    proposals_total=config.proposals,
                    true_evaluations=true_evaluations,
                    skips=skips,
                )
            del add_info

    elapsed = time.perf_counter() - started
    filled, coverage, mean_fit, qd = _archive_metrics(archive, n_cells)
    result = SphereH2Result(
        arm=config.arm,
        seed=config.seed,
        proposals=config.proposals,
        true_evaluations=true_evaluations,
        skips=skips,
        skip_rate=float(skips) / float(config.proposals),
        filled_cells=filled,
        coverage=coverage,
        mean_best_fitness=mean_fit,
        qd_score=qd,
        elapsed_seconds=round(elapsed, 3),
        tau=config.tau if config.arm == "me_filter" else float("nan"),
    )
    _write_summary(
        output_dir / "nightly_run_summary.json", config=config, result=result
    )
    archive_arrays = {
        str(key): np.asarray(value) for key, value in archive.data().items()
    }
    np.savez_compressed(
        str(output_dir / "pyribs_archive.npz"),
        allow_pickle=False,
        **archive_arrays,
    )
    return result


def _sample_parent(
    archive: GridArchive,
    *,
    x0: FloatArray,
    rng: np.random.Generator,
) -> FloatArray:
    if archive.stats.num_elites == 0:
        return x0.copy()
    elites = archive.sample_elites(1)
    return np.asarray(elites["solution"][0], dtype=np.float64)


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
    trace_file: Any,
    *,
    archive: GridArchive,
    n_cells: int,
    proposal: int,
    proposals_total: int,
    true_evaluations: int,
    skips: int,
) -> None:
    filled, coverage, mean_fit, qd = _archive_metrics(archive, n_cells)
    write_archive_trace_line(
        trace_file,
        {
            "proposal": proposal,
            "proposals_total": proposals_total,
            "evaluations": true_evaluations,
            "skips": skips,
            "filled_cells": filled,
            "coverage": round(coverage, 6),
            "mean_best_fitness": (round(mean_fit, 6) if mean_fit is not None else None),
            "qd_score": round(qd, 6),
        },
    )


def _write_summary(
    path: Path,
    *,
    config: SphereH2Config,
    result: SphereH2Result,
) -> None:
    payload = {
        "benchmark": "sphere",
        "study": "sphere_h2_filter",
        "arm": result.arm,
        "seed": result.seed,
        "proposals": result.proposals,
        "true_evaluations": result.true_evaluations,
        "skips": result.skips,
        "skip_rate": result.skip_rate,
        "coverage": result.coverage,
        "coverage_pct": 100.0 * result.coverage,
        "mean_best_fitness": result.mean_best_fitness,
        "qd_score": result.qd_score,
        "filled_cells": result.filled_cells,
        "elapsed_seconds": result.elapsed_seconds,
        "tau": result.tau,
        "sigma": config.sigma,
        "solution_dim": config.solution_dim,
        "archive_dims": list(config.archive_dims),
        "surrogate_path": (
            str(config.surrogate_path) if config.surrogate_path else None
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

"""Maze filter threshold calibration from logged proposals and buffer hold-out."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from worldspace.mazes.surrogate import MazeSurrogate, load_buffer

SHADOW_SKIP_BAND = (0.25, 0.45)
DEFAULT_MAX_UNCERTAINTY = 0.014120567094666964


@dataclass(frozen=True)
class ReplayBatch:
    name: str
    fitness: NDArray[np.float64]
    uncertainty: NDArray[np.float64]
    target_was_empty: NDArray[np.bool_]

    @property
    def n_rows(self) -> int:
        return int(self.fitness.shape[0])


@dataclass(frozen=True)
class ThresholdCandidate:
    min_predicted_fitness: float
    max_uncertainty_to_skip: float
    mean_skip_rate: float
    per_source: dict[str, float]


def load_surrogate_archive(path: Path, *, name: str | None = None) -> ReplayBatch:
    fitness: list[float] = []
    uncertainty: list[float] = []
    target_was_empty: list[bool] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prediction = row["prediction"]
        fitness.append(float(prediction["fitness"]))
        uncertainty.append(float(prediction["uncertainty"]))
        target_was_empty.append(bool(row["target_was_empty"]))
    if not fitness:
        raise ValueError(f"empty surrogate archive: {path}")
    return ReplayBatch(
        name=name or path.parent.parent.name,
        fitness=np.asarray(fitness, dtype=np.float64),
        uncertainty=np.asarray(uncertainty, dtype=np.float64),
        target_was_empty=np.asarray(target_was_empty, dtype=np.bool_),
    )


def buffer_holdout_batch(
    buffer_path: Path,
    checkpoint_path: Path,
    *,
    random_state: int = 1729,
) -> ReplayBatch:
    features, _targets = load_buffer(buffer_path)
    surrogate = MazeSurrogate.load(checkpoint_path)
    rng = np.random.default_rng(random_state)
    order = rng.permutation(features.shape[0])
    split = max(1, int(features.shape[0] * 0.8))
    holdout_idx = order[split:]
    if holdout_idx.size == 0:
        raise ValueError("buffer hold-out split is empty")
    transformed = surrogate.checkpoint.scaler.transform(features[holdout_idx])
    members = np.stack(
        [
            np.asarray(model.predict(transformed))[:, 0]
            for model in surrogate.checkpoint.models
        ]
    )
    fitness = np.clip(np.mean(members, axis=0), 0.0, 1.0)
    uncertainty = (
        np.std(members, axis=0, ddof=0) * surrogate.checkpoint.uncertainty_scale
    )
    return ReplayBatch(
        name="buffer_holdout",
        fitness=fitness,
        uncertainty=uncertainty,
        target_was_empty=np.zeros(holdout_idx.size, dtype=np.bool_),
    )


def replay_skip_rate(
    batch: ReplayBatch,
    *,
    min_predicted_fitness: float,
    max_uncertainty_to_skip: float,
) -> float:
    occupied = ~batch.target_was_empty
    low_fitness = batch.fitness < min_predicted_fitness
    low_uncertainty = batch.uncertainty <= max_uncertainty_to_skip
    return float(np.mean(occupied & low_fitness & low_uncertainty))


def in_shadow_skip_band(skip_rate: float) -> bool:
    return SHADOW_SKIP_BAND[0] <= skip_rate <= SHADOW_SKIP_BAND[1]


def search_fitness_threshold(
    live_batches: tuple[ReplayBatch, ...],
    *,
    max_uncertainty_to_skip: float = DEFAULT_MAX_UNCERTAINTY,
    tau_min: float = 0.45,
    tau_max: float = 0.80,
    tau_step: float = 0.005,
    target_skip: float = 0.35,
) -> ThresholdCandidate:
    if not live_batches:
        raise ValueError("live_batches must not be empty")
    candidates: list[ThresholdCandidate] = []
    for tau in np.arange(tau_min, tau_max + 1e-9, tau_step):
        per_source = {
            batch.name: replay_skip_rate(
                batch,
                min_predicted_fitness=float(tau),
                max_uncertainty_to_skip=max_uncertainty_to_skip,
            )
            for batch in live_batches
        }
        mean_skip = float(np.mean(list(per_source.values())))
        if all(in_shadow_skip_band(rate) for rate in per_source.values()):
            candidates.append(
                ThresholdCandidate(
                    min_predicted_fitness=round(float(tau), 4),
                    max_uncertainty_to_skip=max_uncertainty_to_skip,
                    mean_skip_rate=mean_skip,
                    per_source=per_source,
                )
            )
    if not candidates:
        raise ValueError(
            "no threshold places all live replay sources in the 25–45% skip band"
        )
    return min(candidates, key=lambda item: abs(item.mean_skip_rate - target_skip))

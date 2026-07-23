"""Native discrete-search CMA-ME emitter for pyribs (bit-flip + adaptive step size).

CMA proposes and stores discrete rule bits in {0, 1} (same encoding as genetic ME)
with Gaussian mutation on the three float scalars. Step size adapts via a 1/5
success rule on archive insertions (filter selection) or ranked parents (mu selection).
This is a discrete variation operator with CMA-style sigma control — not continuous
relaxation with rint/Bernoulli decode.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable, Collection
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike

from ribs._utils import check_shape, validate_batch
from ribs.archives import ArchiveBase
from ribs.emitters._emitter_base import EmitterBase
from ribs.emitters.rankers import RankerBase, _get_ranker
from ribs.typing import BatchData, Float, Int

from worldspace.illuminators.emitters.genetics import GENOME_SIZE
from worldspace.specs.world_param_bounds import (
    FLOAT_PARAM_BOUNDS,
    RULE_BIT_MAX,
    RULE_BIT_MIN,
    clip_genome_float_params,
)

__all__ = ["DiscreteCMAEmitter", "discrete_x0"]

_FLOAT_GENE_START = 18
_BIT_FLIP_SCALE = 5.0

_SIGMA_MIN = 0.01
_SIGMA_MAX = 0.5
_SUCCESS_TARGET = 0.2


def discrete_x0() -> np.ndarray:
    """Initial discrete mean: thresholded mid rule bits + mid float params."""
    x0 = np.empty(GENOME_SIZE, dtype=np.float64)
    x0[:_FLOAT_GENE_START] = 0.0
    for index, (lo, hi) in enumerate(FLOAT_PARAM_BOUNDS, start=_FLOAT_GENE_START):
        x0[index] = 0.5 * (lo + hi)
    return x0


def _quantize_rule_bits(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    out[:_FLOAT_GENE_START] = np.clip(
        np.rint(out[:_FLOAT_GENE_START]), RULE_BIT_MIN, RULE_BIT_MAX
    )
    return clip_genome_float_params(out, start_index=_FLOAT_GENE_START)


def _flip_probability(sigma: float) -> float:
    return float(np.clip(sigma * _BIT_FLIP_SCALE, RULE_BIT_MIN, 0.49))


class DiscreteCMAEmitter(EmitterBase):
    """Bit-flip + float-Gaussian emitter with adaptive step size for CMA-ME."""

    def __init__(
        self,
        archive: ArchiveBase,
        *,
        x0: ArrayLike,
        sigma0: Float,
        ranker: Callable[[Int | None], RankerBase] | str = "2imp",
        selection_rule: Literal["mu", "filter"] = "filter",
        restart_rule: Literal["no_improvement", "basic"] | int = "no_improvement",
        bounds: Collection[tuple[None | Float, None | Float]] | None = None,
        lower_bounds: ArrayLike | None = None,
        upper_bounds: ArrayLike | None = None,
        batch_size: Int | None = None,
        seed: Int | None = None,
    ) -> None:
        EmitterBase.__init__(
            self,
            archive,
            solution_dim=archive.solution_dim,
            bounds=bounds,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        seed_sequence = (
            seed
            if isinstance(seed, np.random.SeedSequence)
            else np.random.SeedSequence(seed)
        )
        ranker_seed, rng_seed = seed_sequence.spawn(2)
        self._rng = np.random.default_rng(rng_seed)

        self._x0 = _quantize_rule_bits(np.asarray(x0, dtype=archive.dtypes["solution"]))
        check_shape(self._x0, "x0", archive.solution_dim, "archive.solution_dim")
        self._mean = self._x0.copy()
        self._sigma = float(np.clip(sigma0, _SIGMA_MIN, _SIGMA_MAX))

        if selection_rule not in ["mu", "filter"]:
            raise ValueError(f"Invalid selection_rule {selection_rule}")
        self._selection_rule = selection_rule

        self._restart_rule = restart_rule
        self._restarts = 0
        self._itrs = 0

        _ = self._check_restart(0)

        if batch_size is None or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._batch_size = int(batch_size)

        self._ranker = _get_ranker(ranker, cast(Int | None, ranker_seed))
        self._ranker.reset(self, archive)

    @property
    def x0(self) -> np.ndarray:
        return self._x0

    @property
    def batch_size(self) -> Int:
        return self._batch_size

    @property
    def restarts(self) -> int:
        return self._restarts

    @property
    def itrs(self) -> int:
        return self._itrs

    def ask(self) -> np.ndarray:
        flip_prob = _flip_probability(self._sigma)
        float_sigma = self._sigma
        offspring = np.tile(self._mean, (self._batch_size, 1))
        for index in range(self._batch_size):
            row = offspring[index]
            for bit_index in range(_FLOAT_GENE_START):
                if self._rng.random() < flip_prob:
                    row[bit_index] = RULE_BIT_MAX - row[bit_index]
            row[_FLOAT_GENE_START:] += self._rng.normal(
                0.0, float_sigma, size=len(FLOAT_PARAM_BOUNDS)
            )
            offspring[index] = _quantize_rule_bits(row)
        return offspring.astype(self.archive.dtypes["solution"], copy=False)

    def _check_restart(self, num_parents: int) -> bool:
        if isinstance(self._restart_rule, numbers.Integral):
            return self._itrs > 0 and self._itrs % self._restart_rule == 0
        if self._restart_rule == "no_improvement":
            return num_parents == 0
        if self._restart_rule == "basic":
            return False
        raise ValueError(f"Invalid restart_rule {self._restart_rule}")

    def _adapt_sigma(self, *, num_parents: int) -> None:
        success_rate = num_parents / max(self._batch_size, 1)
        if success_rate > _SUCCESS_TARGET:
            self._sigma = min(self._sigma * np.exp(0.2), _SIGMA_MAX)
        else:
            self._sigma = max(self._sigma / np.exp(0.2), _SIGMA_MIN)

    def _update_mean(self, parents: np.ndarray) -> None:
        if parents.shape[0] == 0:
            return
        weights = np.linspace(1.0, 0.5, parents.shape[0], dtype=np.float64)
        weights /= weights.sum()
        blended = np.average(parents, axis=0, weights=weights)
        self._mean = _quantize_rule_bits(blended)

    def tell(
        self,
        solution: ArrayLike,
        objective: ArrayLike,
        measures: ArrayLike,
        add_info: BatchData,
        **fields: ArrayLike,
    ) -> None:
        batch: BatchData = {
            "solution": np.asarray(solution, dtype=self.archive.dtypes["solution"]),
            "objective": np.asarray(objective, dtype=self.archive.dtypes["objective"]),
            "measures": np.asarray(measures, dtype=self.archive.dtypes["measures"]),
        }
        for name, value in fields.items():
            batch[name] = np.asarray(value)

        data, validated_add_info = cast(
            tuple[BatchData, BatchData],
            validate_batch(self.archive, batch, add_info),
        )

        self._itrs += 1
        new_sols = int(validated_add_info["status"].astype(bool).sum())

        indices, ranking_values = self._ranker.rank(
            self, self.archive, data, validated_add_info
        )

        num_parents = (
            new_sols if self._selection_rule == "filter" else self._batch_size // 2
        )

        if num_parents > 0:
            parent_indices = indices[:num_parents]
            parents = np.asarray(data["solution"][parent_indices], dtype=np.float64)
            self._update_mean(parents)
        self._adapt_sigma(num_parents=num_parents)

        if self._check_restart(new_sols):
            if self.archive.stats.num_elites > 0:
                new_x0 = self.archive.sample_elites(1)["solution"][0]
                self._mean = _quantize_rule_bits(new_x0)
            else:
                self._mean = self._x0.copy()
            self._ranker.reset(self, self.archive)
            self._restarts += 1

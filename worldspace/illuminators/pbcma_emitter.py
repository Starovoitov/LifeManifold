"""pbCMA-style mixed discrete/continuous CMA-ME emitter for pyribs.

Maintains a latent Gaussian N(m, σ²C). Ask thresholds rule bits at 0.5 so the
archive stores {0,1} genotypes (bit-identical to genetic ME), while tell updates
CMA state from the *latent* samples of ranked parents. Margin correction keeps
bit means away from {0,1} freeze. This is the journal-extension control that
tests whether covariance adaptation — not continuous archive encoding — closes
the native bit-flip → rint gap.
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
    clip_genome_float_params,
)

__all__ = ["PBCMAEmitter", "pbcma_x0"]

_FLOAT_GENE_START = 18
_MARGIN = 0.05  # keep P(bit=1) away from {0,1}
_EIG_EPS = 1e-12
_SIGMA_MIN = 1e-4
_SIGMA_MAX = 2.0


def pbcma_x0() -> np.ndarray:
    """Latent mean: mid rule bits (0.5) + mid float params."""
    x0 = np.empty(GENOME_SIZE, dtype=np.float64)
    x0[:_FLOAT_GENE_START] = 0.5
    for index, (lo, hi) in enumerate(FLOAT_PARAM_BOUNDS, start=_FLOAT_GENE_START):
        x0[index] = 0.5 * (lo + hi)
    return x0


def _project_solution(latent: np.ndarray) -> np.ndarray:
    """Threshold bits; clip floats. Archive-facing discrete genotype."""
    out = np.asarray(latent, dtype=np.float64).copy()
    out[:_FLOAT_GENE_START] = (out[:_FLOAT_GENE_START] >= 0.5).astype(np.float64)
    return clip_genome_float_params(out, start_index=_FLOAT_GENE_START)


def _apply_margin(mean: np.ndarray) -> np.ndarray:
    """Push bit means into [margin, 1-margin] (CMA-ES-with-margin style)."""
    out = np.asarray(mean, dtype=np.float64).copy()
    out[:_FLOAT_GENE_START] = np.clip(out[:_FLOAT_GENE_START], _MARGIN, 1.0 - _MARGIN)
    return clip_genome_float_params(out, start_index=_FLOAT_GENE_START)


def _cma_weights(mu: int) -> np.ndarray:
    raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=np.float64))
    raw = np.maximum(raw, 0.0)
    if raw.sum() <= 0:
        raw = np.ones(mu, dtype=np.float64)
    return raw / raw.sum()


class PBCMAEmitter(EmitterBase):
    """Latent-Gaussian CMA with binary projection and margin correction."""

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

        dim = int(archive.solution_dim)
        self._dim = dim
        self._x0 = _apply_margin(np.asarray(x0, dtype=np.float64))
        check_shape(self._x0, "x0", dim, "archive.solution_dim")
        self._mean = self._x0.copy()
        self._sigma = float(np.clip(sigma0, _SIGMA_MIN, _SIGMA_MAX))
        self._C = np.eye(dim, dtype=np.float64)
        self._pc = np.zeros(dim, dtype=np.float64)
        self._ps = np.zeros(dim, dtype=np.float64)
        self._last_latents: np.ndarray | None = None

        if batch_size is None or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._batch_size = int(batch_size)
        self._mu = max(self._batch_size // 2, 1)
        self._weights = _cma_weights(self._mu)
        self._mu_eff = float(1.0 / np.sum(self._weights**2))

        # Standard CMA-ES learning rates (Hansen tutorial).
        n = float(dim)
        self._c_sigma = (self._mu_eff + 2.0) / (n + self._mu_eff + 5.0)
        self._d_sigma = (
            1.0
            + 2.0 * max(0.0, np.sqrt((self._mu_eff - 1.0) / (n + 1.0)) - 1.0)
            + self._c_sigma
        )
        self._c_c = (4.0 + self._mu_eff / n) / (n + 4.0 + 2.0 * self._mu_eff / n)
        self._c_1 = 2.0 / ((n + 1.3) ** 2 + self._mu_eff)
        self._c_mu = min(
            1.0 - self._c_1,
            2.0
            * (self._mu_eff - 2.0 + 1.0 / self._mu_eff)
            / ((n + 2.0) ** 2 + self._mu_eff),
        )
        self._chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))

        if selection_rule not in ["mu", "filter"]:
            raise ValueError(f"Invalid selection_rule {selection_rule}")
        self._selection_rule = selection_rule
        self._restart_rule = restart_rule
        self._restarts = 0
        self._itrs = 0
        _ = self._check_restart(0)

        self._ranker = _get_ranker(ranker, cast(Int | None, ranker_seed))
        self._ranker.reset(self, archive)
        self._refresh_eigensystem()

    def _refresh_eigensystem(self) -> None:
        self._C = 0.5 * (self._C + self._C.T)
        eigvals, eigvecs = np.linalg.eigh(self._C)
        eigvals = np.maximum(eigvals, _EIG_EPS)
        self._eigvals = eigvals
        self._eigvecs = eigvecs
        self._B = eigvecs
        self._D = np.sqrt(eigvals)
        self._invsqrtC = eigvecs @ np.diag(1.0 / self._D) @ eigvecs.T

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

    @property
    def sigma(self) -> float:
        return self._sigma

    def ask(self) -> np.ndarray:
        z = self._rng.standard_normal((self._batch_size, self._dim))
        # y = B D z; latent = m + σ y
        y = (self._B * self._D) @ z.T
        y = y.T
        latents = self._mean + self._sigma * y
        self._last_latents = latents
        solutions = np.empty_like(latents)
        for index in range(self._batch_size):
            solutions[index] = _project_solution(latents[index])
        return solutions.astype(self.archive.dtypes["solution"], copy=False)

    def _check_restart(self, num_parents: int) -> bool:
        if isinstance(self._restart_rule, numbers.Integral):
            return self._itrs > 0 and self._itrs % self._restart_rule == 0
        if self._restart_rule == "no_improvement":
            return num_parents == 0
        if self._restart_rule == "basic":
            return False
        raise ValueError(f"Invalid restart_rule {self._restart_rule}")

    def _cma_update(self, parent_latents: np.ndarray) -> None:
        """(μ/μ_w, λ)-CMA update from ranked latent parents (best first)."""
        mu = min(self._mu, parent_latents.shape[0])
        if mu <= 0:
            return
        parents = parent_latents[:mu]
        weights = self._weights[:mu]
        weights = weights / weights.sum()
        mu_eff = float(1.0 / np.sum(weights**2))

        old_mean = self._mean.copy()
        new_mean = weights @ parents
        self._mean = _apply_margin(new_mean)

        # y-space steps of selected parents.
        y_w = (self._mean - old_mean) / max(self._sigma, _SIGMA_MIN)
        ys = (parents - old_mean) / max(self._sigma, _SIGMA_MIN)

        # Cumulative step-size adaptation (CSA).
        self._ps = (1.0 - self._c_sigma) * self._ps + np.sqrt(
            self._c_sigma * (2.0 - self._c_sigma) * mu_eff
        ) * (self._invsqrtC @ y_w)
        ps_norm = float(np.linalg.norm(self._ps))
        self._sigma = float(
            np.clip(
                self._sigma
                * np.exp(
                    (self._c_sigma / self._d_sigma) * (ps_norm / self._chi_n - 1.0)
                ),
                _SIGMA_MIN,
                _SIGMA_MAX,
            )
        )

        # Rank-one / rank-μ covariance update.
        hsig = float(
            ps_norm
            / np.sqrt(1.0 - (1.0 - self._c_sigma) ** (2 * (self._itrs + 1)))
            / self._chi_n
            < 1.4 + 2.0 / (self._dim + 1.0)
        )
        self._pc = (1.0 - self._c_c) * self._pc + hsig * np.sqrt(
            self._c_c * (2.0 - self._c_c) * mu_eff
        ) * y_w

        artmp = ys.T * np.sqrt(weights)  # (dim, mu)
        self._C = (
            (1.0 - self._c_1 - self._c_mu) * self._C
            + self._c_1
            * (
                np.outer(self._pc, self._pc)
                + (1.0 - hsig) * self._c_c * (2.0 - self._c_c) * self._C
            )
            + self._c_mu * (artmp @ artmp.T)
        )
        self._refresh_eigensystem()

    def _restart(self) -> None:
        if self.archive.stats.num_elites > 0:
            elite = np.asarray(
                self.archive.sample_elites(1)["solution"][0], dtype=np.float64
            )
            # Soften discrete elite into latent mean (bits stay near 0/1 ± margin).
            soft = elite.copy()
            soft[:_FLOAT_GENE_START] = np.where(
                elite[:_FLOAT_GENE_START] >= 0.5,
                1.0 - _MARGIN,
                _MARGIN,
            )
            self._mean = _apply_margin(soft)
        else:
            self._mean = self._x0.copy()
        self._sigma = float(np.clip(self._sigma, _SIGMA_MIN, _SIGMA_MAX))
        self._C = np.eye(self._dim, dtype=np.float64)
        self._pc = np.zeros(self._dim, dtype=np.float64)
        self._ps = np.zeros(self._dim, dtype=np.float64)
        self._refresh_eigensystem()
        self._ranker.reset(self, self.archive)
        self._restarts += 1

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

        indices, _ranking_values = self._ranker.rank(
            self, self.archive, data, validated_add_info
        )

        num_parents = (
            new_sols if self._selection_rule == "filter" else self._batch_size // 2
        )

        if num_parents > 0 and self._last_latents is not None:
            parent_indices = indices[:num_parents]
            parent_latents = np.asarray(
                self._last_latents[parent_indices], dtype=np.float64
            )
            self._cma_update(parent_latents)

        if self._check_restart(new_sols):
            self._restart()

        self._last_latents = None

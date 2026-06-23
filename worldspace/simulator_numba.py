"""Fused numba CA inner loop for ``run_world`` (optional, behind ``numba_simulator``)."""

from __future__ import annotations

from collections import deque

import numpy as np
from numba import njit

from worldspace import math as ws_math

__all__ = [
    "density_tail_from_buffer",
    "random_draws_per_step",
    "run_ca_loop_numba",
]


def random_draws_per_step(
    grid_size: int,
    *,
    noise: float,
    predation: float,
) -> int:
    """Random doubles consumed per CA step (noise, predation, food), matching numpy order."""
    n2 = grid_size * grid_size
    draws = n2
    if noise > 0:
        draws += n2
    if predation > 0:
        draws += n2
    return draws


def density_tail_from_buffer(
    buf: np.ndarray,
    tail_len: int,
    tail_start: int,
) -> deque[float]:
    """Rebuild chronological density tail from the numba circular buffer."""
    maxlen = ws_math.OSCILLATION_DENSITY_WINDOW
    if tail_len <= 0:
        return deque(maxlen=maxlen)
    if tail_len < maxlen:
        values = [float(buf[i]) for i in range(tail_len)]
    else:
        values = [float(buf[i]) for i in range(tail_start, maxlen)]
        values.extend(float(buf[i]) for i in range(tail_start))
    return deque(values, maxlen=maxlen)


def run_ca_loop_numba(
    life: np.ndarray,
    food: np.ndarray,
    ages: np.ndarray,
    birth_mask: np.ndarray,
    survival_mask: np.ndarray,
    *,
    noise: float,
    predation: float,
    resource_regen: float,
    steps: int,
    early_extinction_step: int | None,
    random_buf: np.ndarray,
    use_cache: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    int,
    int,
    int,
    np.ndarray,
    int,
    int,
    float,
    int,
    bool,
]:
    """Run the fused CA loop; ``random_buf`` must follow the numpy RNG stream after init."""
    n = life.shape[0]
    rnd_per_step = random_draws_per_step(n, noise=noise, predation=predation)
    tail_buf = np.zeros(ws_math.OSCILLATION_DENSITY_WINDOW, dtype=np.float64)
    early_step = -1 if early_extinction_step is None else early_extinction_step
    kernel = _run_ca_loop_numba_cached if use_cache else _run_ca_loop_numba_nocache
    (
        life,
        food,
        ages,
        density_mean,
        density_m2,
        density_n,
        death_age_sum,
        death_count,
        tail_buf,
        tail_len,
        tail_start,
        activity_sum,
        activity_steps,
        early_extinct,
    ) = kernel(
        life,
        food,
        ages,
        birth_mask,
        survival_mask,
        noise,
        predation,
        resource_regen,
        steps,
        early_step,
        random_buf,
        rnd_per_step,
        noise > 0,
        predation > 0,
        tail_buf,
    )
    return (
        life,
        food,
        ages,
        density_mean,
        density_m2,
        density_n,
        death_age_sum,
        death_count,
        tail_buf,
        tail_len,
        tail_start,
        activity_sum,
        activity_steps,
        early_extinct,
    )


@njit(cache=True)
def _run_ca_loop_numba_cached(
    life,
    food,
    ages,
    birth_mask,
    survival_mask,
    noise,
    predation,
    resource_regen,
    steps,
    early_extinction_step,
    random_buf,
    rnd_per_step,
    has_noise,
    has_predation,
    tail_buf,
):
    return _run_ca_loop_numba_impl(
        life,
        food,
        ages,
        birth_mask,
        survival_mask,
        noise,
        predation,
        resource_regen,
        steps,
        early_extinction_step,
        random_buf,
        rnd_per_step,
        has_noise,
        has_predation,
        tail_buf,
    )


@njit(cache=False)
def _run_ca_loop_numba_nocache(
    life,
    food,
    ages,
    birth_mask,
    survival_mask,
    noise,
    predation,
    resource_regen,
    steps,
    early_extinction_step,
    random_buf,
    rnd_per_step,
    has_noise,
    has_predation,
    tail_buf,
):
    return _run_ca_loop_numba_impl(
        life,
        food,
        ages,
        birth_mask,
        survival_mask,
        noise,
        predation,
        resource_regen,
        steps,
        early_extinction_step,
        random_buf,
        rnd_per_step,
        has_noise,
        has_predation,
        tail_buf,
    )


@njit(cache=True)
def _run_ca_loop_numba_impl(
    life,
    food,
    ages,
    birth_mask,
    survival_mask,
    noise,
    predation,
    resource_regen,
    steps,
    early_extinction_step,
    random_buf,
    rnd_per_step,
    has_noise,
    has_predation,
    tail_buf,
):
    n = life.shape[0]
    density_mean = 0.0
    density_m2 = 0.0
    density_n = 0
    death_age_sum = 0
    death_count = 0
    tail_len = 0
    tail_start = 0
    tail_cap = tail_buf.shape[0]
    early_extinct = False
    activity_sum = 0.0
    activity_steps = 0

    for step in range(steps):
        next_life = np.empty((n, n), dtype=np.uint8)
        rnd_base = step * rnd_per_step
        noise_off = 0
        pred_off = n * n if has_noise else 0
        food_off = n * n * (int(has_noise) + int(has_predation))

        for i in range(n):
            for j in range(n):
                neighbors = 0
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0:
                            continue
                        ni = (i + di) % n
                        nj = (j + dj) % n
                        neighbors += life[ni, nj]

                born = life[i, j] == 0 and birth_mask[neighbors]
                survive = life[i, j] == 1 and survival_mask[neighbors]
                cell = 1 if (born or survive) else 0

                if has_noise:
                    idx = rnd_base + noise_off + i * n + j
                    if random_buf[idx] < noise:
                        cell = 1 - cell

                if has_predation:
                    idx = rnd_base + pred_off + i * n + j
                    exposure = neighbors / 8.0
                    if random_buf[idx] < predation * exposure * cell:
                        cell = 0

                next_life[i, j] = cell

        flip_count = 0
        for i in range(n):
            for j in range(n):
                if life[i, j] != next_life[i, j]:
                    flip_count += 1
        activity_sum += flip_count / (n * n)
        activity_steps += 1

        for i in range(n):
            for j in range(n):
                idx = rnd_base + food_off + i * n + j
                if random_buf[idx] < resource_regen:
                    food[i, j] = 1

                feed_bonus = 0
                if food[i, j] == 1 and next_life[i, j] == 1:
                    feed_bonus = 1
                    food[i, j] = 0

                if life[i, j] == 1 and next_life[i, j] == 0:
                    death_age_sum += ages[i, j]
                    death_count += 1

                if next_life[i, j] == 1:
                    ages[i, j] = ages[i, j] + 1 + feed_bonus
                else:
                    ages[i, j] = 0

        life = next_life

        density = life.sum() / (n * n)

        density_n += 1
        delta = density - density_mean
        density_mean += delta / density_n
        density_m2 += delta * (density - density_mean)

        if tail_len < tail_cap:
            tail_buf[tail_len] = density
            tail_len += 1
        else:
            tail_buf[tail_start] = density
            tail_start = (tail_start + 1) % tail_cap

        t = step + 1
        if early_extinction_step >= 0 and t < early_extinction_step and density == 0.0:
            early_extinct = True
            break

    return (
        life,
        food,
        ages,
        density_mean,
        density_m2,
        density_n,
        death_age_sum,
        death_count,
        tail_buf,
        tail_len,
        tail_start,
        activity_sum,
        activity_steps,
        early_extinct,
    )

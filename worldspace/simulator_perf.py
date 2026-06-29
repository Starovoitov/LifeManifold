"""Simulator performance options (numba / parallel eval); wired from scheduler YAML."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

__all__ = [
    "DEFAULT_SIMULATOR_PERFORMANCE",
    "METRICS_VERIFY_ATOL",
    "SimulatorPerformanceOptions",
    "effective_numba_enabled",
    "effective_llm_parallel_workers",
    "effective_parallel_workers",
    "resolve_simulator_performance",
    "validate_simulator_performance",
]


METRICS_VERIFY_ATOL = 1e-12


@dataclass(frozen=True)
class SimulatorPerformanceOptions:
    """Optional fast paths for ``run_world`` (numba, parallel eval). Defaults use numpy.

    ``parallel_workers``: ``0`` selects ``os.cpu_count()`` (auto), capped by batch size
    when ``parallel_eval`` is enabled. Set an explicit positive integer to cap workers.
    """

    numba_simulator: bool = False
    numba_cache: bool = True
    parallel_eval: bool = False
    parallel_workers: int = 0  # 0 = auto (os.cpu_count())
    llm_parallel_emit: bool = False
    log_iteration_timing: bool = False
    verify_against_reference: bool = False


DEFAULT_SIMULATOR_PERFORMANCE = SimulatorPerformanceOptions()


def effective_numba_enabled(
    performance: SimulatorPerformanceOptions,
    *,
    ca_step_trace: bool,
) -> bool:
    """Return whether the numba fused step may run (trace path always uses numpy)."""
    if ca_step_trace:
        return False
    return performance.numba_simulator


def effective_llm_parallel_workers(
    performance: SimulatorPerformanceOptions,
    *,
    llm_slot_count: int,
) -> int:
    """Return thread count for parallel LLM HTTP (0 = sequential emit).

    When ``llm_parallel_emit`` is enabled, uses one worker per LLM slot in the
    current batch (``llm_slot_count``).
    """
    if not performance.llm_parallel_emit or llm_slot_count <= 1:
        return 0
    return llm_slot_count


def effective_parallel_workers(
    performance: SimulatorPerformanceOptions,
    *,
    batch_size: int,
) -> int:
    """Return worker count for parallel eval (at least 1, capped by ``batch_size``).

    When ``parallel_workers`` is ``0``, uses ``os.cpu_count()`` (auto).
    """
    if performance.parallel_workers > 0:
        requested = performance.parallel_workers
    else:
        requested = os.cpu_count() or 1
    return max(1, min(int(requested), int(batch_size)))


def resolve_simulator_performance(
    yaml_block: dict[str, Any] | None = None,
) -> SimulatorPerformanceOptions:
    """Build options from scheduler YAML block with env overrides (env wins)."""
    block = yaml_block or {}
    options = SimulatorPerformanceOptions(
        numba_simulator=bool(block.get("numba_simulator", False)),
        numba_cache=bool(block.get("numba_cache", True)),
        parallel_eval=bool(block.get("parallel_eval", False)),
        parallel_workers=int(block.get("parallel_workers", 0)),
        llm_parallel_emit=bool(block.get("llm_parallel_emit", False)),
        log_iteration_timing=bool(block.get("log_iteration_timing", False)),
        verify_against_reference=bool(block.get("verify_against_reference", False)),
    )
    return _apply_env_overrides(options)


def validate_simulator_performance(performance: SimulatorPerformanceOptions) -> None:
    """Reject incompatible optional fast paths before runtime."""
    if performance.numba_simulator and performance.parallel_eval:
        raise ValueError(
            "numba_simulator and parallel_eval cannot both be enabled: "
            "forkserver after numba JIT can deadlock LLVM worker threads. "
            "Use parallel_eval with numba_simulator=false, or disable parallel_eval."
        )


def _apply_env_overrides(
    options: SimulatorPerformanceOptions,
) -> SimulatorPerformanceOptions:
    env_numba = _env_bool("LIFEMANIFOLD_NUMBA_SIM")
    if env_numba is not None:
        options = replace(options, numba_simulator=env_numba)
    env_parallel = _env_bool("LIFEMANIFOLD_PARALLEL_EVAL")
    if env_parallel is not None:
        options = replace(options, parallel_eval=env_parallel)
    env_llm_parallel = _env_bool("LIFEMANIFOLD_LLM_PARALLEL_EMIT")
    if env_llm_parallel is not None:
        options = replace(options, llm_parallel_emit=env_llm_parallel)
    env_log_timing = _env_bool("LIFEMANIFOLD_LOG_ITERATION_TIMING")
    if env_log_timing is not None:
        options = replace(options, log_iteration_timing=env_log_timing)
    env_verify = _env_bool("LIFEMANIFOLD_VERIFY_SIM")
    if env_verify is not None:
        options = replace(options, verify_against_reference=env_verify)
    return options


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() not in ("0", "false", "no", "off", "")

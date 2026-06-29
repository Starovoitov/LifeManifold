"""Parallel LLM HTTP for MAP-Elites emit (thread pool, batch requests)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from worldspace.illuminators.emitters.llm_emitter import LlmPreparedSlot
from worldspace.simulator_perf import (
    SimulatorPerformanceOptions,
    effective_llm_parallel_workers,
)

if TYPE_CHECKING:
    from worldspace.illuminators.emitters.base import CandidateEmitter
    from worldspace.illuminators.emitters.llm_emitter import LlmEmitter
    from worldspace.illuminators.scheduler import SchedulerConfig

__all__ = [
    "ParallelLlmPool",
    "llm_slot_count_for_batch",
    "parallel_llm_context",
    "request_llm_batch",
    "supports_parallel_llm_emit",
]


class ParallelLlmPool:
    """Reusable thread pool for illuminator parallel LLM HTTP."""

    def __init__(self, max_workers: int) -> None:
        if max_workers < 1:
            msg = f"max_workers must be >= 1, got {max_workers}"
            raise ValueError(msg)
        self._max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = None

    def map_requests(
        self,
        request_one: Callable[[LlmPreparedSlot], str],
        prepared: Sequence[LlmPreparedSlot],
        *,
        max_workers: int,
    ) -> list[str]:
        """Run ``request_one`` for each slot; preserve input order."""
        if not prepared:
            return []
        workers = min(max_workers, len(prepared), self._max_workers)
        if workers <= 1:
            return [request_one(slot) for slot in prepared]
        executor = self._executor
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=self._max_workers)
            self._executor = executor
        return list(executor.map(request_one, prepared))

    def shutdown(self, *, wait: bool = True) -> None:
        """Release thread pool resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None


def parallel_llm_context(
    performance: SimulatorPerformanceOptions,
    *,
    max_llm_slots: int,
) -> ParallelLlmPool | None:
    """Create a persistent thread pool when parallel LLM emit is enabled."""
    if not performance.llm_parallel_emit or max_llm_slots <= 1:
        return None
    workers = effective_llm_parallel_workers(
        performance,
        llm_slot_count=max_llm_slots,
    )
    if workers <= 1:
        return None
    return ParallelLlmPool(workers)


def llm_slot_count_for_batch(
    config: SchedulerConfig,
    *,
    candidates_evaluated: int,
) -> int:
    """Count LLM emitter slots in the current batch (after initial-fill overrides)."""
    from worldspace.illuminators.scheduler import resolve_emitter_for_slot

    return sum(
        1
        for candidate_id in range(config.batch_size)
        if resolve_emitter_for_slot(
            config,
            candidate_id=candidate_id,
            candidates_evaluated=candidates_evaluated,
        )
        == "llm"
    )


def supports_parallel_llm_emit(
    emitter: CandidateEmitter,
    config: SchedulerConfig,
) -> bool:
    """True when the emitter can use the parallel LLM HTTP path."""
    from worldspace.illuminators.emitters.base import MapElitesEmitter

    return isinstance(emitter, MapElitesEmitter) and config.llm_enabled


def request_llm_batch(
    llm: LlmEmitter,
    prepared: Sequence[LlmPreparedSlot],
    *,
    max_workers: int,
    llm_pool: ParallelLlmPool | None = None,
) -> list[str]:
    """POST chat completions for each prepared slot; empty string on failure."""
    if not prepared:
        return []

    def request_one(slot: LlmPreparedSlot) -> str:
        try:
            return llm.request_llm(slot)
        except (RuntimeError, ValueError):
            return ""

    if llm_pool is not None:
        return llm_pool.map_requests(
            request_one,
            prepared,
            max_workers=max_workers,
        )
    return _parallel_llm_responses(
        prepared,
        request_one=request_one,
        max_workers=max_workers,
    )


def _parallel_llm_responses(
    prepared: Sequence[LlmPreparedSlot],
    *,
    request_one: Callable[[LlmPreparedSlot], str],
    max_workers: int,
) -> list[str]:
    """Ephemeral thread pool when no ``ParallelLlmPool`` is available."""
    if max_workers <= 1 or len(prepared) <= 1:
        return [request_one(slot) for slot in prepared]
    workers = min(max_workers, len(prepared))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(request_one, prepared))

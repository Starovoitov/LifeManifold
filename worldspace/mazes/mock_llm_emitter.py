"""Deterministic LLM emitter stub for smoke tests and CI factorial runs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from worldspace.mazes.archive import MazeArchive
from worldspace.mazes.emitters import MazeEmitterResult, MazeTarget, emit_genetic
from worldspace.mazes.genetics import mutate_maze
from worldspace.mazes.llm_emitter import tile_distance
from worldspace.mazes.surrogate import MazePrediction


@dataclass
class MockMazeLlmAudit:
    attempts: int = 0
    api_calls: int = 0
    retries: int = 0
    parse_successes: int = 0
    fallbacks: int = 0
    repaired_outputs: int = 0
    repair_collapses: int = 0
    zero_distance: int = 0
    total_tile_distance: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)
    invalid_response_reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "attempts": self.attempts,
            "api_calls": self.api_calls,
            "retries": self.retries,
            "parse_successes": self.parse_successes,
            "parse_success_rate": (
                self.parse_successes / self.attempts if self.attempts else 0.0
            ),
            "fallbacks": self.fallbacks,
            "fallback_rate": self.fallbacks / self.attempts if self.attempts else 0.0,
            "zero_distance": self.zero_distance,
            "repaired_outputs": self.repaired_outputs,
            "repair_rate": (
                self.repaired_outputs / self.parse_successes
                if self.parse_successes
                else 0.0
            ),
            "repair_collapses": self.repair_collapses,
            "repair_collapse_rate": (
                self.repair_collapses / self.attempts if self.attempts else 0.0
            ),
            "failure_reasons": dict(sorted(self.failure_reasons.items())),
            "invalid_response_reasons": dict(
                sorted(self.invalid_response_reasons.items())
            ),
            "mean_tile_distance": (
                self.total_tile_distance / self.parse_successes
                if self.parse_successes
                else 0.0
            ),
        }


class MockMazeLlmEmitter:
    """Emit valid maze mutations without calling a remote LLM."""

    prompt_version = "mock-v1"

    def __init__(self) -> None:
        self.audit = MockMazeLlmAudit()

    def emit(
        self,
        *,
        target: MazeTarget,
        archive: MazeArchive,
        rng: np.random.Generator,
        prediction: MazePrediction | None,
    ) -> MazeEmitterResult:
        del prediction
        self.audit.attempts += 1
        if target.parent is None:
            fallback = emit_genetic(target, archive, rng)
            self.audit.fallbacks += 1
            return MazeEmitterResult(
                spec=fallback.spec,
                parent_id=fallback.parent_id,
                emitter_type="llm_fallback_genetic",
            )
        child = mutate_maze(target.parent.spec, rng)
        distance = tile_distance(target.parent.spec, child)
        self.audit.parse_successes += 1
        self.audit.total_tile_distance += distance
        self.audit.zero_distance += distance == 0
        return MazeEmitterResult(
            spec=child,
            parent_id=target.parent.candidate_id,
            emitter_type="llm",
        )

    def emit_batch(
        self,
        jobs: list[
            tuple[
                MazeTarget,
                MazeArchive,
                np.random.Generator,
                MazePrediction | None,
            ]
        ],
        *,
        max_workers: int = 4,
    ) -> list[MazeEmitterResult]:
        del max_workers
        return [
            self.emit(
                target=job[0],
                archive=job[1],
                rng=job[2],
                prediction=job[3],
            )
            for job in jobs
        ]

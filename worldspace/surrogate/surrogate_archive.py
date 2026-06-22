"""Append-only SurrogateArchive JSONL writer (schema 1.0)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from worldspace.illuminators.archive import InsertResult
from worldspace.illuminators.evaluation import EvalResult
from worldspace.illuminators.scheduler import TargetBin
from worldspace.surrogate.acquisition import AcquisitionDecision
from worldspace.surrogate.acquisition_config import AcquisitionMode
from worldspace.surrogate.types import SurrogatePrediction

SURROGATE_ARCHIVE_SCHEMA_VERSION = "1.0"
DEFAULT_FLUSH_EVERY = 32

__all__ = [
    "DEFAULT_FLUSH_EVERY",
    "SURROGATE_ARCHIVE_SCHEMA_VERSION",
    "NoOpSurrogateArchiveWriter",
    "SurrogateArchiveWriter",
    "open_surrogate_archive",
    "serialize_eval_outcome",
    "serialize_prediction",
    "surrogate_archive_path_for_output",
]


class SurrogateArchiveWriterProtocol(Protocol):
    """Minimal writer surface used by the illuminator loop."""

    def append_slot(
        self,
        *,
        iteration: int,
        candidate_id: int,
        emitter_type: str,
        target: TargetBin,
        target_cell_id: int,
        world_spec_hash: str,
        prediction: SurrogatePrediction,
        decision: AcquisitionDecision,
        acquisition_mode: AcquisitionMode,
        eval_result: EvalResult | None = None,
        insert: InsertResult | None = None,
    ) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


def surrogate_archive_path_for_output(output_dir: Path | str) -> Path:
    """Return per-run SurrogateArchive path under an illuminator output directory."""
    return Path(output_dir).expanduser() / "surrogate_archive.jsonl"


def open_surrogate_archive(
    path: Path | str,
    *,
    run_id: str,
    enabled: bool,
    flush_every: int = DEFAULT_FLUSH_EVERY,
) -> SurrogateArchiveWriterProtocol:
    """Open a batched writer or a no-op implementation when logging is disabled."""
    if not enabled:
        return NoOpSurrogateArchiveWriter()
    return SurrogateArchiveWriter(
        path=path,
        run_id=run_id,
        flush_every=flush_every,
    )


def serialize_prediction(prediction: SurrogatePrediction) -> dict[str, Any]:
    """Serialize ``SurrogatePrediction`` for JSONL storage."""
    return {
        "components": {k: float(v) for k, v in prediction.components.items()},
        "measures": {k: float(v) for k, v in prediction.measures.items()},
        "fitness": float(prediction.fitness),
        "uncertainty": float(prediction.uncertainty),
    }


def serialize_eval_outcome(
    eval_result: EvalResult,
    insert: InsertResult,
) -> dict[str, Any]:
    """Serialize real evaluation outcome for one accepted/rejected slot."""
    return {
        "fitness": float(eval_result.fitness),
        "measures": {k: float(v) for k, v in eval_result.measures.items()},
        "accepted": bool(insert.accepted),
        "improved": bool(insert.improved),
    }


@dataclass
class SurrogateArchiveWriter:
    """Batched append-only JSONL writer for acquisition audit records."""

    path: Path | str
    run_id: str
    flush_every: int = DEFAULT_FLUSH_EVERY
    _pending: list[dict[str, Any]] = field(default_factory=list)

    def append_slot(
        self,
        *,
        iteration: int,
        candidate_id: int,
        emitter_type: str,
        target: TargetBin,
        target_cell_id: int,
        world_spec_hash: str,
        prediction: SurrogatePrediction,
        decision: AcquisitionDecision,
        acquisition_mode: AcquisitionMode,
        eval_result: EvalResult | None = None,
        insert: InsertResult | None = None,
    ) -> None:
        """Queue one slot record and flush in batches."""
        record = build_archive_record(
            run_id=self.run_id,
            iteration=iteration,
            candidate_id=candidate_id,
            emitter_type=emitter_type,
            target=target,
            target_cell_id=target_cell_id,
            world_spec_hash=world_spec_hash,
            prediction=prediction,
            decision=decision,
            acquisition_mode=acquisition_mode,
            eval_result=eval_result,
            insert=insert,
        )
        self._pending.append(record)
        if len(self._pending) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Persist all pending records."""
        if not self._pending:
            return
        target = Path(self.path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            for row in self._pending:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        self._pending.clear()

    def close(self) -> None:
        """Flush pending rows on run shutdown."""
        self.flush()


@dataclass(frozen=True)
class NoOpSurrogateArchiveWriter:
    """Writer that discards records (acquisition off or surrogate disabled)."""

    def append_slot(
        self,
        *,
        iteration: int,
        candidate_id: int,
        emitter_type: str,
        target: TargetBin,
        target_cell_id: int,
        world_spec_hash: str,
        prediction: SurrogatePrediction,
        decision: AcquisitionDecision,
        acquisition_mode: AcquisitionMode,
        eval_result: EvalResult | None = None,
        insert: InsertResult | None = None,
    ) -> None:
        del (
            iteration,
            candidate_id,
            emitter_type,
            target,
            target_cell_id,
            world_spec_hash,
            prediction,
            decision,
            acquisition_mode,
            eval_result,
            insert,
        )

    def flush(self) -> None:
        return

    def close(self) -> None:
        return


def build_archive_record(
    *,
    run_id: str,
    iteration: int,
    candidate_id: int,
    emitter_type: str,
    target: TargetBin,
    target_cell_id: int,
    world_spec_hash: str,
    prediction: SurrogatePrediction,
    decision: AcquisitionDecision,
    acquisition_mode: AcquisitionMode,
    eval_result: EvalResult | None,
    insert: InsertResult | None,
) -> dict[str, Any]:
    """Build one schema 1.0 JSON object for SurrogateArchive."""
    outcome = None
    if eval_result is not None and insert is not None:
        outcome = serialize_eval_outcome(eval_result, insert)
    return {
        "schema_version": SURROGATE_ARCHIVE_SCHEMA_VERSION,
        "run_id": run_id,
        "iteration": int(iteration),
        "candidate_id": int(candidate_id),
        "emitter_type": emitter_type,
        "target_bin": [int(target.bin[0]), int(target.bin[1])],
        "target_cell_id": int(target_cell_id),
        "world_spec_hash": world_spec_hash,
        "prediction": serialize_prediction(prediction),
        "decision": decision.action,
        "decision_reason": decision.reason,
        "acquisition_mode": acquisition_mode,
        "eval_outcome": outcome,
    }

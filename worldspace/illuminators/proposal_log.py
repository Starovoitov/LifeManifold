"""Per-run evaluated-proposal JSONL (accepts + rejects).

``map_elites_archive.jsonl`` only records accepted inserts. This log records
every *evaluated* slot so all-evaluated LLM (and optional all-emitter) quality
and fail anatomy are available per seed.

Enabled under ``output_dir/proposal_log.jsonl`` unless
``LIFEMANIFOLD_PROPOSAL_LOG=0``. Override path with
``LIFEMANIFOLD_PROPOSAL_LOG=/path``. By default only ``emitter_type`` values
starting with ``llm`` are written; set ``LIFEMANIFOLD_PROPOSAL_LOG_ALL_EMITTERS=1``
to log random/genetic slots too.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from worldspace.illuminators.archive import InsertResult
from worldspace.illuminators.evaluation import EvalResult
from worldspace.illuminators.scheduler import TargetBin
from worldspace.specs.spec import WorldSpec
from worldspace.surrogate.canonical_hash import world_spec_canonical_hash
from worldspace.surrogate.types import SurrogatePrediction

PROPOSAL_LOG_SCHEMA_VERSION = "1.0"
DEFAULT_PROPOSAL_LOG_NAME = "proposal_log.jsonl"
DEFAULT_FLUSH_EVERY = 64

__all__ = [
    "DEFAULT_FLUSH_EVERY",
    "DEFAULT_PROPOSAL_LOG_NAME",
    "PROPOSAL_LOG_SCHEMA_VERSION",
    "NoOpProposalLogWriter",
    "ProposalLogWriter",
    "ProposalLogWriterProtocol",
    "configure_proposal_log",
    "insert_outcome_label",
    "open_proposal_log",
    "proposal_log_enabled_for_emitter",
    "proposal_log_path_for_output",
    "resolve_proposal_log_path",
    "serialize_proposal_record",
]

_configured_path: Path | None = None


def configure_proposal_log(path: str | Path | None) -> Path | None:
    """Set the process-wide proposal-log path (or disable with ``None``)."""
    global _configured_path
    if path is None:
        _configured_path = None
        return None
    dest = Path(path).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    _configured_path = dest
    return dest


def proposal_log_path_for_output(output_dir: Path | str) -> Path:
    """Return default per-run proposal log path."""
    return Path(output_dir).expanduser() / DEFAULT_PROPOSAL_LOG_NAME


def resolve_proposal_log_path(
    *,
    output_dir: str | Path | None = None,
) -> Path | None:
    """Resolve log path from env and optional ``output_dir`` default."""
    raw = os.environ.get("LIFEMANIFOLD_PROPOSAL_LOG")
    if raw is not None:
        value = raw.strip()
        if value in {"0", "false", "False", "off", "OFF"}:
            return None
        if value in {"1", "true", "True", "on", "ON"}:
            if _configured_path is not None:
                return _configured_path
            if output_dir is not None:
                return proposal_log_path_for_output(output_dir)
            return _configured_path
        return Path(value).expanduser()
    if _configured_path is not None:
        return _configured_path
    if output_dir is not None:
        return proposal_log_path_for_output(output_dir)
    return None


def proposal_log_enabled_for_emitter(emitter_type: str) -> bool:
    """Return whether this emitter type should be written to the proposal log."""
    raw = os.environ.get("LIFEMANIFOLD_PROPOSAL_LOG_ALL_EMITTERS", "").strip()
    if raw in {"1", "true", "True", "on", "ON"}:
        return True
    return str(emitter_type).startswith("llm")


def insert_outcome_label(insert: InsertResult) -> str:
    """Map InsertResult flags to a short outcome label."""
    if insert.improved:
        return "improve"
    if insert.accepted:
        return "fill_empty"
    return "occupied_not_better"


class ProposalLogWriterProtocol(Protocol):
    """Minimal writer surface used by the illuminator loop."""

    def append_evaluated(
        self,
        *,
        iteration: int,
        candidate_id: int,
        emitter_type: str,
        target: TargetBin,
        target_cell_id: int,
        eval_result: EvalResult,
        insert: InsertResult,
        parent_id: str | None = None,
        incumbent_fitness: float | None = None,
        prediction: SurrogatePrediction | None = None,
    ) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


def open_proposal_log(
    path: Path | str | None,
    *,
    run_id: str,
    enabled: bool,
    flush_every: int = DEFAULT_FLUSH_EVERY,
) -> ProposalLogWriterProtocol:
    """Open a batched writer or a no-op when logging is disabled."""
    if not enabled or path is None:
        return NoOpProposalLogWriter()
    return ProposalLogWriter(
        path=path,
        run_id=run_id,
        flush_every=flush_every,
    )


def serialize_proposal_record(
    *,
    run_id: str,
    iteration: int,
    candidate_id: int,
    emitter_type: str,
    target: TargetBin,
    target_cell_id: int,
    eval_result: EvalResult,
    insert: InsertResult,
    parent_id: str | None,
    incumbent_fitness: float | None,
    prediction: SurrogatePrediction | None,
) -> dict[str, Any]:
    """Build one schema 1.0 JSON object for the proposal log."""
    spec = eval_result.world_spec
    record: dict[str, Any] = {
        "schema_version": PROPOSAL_LOG_SCHEMA_VERSION,
        "run_id": run_id,
        "iteration": int(iteration),
        "candidate_id": int(candidate_id),
        "emitter_type": str(emitter_type),
        "parent_id": parent_id,
        "target_bin": [int(target.bin[0]), int(target.bin[1])],
        "target_cell_id": int(target_cell_id),
        "realized_bin": [int(eval_result.bin[0]), int(eval_result.bin[1])],
        "world_spec_hash": world_spec_canonical_hash(spec),
        "world_spec": _world_spec_dict(spec),
        "fitness": float(eval_result.fitness),
        "measures": {k: float(v) for k, v in eval_result.measures.items()},
        "early_extinct": bool(eval_result.early_extinct),
        "accepted": bool(insert.accepted),
        "improved": bool(insert.improved),
        "rejected": bool(insert.rejected),
        "outcome": insert_outcome_label(insert),
        "incumbent_fitness": (
            float(incumbent_fitness) if incumbent_fitness is not None else None
        ),
    }
    if prediction is not None:
        record["prediction"] = {
            "fitness": float(prediction.fitness),
            "uncertainty": float(prediction.uncertainty),
        }
    return record


def _world_spec_dict(spec: WorldSpec) -> dict[str, Any]:
    return spec.to_json_dict()


@dataclass
class ProposalLogWriter:
    """Batched append-only JSONL writer for evaluated proposals."""

    path: Path | str
    run_id: str
    flush_every: int = DEFAULT_FLUSH_EVERY
    _pending: list[dict[str, Any]] = field(default_factory=list)

    def append_evaluated(
        self,
        *,
        iteration: int,
        candidate_id: int,
        emitter_type: str,
        target: TargetBin,
        target_cell_id: int,
        eval_result: EvalResult,
        insert: InsertResult,
        parent_id: str | None = None,
        incumbent_fitness: float | None = None,
        prediction: SurrogatePrediction | None = None,
    ) -> None:
        """Queue one evaluated-slot record when the emitter filter allows it."""
        if not proposal_log_enabled_for_emitter(emitter_type):
            return
        self._pending.append(
            serialize_proposal_record(
                run_id=self.run_id,
                iteration=iteration,
                candidate_id=candidate_id,
                emitter_type=emitter_type,
                target=target,
                target_cell_id=target_cell_id,
                eval_result=eval_result,
                insert=insert,
                parent_id=parent_id,
                incumbent_fitness=incumbent_fitness,
                prediction=prediction,
            )
        )
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
class NoOpProposalLogWriter:
    """Writer that discards records when proposal logging is disabled."""

    def append_evaluated(
        self,
        *,
        iteration: int,
        candidate_id: int,
        emitter_type: str,
        target: TargetBin,
        target_cell_id: int,
        eval_result: EvalResult,
        insert: InsertResult,
        parent_id: str | None = None,
        incumbent_fitness: float | None = None,
        prediction: SurrogatePrediction | None = None,
    ) -> None:
        del (
            iteration,
            candidate_id,
            emitter_type,
            target,
            target_cell_id,
            eval_result,
            insert,
            parent_id,
            incumbent_fitness,
            prediction,
        )

    def flush(self) -> None:
        return

    def close(self) -> None:
        return

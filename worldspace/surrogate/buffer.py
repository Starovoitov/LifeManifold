"""Append-only JSONL training buffer with batched flush."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from worldspace.illuminators.evaluation import extinction_probability
from worldspace.specs.spec import WorldSpec

if TYPE_CHECKING:
    from worldspace.illuminators.evaluation import EvalResult
from worldspace.surrogate.feature_extractor import FEATURE_SCHEMA_VERSION, extract
from worldspace.surrogate.model import FITNESS_TARGET_KEY, TARGET_KEYS

__all__ = [
    "SurrogateBuffer",
    "append_eval_to_buffer",
    "buffer_record",
    "count_buffer_rows",
    "targets_from_eval_result",
    "world_spec_dict_for_buffer",
]


def count_buffer_rows(path: Path | str) -> int:
    """Count non-empty lines in an append-only surrogate buffer JSONL file.

    Returns 0 when the file is missing or cannot be read (permissions, encoding, I/O).
    """
    target = Path(path).expanduser()
    if not target.is_file():
        return 0
    try:
        count = 0
        with target.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    count += 1
        return count
    except (OSError, UnicodeDecodeError):
        return 0


def targets_from_eval_result(result: EvalResult) -> dict[str, float]:
    """Build Strategy A training targets from one real illuminator evaluation."""
    metrics = result.metrics
    final_density = float(metrics.density_mean)
    return {
        "stability": float(result.measures["stability"]),
        "diversity": float(result.measures["diversity"]),
        "oscillation_score": float(metrics.oscillation_score),
        "topology_interface_index": float(metrics.topology_interface_index),
        "topology_window_heterogeneity": float(metrics.topology_window_heterogeneity),
        "final_density": final_density,
        "early_extinction_prob": extinction_probability(final_density),
        FITNESS_TARGET_KEY: float(result.fitness),
    }


def world_spec_dict_for_buffer(spec: WorldSpec) -> dict[str, Any]:
    """Return a canonicalized ``WorldSpec`` dict for buffer JSONL rows."""
    from worldspace.illuminators.evaluation import apply_canonical_seed

    apply_canonical_seed(spec)
    return spec.to_json_dict()


def append_eval_to_buffer(
    buffer: SurrogateBuffer,
    result: EvalResult,
    *,
    emitter_type: str,
) -> None:
    """Append one evaluated candidate to the training buffer."""
    world_spec = world_spec_dict_for_buffer(result.world_spec)
    features = extract(result.world_spec)
    buffer.append(
        features=features,
        targets=targets_from_eval_result(result),
        emitter_type=emitter_type,
        world_spec=world_spec,
        metadata={"source": "live_eval"},
    )


def buffer_record(
    *,
    features: np.ndarray,
    targets: dict[str, float],
    emitter_type: str,
    world_spec: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one validated schema 2.0 JSON-serializable training record."""
    _validate_targets(targets)
    if not emitter_type:
        raise ValueError("emitter_type must be a non-empty string")
    if not isinstance(world_spec, dict) or not world_spec:
        raise ValueError("world_spec must be a non-empty dict")
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "emitter_type": emitter_type,
        "features": [
            float(value) for value in np.asarray(features, dtype=float).tolist()
        ],
        "targets": _serialize_targets(targets),
        "world_spec": dict(world_spec),
        "metadata": dict(metadata or {}),
    }


@dataclass
class SurrogateBuffer:
    """Append-only JSONL writer with in-memory batching."""

    path: Path | str
    flush_every: int = 32
    _pending: list[dict[str, Any]] = field(default_factory=list)
    _written: int = 0

    def append(
        self,
        *,
        features: np.ndarray,
        targets: dict[str, float],
        emitter_type: str,
        world_spec: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Queue one record and flush in batches."""
        record = buffer_record(
            features=features,
            targets=targets,
            emitter_type=emitter_type,
            world_spec=world_spec,
            metadata=metadata,
        )
        self._pending.append(record)
        if len(self._pending) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Persist all pending records to append-only JSONL file."""
        if not self._pending:
            return
        target = Path(self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            for row in self._pending:
                fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        self._written += len(self._pending)
        self._pending.clear()

    def stats(self) -> dict[str, int]:
        """Return current in-memory and persisted record counts."""
        return {"pending": len(self._pending), "written": self._written}


def _serialize_targets(targets: dict[str, float]) -> dict[str, float]:
    serialized = {key: float(targets[key]) for key in TARGET_KEYS}
    if FITNESS_TARGET_KEY in targets:
        serialized[FITNESS_TARGET_KEY] = float(targets[FITNESS_TARGET_KEY])
    return serialized


def _validate_targets(targets: dict[str, float]) -> None:
    missing = [key for key in TARGET_KEYS if key not in targets]
    if missing:
        raise ValueError(f"Missing required target keys: {missing}")

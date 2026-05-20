"""Append-only JSONL training buffer with batched flush."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from worldspace.surrogate.feature_extractor import FEATURE_SCHEMA_VERSION
from worldspace.surrogate.model import TARGET_KEYS

__all__ = [
    "SurrogateBuffer",
    "buffer_record",
]


def buffer_record(
    *,
    features: np.ndarray,
    targets: dict[str, float],
    emitter_type: str,
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one validated JSON-serializable training record."""
    _validate_targets(targets)
    if not emitter_type:
        raise ValueError("emitter_type must be a non-empty string")
    return {
        "feature_schema_version": str(feature_schema_version),
        "emitter_type": emitter_type,
        "features": [float(v) for v in np.asarray(features, dtype=float).tolist()],
        "targets": {k: float(targets[k]) for k in TARGET_KEYS},
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
        feature_schema_version: str = FEATURE_SCHEMA_VERSION,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Queue one record and flush in batches."""
        record = buffer_record(
            features=features,
            targets=targets,
            emitter_type=emitter_type,
            feature_schema_version=feature_schema_version,
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


def _validate_targets(targets: dict[str, float]) -> None:
    missing = [key for key in TARGET_KEYS if key not in targets]
    if missing:
        raise ValueError(f"Missing required target keys: {missing}")

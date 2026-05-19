"""MAP-Elites grid archive: one elite per behavioral niche and JSONL persistence."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from worldspace.illuminators.evaluation import EvalResult
from worldspace.metrics import WorldMetrics, metrics_vector_to_dict
from worldspace.specs.spec import WorldSpec

BC_MIN = 0.0
BC_MAX = 1.0
DEFAULT_GRID_RESOLUTION = 50
ARCHIVE_SCHEMA_VERSION = "1.2"
DEFAULT_ARCHIVE_JSONL_PATH = "output/map_elites_archive.jsonl"

__all__ = [
    "ARCHIVE_SCHEMA_VERSION",
    "BC_MAX",
    "BC_MIN",
    "DEFAULT_ARCHIVE_JSONL_PATH",
    "DEFAULT_GRID_RESOLUTION",
    "ArchiveElite",
    "EliteMetadata",
    "GridArchive",
    "InsertResult",
    "append_archive_line",
    "elite_from_eval",
    "elite_to_archive_record",
    "insert_and_persist",
    "insert_evaluated",
    "new_elite_metadata",
]


@dataclass
class EliteMetadata:
    """Emitter lineage and audit fields for one archive row."""

    id: str
    parent_id: str | None
    generated_by: str
    emitter_type: str
    timestamp: str
    prompt_version: str | None = None


@dataclass
class ArchiveElite:
    """Best candidate stored in one archive cell."""

    bin: tuple[int, int]
    fitness: float
    world_spec: WorldSpec | None = None
    measures: dict[str, float] | None = None
    metrics: WorldMetrics | None = None
    metadata: EliteMetadata | None = None


@dataclass(frozen=True)
class InsertResult:
    """Outcome of ``GridArchive.try_insert``."""

    accepted: bool
    improved: bool
    rejected: bool


class GridArchive:
    """In-memory ``resolution x resolution`` archive with fixed BC range [0, 1]."""

    def __init__(self, resolution: int = DEFAULT_GRID_RESOLUTION) -> None:
        if resolution < 1:
            msg = f"resolution must be >= 1, got {resolution}"
            raise ValueError(msg)
        self._resolution = resolution
        self._cells: list[ArchiveElite | None] = [None] * (resolution * resolution)

    @property
    def resolution(self) -> int:
        return self._resolution

    @property
    def bc_min(self) -> float:
        return BC_MIN

    @property
    def bc_max(self) -> float:
        return BC_MAX

    def get(self, i: int, j: int) -> ArchiveElite | None:
        """Return the elite at ``(i, j)`` or ``None`` if the cell is empty."""
        return self._cells[self._cell_index(i, j)]

    def is_empty(self, i: int, j: int) -> bool:
        return self.get(i, j) is None

    def filled_count(self) -> int:
        return sum(1 for cell in self._cells if cell is not None)

    def empty_count(self) -> int:
        return len(self._cells) - self.filled_count()

    def try_insert(self, elite: ArchiveElite) -> InsertResult:
        """Insert or replace the elite at ``elite.bin`` using strict fitness improvement."""
        i, j = elite.bin
        idx = self._cell_index(i, j)
        current = self._cells[idx]
        if current is None:
            self._cells[idx] = elite
            return InsertResult(accepted=True, improved=False, rejected=False)
        if elite.fitness > current.fitness:
            self._cells[idx] = elite
            return InsertResult(accepted=True, improved=True, rejected=False)
        return InsertResult(accepted=False, improved=False, rejected=True)

    def _cell_index(self, i: int, j: int) -> int:
        _validate_bin(i, j, self._resolution)
        return i * self._resolution + j


def new_elite_metadata(
    *,
    generated_by: str,
    emitter_type: str,
    parent_id: str | None = None,
    prompt_version: str | None = None,
    elite_id: str | None = None,
    timestamp: str | None = None,
) -> EliteMetadata:
    """Build metadata for a newly evaluated candidate."""
    return EliteMetadata(
        id=elite_id or str(uuid.uuid4()),
        parent_id=parent_id,
        generated_by=generated_by,
        emitter_type=emitter_type,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        prompt_version=prompt_version,
    )


def elite_from_eval(eval_result: EvalResult, metadata: EliteMetadata) -> ArchiveElite:
    """Build a full archive elite from an evaluation result."""
    return ArchiveElite(
        bin=eval_result.bin,
        fitness=eval_result.fitness,
        world_spec=eval_result.world_spec,
        measures=dict(eval_result.measures),
        metrics=eval_result.metrics,
        metadata=metadata,
    )


def insert_evaluated(
    archive: GridArchive,
    eval_result: EvalResult,
    metadata: EliteMetadata,
) -> InsertResult:
    """Insert an evaluated candidate into the archive at ``eval_result.bin``."""
    _validate_bin(eval_result.bin[0], eval_result.bin[1], archive.resolution)
    return archive.try_insert(elite_from_eval(eval_result, metadata))


def elite_to_archive_record(elite: ArchiveElite) -> dict:
    """Serialize one elite to a JSONL-ready dict (schema 1.2)."""
    if elite.world_spec is None:
        msg = "world_spec is required for archive JSONL records"
        raise ValueError(msg)
    if elite.measures is None:
        msg = "measures is required for archive JSONL records"
        raise ValueError(msg)
    if elite.metadata is None:
        msg = "metadata is required for archive JSONL records"
        raise ValueError(msg)

    record: dict = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "bin": [elite.bin[0], elite.bin[1]],
        "world_spec": elite.world_spec.to_json_dict(),
        "fitness": elite.fitness,
        "measures": dict(elite.measures),
        "metadata": {
            "id": elite.metadata.id,
            "parent_id": elite.metadata.parent_id,
            "generated_by": elite.metadata.generated_by,
            "emitter_type": elite.metadata.emitter_type,
            "timestamp": elite.metadata.timestamp,
            "prompt_version": elite.metadata.prompt_version,
        },
    }
    if elite.metrics is not None:
        record["metrics"] = metrics_vector_to_dict(elite.metrics.as_vector())
    return record


def append_archive_line(path: str | Path, record: dict) -> None:
    """Append one JSON object as a line to the MAP-Elites archive file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def insert_and_persist(
    archive: GridArchive,
    eval_result: EvalResult,
    metadata: EliteMetadata,
    jsonl_path: str | Path,
) -> InsertResult:
    """Insert into the archive and append JSONL only when the insert is accepted."""
    result = insert_evaluated(archive, eval_result, metadata)
    if result.accepted:
        elite = archive.get(eval_result.bin[0], eval_result.bin[1])
        assert elite is not None
        append_archive_line(jsonl_path, elite_to_archive_record(elite))
    return result


def _validate_bin(i: int, j: int, resolution: int) -> None:
    if i < 0 or i >= resolution or j < 0 or j >= resolution:
        msg = f"bin ({i}, {j}) out of range for resolution {resolution}"
        raise IndexError(msg)

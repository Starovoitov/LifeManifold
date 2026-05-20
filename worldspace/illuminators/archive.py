"""MAP-Elites grid archive: one elite per behavioral niche and JSONL persistence."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from worldspace.illuminators.evaluation import EvalResult
from worldspace.metrics import METRIC_KEYS, WorldMetrics, metrics_vector_to_dict
from worldspace.specs.spec import WorldSpec

logger = logging.getLogger(__name__)

InvalidLineMode = Literal["raise", "skip"]

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
    "InvalidLineMode",
    "append_archive_line",
    "archive_record_to_elite",
    "count_archive_jsonl_lines",
    "elite_from_eval",
    "elite_to_archive_record",
    "insert_and_persist",
    "insert_evaluated",
    "load_and_collapse_jsonl",
    "merge_archives",
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
        """Insert or replace at ``elite.bin``; replace only when fitness strictly improves."""
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


def archive_record_to_elite(record: dict) -> ArchiveElite:
    """Parse one JSONL archive record into an in-memory elite."""
    schema_version = record.get("schema_version")
    if schema_version != ARCHIVE_SCHEMA_VERSION:
        msg = f"unsupported schema_version {schema_version!r}"
        raise ValueError(msg)

    bin_raw = record["bin"]
    if not isinstance(bin_raw, list) or len(bin_raw) != 2:
        msg = "bin must be a list of two integers"
        raise ValueError(msg)
    bin_coord = (int(bin_raw[0]), int(bin_raw[1]))

    measures_raw = record["measures"]
    if not isinstance(measures_raw, dict):
        msg = "measures must be an object"
        raise ValueError(msg)

    metadata_raw = record["metadata"]
    if not isinstance(metadata_raw, dict):
        msg = "metadata must be an object"
        raise ValueError(msg)

    metrics = None
    if "metrics" in record and record["metrics"] is not None:
        metrics = _world_metrics_from_dict(record["metrics"])

    return ArchiveElite(
        bin=bin_coord,
        fitness=float(record["fitness"]),
        world_spec=WorldSpec.from_json_dict(record["world_spec"]),
        measures={
            "stability": float(measures_raw["stability"]),
            "diversity": float(measures_raw["diversity"]),
        },
        metrics=metrics,
        metadata=_elite_metadata_from_dict(metadata_raw),
    )


def count_archive_jsonl_lines(
    path: str | Path,
    *,
    on_invalid_line: InvalidLineMode = "skip",
) -> int:
    """Count parseable archive JSONL records.

    Blank and invalid lines are skipped by default (same rules as
    ``load_and_collapse_jsonl``).
    """
    target = Path(path)
    if not target.is_file():
        msg = f"archive file not found: {target}"
        raise FileNotFoundError(msg)
    return sum(
        1
        for _ in _iter_archive_elites_from_jsonl(
            target, on_invalid_line=on_invalid_line
        )
    )


def load_and_collapse_jsonl(
    path: str | Path,
    *,
    resolution: int = DEFAULT_GRID_RESOLUTION,
    on_invalid_line: InvalidLineMode = "skip",
) -> GridArchive:
    """Load JSONL lines and keep the best fitness per bin (first wins on ties)."""
    target = Path(path)
    if not target.is_file():
        msg = f"archive file not found: {target}"
        raise FileNotFoundError(msg)

    collapsed = _collapse_records_by_bin(target, on_invalid_line=on_invalid_line)
    archive = GridArchive(resolution)
    for elite in collapsed.values():
        _validate_bin(elite.bin[0], elite.bin[1], resolution)
        archive.try_insert(elite)
    return archive


def merge_archives(base: GridArchive, incoming: GridArchive) -> GridArchive:
    """Merge elites into ``base``; per cell the higher fitness wins (strict ``>``)."""
    if base.resolution != incoming.resolution:
        msg = (
            f"resolution mismatch: base={base.resolution}, "
            f"incoming={incoming.resolution}"
        )
        raise ValueError(msg)
    size = incoming.resolution
    for i in range(size):
        for j in range(size):
            elite = incoming.get(i, j)
            if elite is not None:
                base.try_insert(elite)
    return base


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


def _iter_archive_elites_from_jsonl(
    path: Path,
    *,
    on_invalid_line: InvalidLineMode,
) -> Iterator[ArchiveElite]:
    """Yield elites from JSONL lines; invalid lines skip or raise per ``on_invalid_line``."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                yield archive_record_to_elite(record)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                if on_invalid_line == "raise":
                    msg = f"invalid archive JSONL at {path}:{line_number}"
                    raise ValueError(msg) from exc
                logger.warning(
                    "skipping invalid archive line %s:%d: %s",
                    path,
                    line_number,
                    exc,
                )


def _collapse_records_by_bin(
    path: Path,
    *,
    on_invalid_line: InvalidLineMode,
) -> dict[tuple[int, int], ArchiveElite]:
    best: dict[tuple[int, int], ArchiveElite] = {}
    for elite in _iter_archive_elites_from_jsonl(path, on_invalid_line=on_invalid_line):
        key = elite.bin
        current = best.get(key)
        if current is None or elite.fitness > current.fitness:
            best[key] = elite
    return best


def _elite_metadata_from_dict(data: dict) -> EliteMetadata:
    parent_id = data.get("parent_id")
    return EliteMetadata(
        id=str(data["id"]),
        parent_id=None if parent_id is None else str(parent_id),
        generated_by=str(data["generated_by"]),
        emitter_type=str(data["emitter_type"]),
        timestamp=str(data["timestamp"]),
        prompt_version=data.get("prompt_version"),
    )


def _world_metrics_from_dict(data: dict) -> WorldMetrics:
    return WorldMetrics(
        **{key: float(data[key]) for key in METRIC_KEYS},
    )


def _validate_bin(i: int, j: int, resolution: int) -> None:
    if i < 0 or i >= resolution or j < 0 or j >= resolution:
        msg = f"bin ({i}, {j}) out of range for resolution {resolution}"
        raise IndexError(msg)

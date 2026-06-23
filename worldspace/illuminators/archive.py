"""MAP-Elites grid archive: one elite per behavioral niche and JSONL persistence."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from worldspace.illuminators.evaluation import EvalResult, bin_center, bin_index
from worldspace.illuminators.grid_neighbors import cardinal_neighbors_bounded
from worldspace.metrics import METRIC_KEYS, WorldMetrics, metrics_vector_to_dict
from worldspace.specs.spec import WorldSpec

if TYPE_CHECKING:
    from worldspace.illuminators.archive_protocol import ArchiveProtocol

logger = logging.getLogger(__name__)

InvalidLineMode = Literal["raise", "skip"]

BC_MIN = 0.0
BC_MAX = 1.0
DEFAULT_GRID_RESOLUTION = 50
ARCHIVE_SCHEMA_VERSION = "1.2"
ARCHIVE_SCHEMA_VERSION_V1_2 = "1.2"
ARCHIVE_SCHEMA_VERSION_V1_3 = "1.3"
DEFAULT_ARCHIVE_JSONL_PATH = "output/map_elites_archive.jsonl"

__all__ = [
    "ARCHIVE_SCHEMA_VERSION",
    "ARCHIVE_SCHEMA_VERSION_V1_2",
    "ARCHIVE_SCHEMA_VERSION_V1_3",
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
    "bin_ij_from_flat_cell_id",
    "count_archive_jsonl_lines",
    "cvt_cell_id",
    "flat_cell_id",
    "elite_from_eval",
    "elite_to_archive_record",
    "insert_and_persist",
    "insert_evaluated",
    "load_and_collapse_jsonl",
    "merge_archives",
    "new_elite_metadata",
    "normalize_archive_record_metadata",
    "prompt_version_from_json",
    "prompt_version_for_json",
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

    @property
    def bin_ij(self) -> tuple[int, int]:
        """Grid ``(i, j)`` or CVT ``(cell_id, 0)`` niche coordinates."""
        return self.bin


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

    @property
    def archive_type(self) -> str:
        return "grid"

    @property
    def n_cells(self) -> int:
        return self._resolution * self._resolution

    def get(self, i: int, j: int) -> ArchiveElite | None:
        """Return the elite at ``(i, j)`` or ``None`` if the cell is empty."""
        return self._cells[self._cell_index(i, j)]

    def is_empty(self, i: int, j: int) -> bool:
        return self.get(i, j) is None

    def get_cell(self, cell_id: int) -> ArchiveElite | None:
        """Return the elite for a flat niche index."""
        i, j = self.bin_from_cell_id(cell_id)
        return self.get(i, j)

    def is_empty_cell(self, cell_id: int) -> bool:
        return self.get_cell(cell_id) is None

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

    def cell_center(self, cell_id: int) -> tuple[float, float]:
        i, j = self.bin_from_cell_id(cell_id)
        return bin_center(i, j, self._resolution)

    def neighbors(self, cell_id: int) -> tuple[int, ...]:
        i, j = self.bin_from_cell_id(cell_id)
        adjacent = cardinal_neighbors_bounded(i, j, self._resolution)
        return tuple(sorted(self.cell_id_from_bin(neighbor) for neighbor in adjacent))

    def assign_cell_id(self, stability: float, diversity: float) -> int:
        i, j = bin_index(stability, diversity, self._resolution)
        return self.cell_id_from_bin((i, j))

    def cell_id_from_bin(self, bin_ij: tuple[int, int]) -> int:
        i, j = bin_ij
        return self._cell_index(i, j)

    def bin_from_cell_id(self, cell_id: int) -> tuple[int, int]:
        _validate_flat_cell_id(cell_id, self.n_cells, self._resolution)
        return divmod(cell_id, self._resolution)

    def _cell_index(self, i: int, j: int) -> int:
        _validate_bin(i, j, self._resolution)
        return i * self._resolution + j


def flat_cell_id(bin_ij: tuple[int, int], *, resolution: int) -> int:
    """Map a grid ``(i, j)`` bin to a flat niche index."""
    i, j = bin_ij
    _validate_bin(i, j, resolution)
    return i * resolution + j


def bin_ij_from_flat_cell_id(cell_id: int, *, resolution: int) -> tuple[int, int]:
    """Map a flat grid niche index back to ``(i, j)``."""
    n_cells = resolution * resolution
    _validate_flat_cell_id(cell_id, n_cells, resolution)
    return divmod(cell_id, resolution)


def cvt_cell_id(bin_ij: tuple[int, int]) -> int:
    """Return the CVT niche index stored in ``bin_ij[0]``."""
    return bin_ij[0]


def prompt_version_for_json(value: str | None) -> str:
    """Serialize ``prompt_version`` as a JSON string (empty when unset).

    NDJSON readers (e.g. Polars) infer column types from early rows; mixing
    ``null`` and strings breaks schema inference for mixed LLM / non-LLM archives.
    """
    if value is None or value == "":
        return ""
    return str(value)


def prompt_version_from_json(value: object) -> str | None:
    """Parse ``prompt_version`` from archive JSON; ``""`` and ``null`` → unset."""
    if value is None or value == "":
        return None
    return str(value)


def normalize_archive_record_metadata(record: dict) -> dict:
    """Return a shallow copy with stable ``metadata.prompt_version`` JSON typing."""
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return record
    normalized = dict(record)
    meta_copy = dict(metadata)
    meta_copy["prompt_version"] = prompt_version_for_json(
        prompt_version_from_json(meta_copy.get("prompt_version")),
    )
    normalized["metadata"] = meta_copy
    return normalized


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
    archive: "ArchiveProtocol",
    eval_result: EvalResult,
    metadata: EliteMetadata,
) -> InsertResult:
    """Insert an evaluated candidate into the archive at ``eval_result.bin``."""
    archive.cell_id_from_bin(eval_result.bin)
    return archive.try_insert(elite_from_eval(eval_result, metadata))


def elite_to_archive_record(
    elite: ArchiveElite,
    *,
    archive_type: str = "grid",
    schema_version: str = ARCHIVE_SCHEMA_VERSION,
    resolution: int | None = None,
) -> dict:
    """Serialize one elite to a JSONL-ready dict."""
    if elite.world_spec is None:
        msg = "world_spec is required for archive JSONL records"
        raise ValueError(msg)
    if elite.measures is None:
        msg = "measures is required for archive JSONL records"
        raise ValueError(msg)
    if elite.metadata is None:
        msg = "metadata is required for archive JSONL records"
        raise ValueError(msg)

    if schema_version not in (ARCHIVE_SCHEMA_VERSION_V1_2, ARCHIVE_SCHEMA_VERSION_V1_3):
        msg = f"unsupported schema_version {schema_version!r}"
        raise ValueError(msg)

    record: dict = {
        "schema_version": schema_version,
        "world_spec": elite.world_spec.to_json_dict(),
        "fitness": elite.fitness,
        "measures": dict(elite.measures),
        "metadata": {
            "id": elite.metadata.id,
            "parent_id": elite.metadata.parent_id,
            "generated_by": elite.metadata.generated_by,
            "emitter_type": elite.metadata.emitter_type,
            "timestamp": elite.metadata.timestamp,
            "prompt_version": prompt_version_for_json(elite.metadata.prompt_version),
        },
    }
    if schema_version == ARCHIVE_SCHEMA_VERSION_V1_2:
        record["bin"] = [elite.bin[0], elite.bin[1]]
    else:
        record["archive_type"] = archive_type
        if archive_type == "grid":
            if resolution is None:
                msg = "resolution is required for schema 1.3 grid records"
                raise ValueError(msg)
            cell_id = flat_cell_id(elite.bin, resolution=resolution)
            record["cell_id"] = int(cell_id)
            record["bin"] = [elite.bin[0], elite.bin[1]]
        elif archive_type == "cvt":
            cell_id = cvt_cell_id(elite.bin)
            record["cell_id"] = int(cell_id)
        else:
            msg = f"unsupported archive_type {archive_type!r}"
            raise ValueError(msg)
    if elite.metrics is not None:
        record["metrics"] = metrics_vector_to_dict(elite.metrics.as_vector())
    return record


def archive_record_to_elite(
    record: dict,
    *,
    resolution: int = DEFAULT_GRID_RESOLUTION,
) -> ArchiveElite:
    """Parse one JSONL archive record into an in-memory elite."""
    schema_version = record.get("schema_version")
    if schema_version not in (ARCHIVE_SCHEMA_VERSION_V1_2, ARCHIVE_SCHEMA_VERSION_V1_3):
        msg = f"unsupported schema_version {schema_version!r}"
        raise ValueError(msg)

    if schema_version == ARCHIVE_SCHEMA_VERSION_V1_2:
        bin_coord = _parse_bin_coord(record["bin"])
    else:
        archive_type = str(record.get("archive_type", "grid"))
        cell_raw = record.get("cell_id")
        if cell_raw is None:
            if archive_type == "grid" and "bin" in record:
                bin_coord = _parse_bin_coord(record["bin"])
            else:
                msg = "cell_id is required for schema 1.3 records"
                raise ValueError(msg)
        else:
            cell_id = int(cell_raw)
            if archive_type == "grid":
                bin_coord = bin_ij_from_flat_cell_id(cell_id, resolution=resolution)
            elif archive_type == "cvt":
                bin_coord = (cell_id, 0)
            else:
                msg = f"unsupported archive_type {archive_type!r}"
                raise ValueError(msg)

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
    resolution: int = DEFAULT_GRID_RESOLUTION,
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
            target,
            resolution=resolution,
            on_invalid_line=on_invalid_line,
        )
    )


def load_and_collapse_jsonl(
    path: str | Path,
    *,
    archive_type: str = "grid",
    resolution: int = DEFAULT_GRID_RESOLUTION,
    centroids_path: str | Path | None = None,
    on_invalid_line: InvalidLineMode = "skip",
) -> "ArchiveProtocol":
    """Load JSONL lines and keep the best fitness per bin (first wins on ties)."""
    target = Path(path)
    if not target.is_file():
        msg = f"archive file not found: {target}"
        raise FileNotFoundError(msg)

    collapsed = _collapse_records_by_bin(
        target,
        resolution=resolution,
        on_invalid_line=on_invalid_line,
    )
    from worldspace.illuminators.archive_factory import (
        ArchiveFactoryConfig,
        create_empty_archive,
        normalize_archive_type,
    )

    archive = create_empty_archive(
        ArchiveFactoryConfig(
            archive_type=normalize_archive_type(archive_type),
            resolution=resolution,
        ),
        centroids_path=centroids_path,
    )
    for elite in collapsed.values():
        archive.cell_id_from_bin(elite.bin)
        archive.try_insert(elite)
    return archive


def merge_archives(
    base: "ArchiveProtocol", incoming: "ArchiveProtocol"
) -> "ArchiveProtocol":
    """Merge elites into ``base``; per niche the higher fitness wins (strict ``>``)."""
    _validate_archive_compatibility(base, incoming)
    for cell_id in range(incoming.n_cells):
        elite = incoming.get_cell(cell_id)
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
    archive: "ArchiveProtocol",
    eval_result: EvalResult,
    metadata: EliteMetadata,
    jsonl_path: str | Path,
    *,
    schema_version: str = ARCHIVE_SCHEMA_VERSION,
) -> InsertResult:
    """Insert into the archive and append JSONL only when the insert is accepted."""
    result = insert_evaluated(archive, eval_result, metadata)
    if result.accepted:
        elite = archive.get_cell(archive.cell_id_from_bin(eval_result.bin))
        assert elite is not None
        resolution = getattr(archive, "resolution", None)
        append_archive_line(
            jsonl_path,
            elite_to_archive_record(
                elite,
                archive_type=archive.archive_type,
                schema_version=schema_version,
                resolution=resolution,
            ),
        )
    return result


def _iter_archive_elites_from_jsonl(
    path: Path,
    *,
    resolution: int,
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
                yield archive_record_to_elite(record, resolution=resolution)
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
    resolution: int,
    on_invalid_line: InvalidLineMode,
) -> dict[tuple[int, int], ArchiveElite]:
    best: dict[tuple[int, int], ArchiveElite] = {}
    for elite in _iter_archive_elites_from_jsonl(
        path,
        resolution=resolution,
        on_invalid_line=on_invalid_line,
    ):
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
        prompt_version=prompt_version_from_json(data.get("prompt_version")),
    )


def _world_metrics_from_dict(data: dict) -> WorldMetrics:
    return WorldMetrics(
        **{key: float(data[key]) for key in METRIC_KEYS},
    )


def _validate_archive_compatibility(
    base: "ArchiveProtocol",
    incoming: "ArchiveProtocol",
) -> None:
    if base.archive_type != incoming.archive_type:
        msg = (
            f"archive_type mismatch: base={base.archive_type!r}, "
            f"incoming={incoming.archive_type!r}"
        )
        raise ValueError(msg)
    if base.n_cells != incoming.n_cells:
        msg = f"n_cells mismatch: base={base.n_cells}, incoming={incoming.n_cells}"
        raise ValueError(msg)
    if base.archive_type == "grid":
        base_resolution = getattr(base, "resolution", None)
        incoming_resolution = getattr(incoming, "resolution", None)
        if base_resolution != incoming_resolution:
            msg = (
                f"resolution mismatch: base={base_resolution}, "
                f"incoming={incoming_resolution}"
            )
            raise ValueError(msg)
    if base.archive_type == "cvt":
        base_centroids = getattr(base, "centroids", None)
        incoming_centroids = getattr(incoming, "centroids", None)
        if base_centroids is None or incoming_centroids is None:
            msg = "cvt archives must expose centroids for compatibility checks"
            raise ValueError(msg)
        if not _same_centroids(base_centroids, incoming_centroids):
            msg = "cvt centroids mismatch between archives"
            raise ValueError(msg)


def _same_centroids(left: object, right: object) -> bool:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    return bool(
        left_arr.shape == right_arr.shape and np.array_equal(left_arr, right_arr)
    )


def _parse_bin_coord(value: object) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        msg = "bin must be a list of two integers"
        raise ValueError(msg)
    return (int(value[0]), int(value[1]))


def _validate_bin(i: int, j: int, resolution: int) -> None:
    if i < 0 or i >= resolution or j < 0 or j >= resolution:
        msg = f"bin ({i}, {j}) out of range for resolution {resolution}"
        raise IndexError(msg)


def _validate_flat_cell_id(cell_id: int, n_cells: int, resolution: int) -> None:
    if cell_id < 0 or cell_id >= n_cells:
        msg = f"cell_id {cell_id} out of range for {n_cells} niches"
        raise IndexError(msg)
    _, j = divmod(cell_id, resolution)
    if j < 0 or j >= resolution:
        msg = f"cell_id {cell_id} is not a valid flat index for resolution {resolution}"
        raise IndexError(msg)

"""Factory helpers for grid and CVT MAP-Elites archives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from worldspace.illuminators.archive import DEFAULT_GRID_RESOLUTION, GridArchive
from worldspace.illuminators.archive_protocol import ArchiveProtocol
from worldspace.illuminators.cvt import (
    DEFAULT_LLOYD_ITERATIONS,
    centroids_path_for_output,
    generate_centroids,
    load_centroids,
    save_centroids,
)
from worldspace.illuminators.cvt_archive import CvtArchive

if TYPE_CHECKING:
    from worldspace.illuminators.scheduler import SchedulerConfig

__all__ = [
    "ArchiveFactoryConfig",
    "archive_factory_config_from_scheduler",
    "create_archive",
    "create_empty_archive",
    "create_grid_archive",
    "normalize_archive_type",
]


@dataclass(frozen=True)
class ArchiveFactoryConfig:
    """Runtime archive settings derived from scheduler YAML or direct construction."""

    archive_type: Literal["grid", "cvt"] = "grid"
    resolution: int = DEFAULT_GRID_RESOLUTION
    n_centroids: int = 50 * 50  # default cvt size
    cvt_seed: int = 0
    lloyd_iterations: int = DEFAULT_LLOYD_ITERATIONS


def create_archive(
    config: ArchiveFactoryConfig,
    *,
    output_dir: str | Path | None = None,
    centroids_path: str | Path | None = None,
) -> ArchiveProtocol:
    """Build a grid or CVT archive, persisting CVT centroids when ``output_dir`` is set."""
    if config.archive_type == "grid":
        return GridArchive(config.resolution)
    return _create_cvt_archive(
        config,
        output_dir=output_dir,
        centroids_path=centroids_path,
    )


def create_empty_archive(
    config: ArchiveFactoryConfig,
    *,
    centroids_path: str | Path | None = None,
) -> ArchiveProtocol:
    """Build an empty in-memory archive for JSONL collapse / resume (no centroid generation)."""
    if config.archive_type == "grid":
        return GridArchive(config.resolution)
    if centroids_path is None:
        msg = "centroids_path is required when loading a CVT archive"
        raise ValueError(msg)
    path = Path(centroids_path).expanduser()
    if not path.is_file():
        msg = f"centroids file not found: {path}"
        raise FileNotFoundError(msg)
    return CvtArchive(load_centroids(path))


def normalize_archive_type(archive_type: str) -> Literal["grid", "cvt"]:
    """Validate and normalize a runtime archive type string."""
    if archive_type == "grid":
        return "grid"
    if archive_type == "cvt":
        return "cvt"
    msg = f"unsupported archive_type {archive_type!r}"
    raise ValueError(msg)


def create_grid_archive(resolution: int = DEFAULT_GRID_RESOLUTION) -> GridArchive:
    """Convenience wrapper for the legacy grid-only call sites."""
    return GridArchive(resolution)


def archive_factory_config_from_scheduler(
    config: SchedulerConfig,
) -> ArchiveFactoryConfig:
    """Build archive factory settings from a loaded scheduler config."""
    return ArchiveFactoryConfig(
        archive_type=config.archive_type,
        resolution=config.grid_resolution,
        n_centroids=config.n_centroids,
        cvt_seed=config.cvt_seed,
        lloyd_iterations=config.lloyd_iterations,
    )


def _create_cvt_archive(
    config: ArchiveFactoryConfig,
    *,
    output_dir: str | Path | None,
    centroids_path: str | Path | None,
) -> CvtArchive:
    resolved_path = _resolve_centroids_path(
        output_dir=output_dir,
        centroids_path=centroids_path,
    )
    if resolved_path is not None and resolved_path.is_file():
        centroids = load_centroids(resolved_path)
        return CvtArchive(centroids)

    centroids = generate_centroids(
        config.n_centroids,
        seed=config.cvt_seed,
        lloyd_iterations=config.lloyd_iterations,
    )
    if resolved_path is not None:
        save_centroids(resolved_path, centroids)
    return CvtArchive(centroids)


def _resolve_centroids_path(
    *,
    output_dir: str | Path | None,
    centroids_path: str | Path | None,
) -> Path | None:
    if centroids_path is not None:
        return Path(centroids_path).expanduser()
    if output_dir is not None:
        return centroids_path_for_output(output_dir)
    return None

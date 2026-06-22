"""Factory helpers for grid and CVT MAP-Elites archives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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

__all__ = [
    "ArchiveFactoryConfig",
    "create_archive",
    "create_grid_archive",
]


@dataclass(frozen=True)
class ArchiveFactoryConfig:
    """Runtime archive settings until scheduler schema 1.3 is wired in."""

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


def create_grid_archive(resolution: int = DEFAULT_GRID_RESOLUTION) -> GridArchive:
    """Convenience wrapper for the legacy grid-only call sites."""
    return GridArchive(resolution)


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

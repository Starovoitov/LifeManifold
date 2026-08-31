"""Authoritative canonical genotype and archive fingerprints."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from worldspace.attribution.hashing import canonical_sha256
from worldspace.mazes.spec import MazeSpec
from worldspace.specs.spec import WorldSpec


def canonical_ca_genotype(spec: WorldSpec) -> dict[str, Any]:
    """Return the seed-free canonical CA genotype payload."""
    return spec.to_canonical_dict()


def ca_genotype_hash(spec: WorldSpec) -> str:
    """Return the authoritative full SHA-256 for a CA genotype."""
    return canonical_sha256(canonical_ca_genotype(spec))


def canonical_maze_genotype(spec: MazeSpec) -> dict[str, object]:
    """Return canonical maze tile rows without display-only metadata."""
    return {"rows": list(spec.rows)}


def maze_genotype_hash(spec: MazeSpec) -> str:
    """Return the authoritative full SHA-256 for a maze genotype."""
    return canonical_sha256(canonical_maze_genotype(spec))


def archive_fingerprint(
    entries: Iterable[Mapping[str, Any]],
    *,
    evaluator_hash: str,
) -> str:
    """Hash canonical elite facts sorted by cell ID.

    Each entry must provide ``cell_id``, ``genotype_hash``, ``descriptors``,
    and ``fitness``. Paths, timestamps, and reader-facing metadata are excluded.
    """
    canonical_entries = [
        {
            "cell_id": int(entry["cell_id"]),
            "genotype_hash": str(entry["genotype_hash"]),
            "descriptors": entry["descriptors"],
            "fitness": float(entry["fitness"]),
            "evaluator_hash": evaluator_hash,
        }
        for entry in entries
    ]
    canonical_entries.sort(key=lambda entry: int(entry["cell_id"]))
    return canonical_sha256(canonical_entries)

"""Strict read-only helpers for native run artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from worldspace.attribution.adapters.base import NormalizationError
from worldspace.attribution.manifest import RunManifest
from worldspace.attribution.records import ArtifactEntry, ArtifactManifest

PrivacyClass = Literal["public", "private", "discard"]


@dataclass(frozen=True)
class ArtifactSource:
    """Metadata needed to register one existing native artifact."""

    path: Path
    schema_version: str | None
    privacy_class: PrivacyClass = "public"
    producer: str = "native-runner"


def require_file(path: Path) -> Path:
    """Return an existing regular file or fail normalization."""
    if not path.is_file():
        raise NormalizationError(f"required native artifact is missing: {path}")
    return path


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one strict JSON object."""
    require_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NormalizationError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise NormalizationError(f"expected JSON object at {path}")
    return value


def read_jsonl_objects(path: Path) -> tuple[dict[str, Any], ...]:
    """Read nonblank JSONL rows and reject malformed/non-object lines."""
    require_file(path)
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise NormalizationError(f"cannot read JSONL {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise NormalizationError(
                f"malformed JSONL at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise NormalizationError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return tuple(rows)


def sha256_file(path: Path) -> str:
    """Hash an existing file without modifying it."""
    require_file(path)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise NormalizationError(f"cannot hash native artifact {path}: {exc}") from exc
    return digest.hexdigest()


def build_artifact_manifest(
    manifest: RunManifest,
    sources: Mapping[str, ArtifactSource],
) -> ArtifactManifest:
    """Build integrity metadata for existing files only."""
    entries: list[ArtifactEntry] = []
    for logical_name, source in sorted(sources.items()):
        path = require_file(source.path)
        entries.append(
            ArtifactEntry(
                logical_name=logical_name,
                path=str(path.resolve()),
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                schema_version=source.schema_version,
                privacy_class=source.privacy_class,
                producer=source.producer,
            )
        )
    return ArtifactManifest(
        run_id=manifest.run_id,
        run_manifest_hash=manifest.run_manifest_hash,
        artifacts=tuple(entries),
    )


def existing_sources(
    items: Iterable[tuple[str, ArtifactSource]],
) -> dict[str, ArtifactSource]:
    """Keep optional artifact sources that exist as regular files."""
    return {name: source for name, source in items if source.path.is_file()}

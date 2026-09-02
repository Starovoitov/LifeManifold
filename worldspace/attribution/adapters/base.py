"""Shared read-only normalization adapter contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from worldspace.attribution.manifest import AdapterCapabilities, RunManifest
from worldspace.attribution.records import (
    ArtifactManifest,
    BudgetCheckpoint,
    ProposalEvent,
    RunSummary,
)


@dataclass(frozen=True)
class NormalizationIssue:
    """One non-fatal loss of information in a native artifact bundle."""

    code: str
    message: str


class NormalizationError(ValueError):
    """Raised when native artifacts cannot be normalized safely."""


@dataclass(frozen=True)
class NativeRunInputs:
    """Paths supplied to a normalizer without granting write behavior."""

    run_dir: Path
    initial_archive_path: Path | None = None
    centroids_path: Path | None = None

    def path(self, filename: str) -> Path:
        """Return a path below the native run directory."""
        return self.run_dir / filename


@dataclass(frozen=True)
class NormalizedRunBundle:
    """In-memory common records derived from one native run."""

    summary: RunSummary
    checkpoints: tuple[BudgetCheckpoint, ...]
    events: tuple[ProposalEvent, ...]
    artifacts: ArtifactManifest
    issues: tuple[NormalizationIssue, ...]


class ReadOnlyNormalizationAdapter(Protocol):
    """Minimal adapter surface."""

    def capabilities(self) -> AdapterCapabilities: ...

    def normalize(
        self,
        manifest: RunManifest,
        inputs: NativeRunInputs,
    ) -> NormalizedRunBundle: ...

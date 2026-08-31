"""JSON Schema bundle for Phase 1 attribution records."""

from __future__ import annotations

from typing import Any

from worldspace.attribution.design import DesignMatrix, JobPlan
from worldspace.attribution.manifest import (
    AdapterCapabilities,
    RunManifest,
    StudyManifest,
)
from worldspace.attribution.records import (
    ArtifactManifest,
    BudgetCheckpoint,
    ProposalEvent,
    RunSummary,
)

SCHEMA_MODELS = (
    StudyManifest,
    RunManifest,
    AdapterCapabilities,
    ProposalEvent,
    BudgetCheckpoint,
    RunSummary,
    ArtifactManifest,
    DesignMatrix,
    JobPlan,
)


def attribution_schema_bundle() -> dict[str, dict[str, Any]]:
    """Return JSON Schemas keyed by stable record name."""
    return {
        model.__name__: model.model_json_schema(mode="validation")
        for model in SCHEMA_MODELS
    }

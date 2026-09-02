"""Shared normalization for public-domain run directories."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from worldspace.attribution.adapters.base import (
    NativeRunInputs,
    NormalizationError,
    NormalizationIssue,
    NormalizedRunBundle,
)
from worldspace.attribution.adapters.common import (
    archive_state_from_fitnesses,
    assert_close,
    checkpoints_from_trace,
)
from worldspace.attribution.adapters.io import (
    ArtifactSource,
    build_artifact_manifest,
    existing_sources,
    read_json_object,
    read_jsonl_objects,
)
from worldspace.attribution.hashing import canonical_sha256
from worldspace.attribution.manifest import (
    AdapterCapabilities,
    BudgetAxis,
    RunManifest,
)
from worldspace.attribution.public_loop import (
    SUMMARY_FILENAME,
    SUMMARY_SCHEMA,
    TRACE_FILENAME,
)
from worldspace.attribution.records import (
    BudgetCounters,
    RunSummary,
    SourceCompleteness,
)


def normalize_public_run(
    manifest: RunManifest,
    inputs: NativeRunInputs,
    *,
    capabilities: AdapterCapabilities,
    archive_filename: str,
    fitness_key: str = "fitness",
) -> NormalizedRunBundle:
    """Normalize summary + archive + trace written by nas201/pcg runners."""
    _validate_manifest(manifest, capabilities)
    summary_path = inputs.path(SUMMARY_FILENAME)
    archive_path = inputs.path(archive_filename)
    trace_path = inputs.path(TRACE_FILENAME)
    summary = read_json_object(summary_path)
    _validate_summary_identity(manifest, summary, capabilities.domain_id)
    capacity = int(summary["n_cells"])
    archive_rows = read_jsonl_objects(archive_path)
    fitnesses = [float(row[fitness_key]) for row in archive_rows]
    final_state = archive_state_from_fitnesses(fitnesses, capacity=capacity)
    assert_close(final_state.coverage, summary.get("coverage"), label="coverage")
    assert_close(final_state.raw_qd_score, summary.get("qd_score"), label="qd_score")
    assert_close(
        float(final_state.occupied_cells),
        summary.get("filled_cells"),
        label="filled_cells",
    )
    trace_rows = read_jsonl_objects(trace_path)
    if not trace_rows:
        raise NormalizationError("public run archive trace must not be empty")
    last = trace_rows[-1]
    assert_close(
        _optional_float(last.get("filled_cells")),
        summary.get("filled_cells"),
        label="trace filled_cells",
    )
    checkpoints = checkpoints_from_trace(
        trace_rows,
        run_id=manifest.run_id,
        capacity=capacity,
    )
    counters = BudgetCounters(
        proposal_slots=int(summary["proposals"]),
        valid_proposals=int(summary["valid_proposals"]),
        evaluator_attempts=int(summary["evaluations"]),
        # Public runners evaluate synchronously, so every attempt completes.
        evaluator_completions=int(summary["evaluations"]),
        llm_attempts=_optional_int(summary.get("llm_calls_attempted")),
        llm_completions=_optional_int(summary.get("llm_calls_completed")),
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        evaluator_seconds=None,
        llm_latency_seconds=None,
        wall_seconds=_optional_float(summary.get("wall_seconds")),
        monetary_cost=None,
    )
    counter_completeness: dict[BudgetAxis, SourceCompleteness] = {
        "proposal": "observed",
        "valid_proposal": "observed",
        "evaluation": "observed",
        "llm_call_attempted": "observed",
        "llm_call_completed": "observed",
        "prompt_token": "unavailable",
        "completion_token": "unavailable",
        "token": "unavailable",
        "evaluator_wall_time": "unavailable",
        "llm_latency": "unavailable",
        "wall_time": "observed",
        "monetary": "unavailable",
    }
    normalized_summary = RunSummary(
        run_id=manifest.run_id,
        study_id=manifest.study_id,
        arm_id=manifest.arm_id,
        pair_id=manifest.pair_id,
        evidence_tier=manifest.evidence_tier,
        domain_id=manifest.domain_id,
        domain_version=manifest.domain_version,
        adapter_id=manifest.adapter_id,
        adapter_version=manifest.adapter_version,
        evaluator_hash=canonical_sha256(manifest.evaluator),
        seed=manifest.seed,
        domain_instance_id=manifest.domain_instance_id,
        initial_archive_hash=manifest.initial_archive_hash,
        protocol_hash=manifest.protocol_hash,
        study_manifest_hash=manifest.study_manifest_hash,
        run_manifest_hash=manifest.run_manifest_hash,
        treatment_hash=manifest.treatment_hash,
        event_completeness="summary_only",
        final_counters=counters,
        counter_completeness=counter_completeness,
        final_archive=final_state,
        archive_metric_completeness={
            "coverage": "observed",
            "raw_qd_score": "observed",
            "normalized_qd_score": "derived",
            "maximum_elite_quality": "derived",
            "occupied_mean_quality": "derived",
        },
        completed=bool(summary.get("completed", True)),
        failure_reason=None,
    )
    artifacts = build_artifact_manifest(
        manifest,
        existing_sources(
            (
                ("native_summary", ArtifactSource(summary_path, SUMMARY_SCHEMA)),
                ("native_archive", ArtifactSource(archive_path, SUMMARY_SCHEMA)),
                ("native_trace", ArtifactSource(trace_path, None)),
            )
        ),
    )
    issues = (
        NormalizationIssue(
            "public.events_summary_only",
            "public runner writes summary/trace; prospective capture is optional later",
        ),
    )
    return NormalizedRunBundle(
        summary=normalized_summary,
        checkpoints=checkpoints,
        events=(),
        artifacts=artifacts,
        issues=issues,
    )


def _validate_manifest(
    manifest: RunManifest,
    capabilities: AdapterCapabilities,
) -> None:
    if manifest.domain_id != capabilities.domain_id:
        raise NormalizationError(
            f"{capabilities.domain_id} adapter cannot normalize "
            f"domain {manifest.domain_id!r}"
        )
    if (
        manifest.adapter_id != capabilities.adapter_id
        or manifest.adapter_version != capabilities.adapter_version
    ):
        raise NormalizationError(
            f"run manifest does not identify the current {capabilities.domain_id} adapter"
        )
    archive_type = manifest.treatment.replacement.parameters.get("archive_type")
    if archive_type != "grid":
        raise NormalizationError(
            f"{capabilities.domain_id} runner requires grid archive"
        )
    if manifest.treatment.gate.kind != "off":
        raise NormalizationError(
            f"{capabilities.domain_id} confirmatory gate must be off"
        )


def _validate_summary_identity(
    manifest: RunManifest,
    summary: Mapping[str, Any],
    domain_id: str,
) -> None:
    if summary.get("schema_version") != SUMMARY_SCHEMA:
        raise NormalizationError(
            f"unsupported public summary schema {summary.get('schema_version')!r}"
        )
    if summary.get("domain") != domain_id and summary.get("benchmark") != domain_id:
        raise NormalizationError(f"native summary is not marked as {domain_id}")
    if int(summary["seed"]) != manifest.seed:
        raise NormalizationError("native seed does not match run manifest")
    if summary.get("target_selection") != manifest.treatment.selector.kind:
        raise NormalizationError("native selector does not match treatment")
    if summary.get("generator") != manifest.treatment.generator.kind:
        raise NormalizationError("native generator does not match treatment")
    if summary.get("allocation") != manifest.treatment.allocation.kind:
        raise NormalizationError("native allocation does not match treatment")
    expected_prompt = manifest.treatment.prompt_channel.kind
    if summary.get("prompt_channel") != expected_prompt:
        raise NormalizationError("native prompt_channel does not match treatment")
    repair = summary.get("repair")
    expected_repair = manifest.treatment.repair_fallback.kind
    if expected_repair == "genetic_fallback":
        # genetic_fallback is parse-path; realized repair remains identity/counts.
        if repair not in {"identity", "structural_counts"}:
            raise NormalizationError("native repair does not match treatment")
    elif repair != expected_repair:
        raise NormalizationError("native repair does not match treatment")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)  # type: ignore[arg-type]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)  # type: ignore[arg-type]

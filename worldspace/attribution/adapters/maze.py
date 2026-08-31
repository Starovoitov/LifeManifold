"""Read-only normalizer for native maze MAP-Elites runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from worldspace.attribution.adapters.base import (
    NativeRunInputs,
    NormalizationError,
    NormalizationIssue,
    NormalizedRunBundle,
)
from worldspace.attribution.adapters.ca import (
    _optional_float,
    _optional_int,
    _required_int,
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
from worldspace.attribution.capabilities import maze_capabilities
from worldspace.attribution.capture import (
    BUDGET_LEDGER_FILENAME,
    PROSPECTIVE_EVENT_FILENAME,
    read_budget_ledger,
    read_prospective_events,
    reconcile_budget_ledger,
    reconcile_event_ledger,
)
from worldspace.attribution.hashing import canonical_sha256
from worldspace.attribution.manifest import SCHEMA_VERSION, BudgetAxis, RunManifest
from worldspace.attribution.records import (
    ArchiveMetric,
    ArchiveState,
    BudgetCounters,
    ProposalEvent,
    RunSummary,
    SourceCompleteness,
)

SUMMARY_FILENAME = "nightly_run_summary.json"
ARCHIVE_FILENAME = "maze_archive.jsonl"
TRACE_FILENAME = "archive_trace.jsonl"
SURROGATE_FILENAME = "surrogate_archive.jsonl"
LLM_CALL_FILENAME = "llm_call_log.jsonl"


class MazeNormalizationAdapter:
    """Normalize existing maze artifacts without creating proposal events."""

    def capabilities(self):
        return maze_capabilities()

    def normalize(
        self,
        manifest: RunManifest,
        inputs: NativeRunInputs,
    ) -> NormalizedRunBundle:
        self._validate_manifest(manifest)
        summary_path = inputs.path(SUMMARY_FILENAME)
        archive_path = inputs.path(ARCHIVE_FILENAME)
        trace_path = inputs.path(TRACE_FILENAME)
        surrogate_path = inputs.path(SURROGATE_FILENAME)
        llm_path = inputs.path(LLM_CALL_FILENAME)

        summary = read_json_object(summary_path)
        self._validate_summary_identity(manifest, summary)
        capacity = _required_int(summary, "n_cells")
        archive_rows = read_jsonl_objects(archive_path)
        final_state = _maze_archive_state(archive_rows, capacity=capacity)
        _validate_terminal_archive(summary, final_state)

        trace_rows = read_jsonl_objects(trace_path)
        if not trace_rows:
            raise NormalizationError("maze archive trace must not be empty")
        _validate_trace_terminal(trace_rows[-1], final_state, summary)

        prospective_path = inputs.path(PROSPECTIVE_EVENT_FILENAME)
        ledger_path = inputs.path(BUDGET_LEDGER_FILENAME)
        if prospective_path.is_file():
            try:
                events = read_prospective_events(
                    prospective_path,
                    manifest=manifest,
                )
            except ValueError as exc:
                raise NormalizationError(str(exc)) from exc
            _validate_prospective_events(events, summary, final_state)
            if not ledger_path.is_file():
                raise NormalizationError(
                    "prospective maze run is missing budget_ledger.jsonl"
                )
            try:
                ledger_checkpoints = read_budget_ledger(
                    ledger_path,
                    run_id=manifest.run_id,
                )
                reconcile_budget_ledger(
                    ledger_checkpoints,
                    events,
                    llm_applicable=bool(summary.get("llm_enabled")),
                )
            except ValueError as exc:
                raise NormalizationError(str(exc)) from exc
        else:
            events = ()
            ledger_checkpoints = ()
        counters, counter_completeness = _terminal_counters(
            summary,
            events=events,
        )
        checkpoints = ledger_checkpoints or checkpoints_from_trace(
            trace_rows,
            run_id=manifest.run_id,
            capacity=capacity,
        )
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
            event_completeness="full" if events else "summary_only",
            final_counters=counters,
            counter_completeness=counter_completeness,
            final_archive=final_state,
            archive_metric_completeness=_archive_completeness(),
            completed=True,
            failure_reason=None,
        )
        if events:
            try:
                reconcile_event_ledger(
                    events,
                    normalized_summary,
                    llm_applicable=bool(summary.get("llm_enabled")),
                )
            except ValueError as exc:
                raise NormalizationError(str(exc)) from exc
        artifacts = build_artifact_manifest(
            manifest,
            existing_sources(
                (
                    (
                        "native_summary",
                        ArtifactSource(summary_path, "maze-1.0"),
                    ),
                    (
                        "native_archive",
                        ArtifactSource(archive_path, "maze-1.0"),
                    ),
                    ("native_trace", ArtifactSource(trace_path, None)),
                    (
                        "native_surrogate_log",
                        ArtifactSource(surrogate_path, None),
                    ),
                    (
                        "native_llm_call_log",
                        ArtifactSource(
                            llm_path,
                            "2",
                            privacy_class="private",
                        ),
                    ),
                    (
                        "prospective_attribution_events",
                        ArtifactSource(
                            prospective_path,
                            SCHEMA_VERSION,
                            producer="attribution-sidecar",
                        ),
                    ),
                    (
                        "prospective_budget_ledger",
                        ArtifactSource(
                            ledger_path,
                            SCHEMA_VERSION,
                            producer="attribution-sidecar",
                        ),
                    ),
                )
            ),
        )
        issues = [
            NormalizationIssue(
                "maze.native_identity_manifest_only",
                "native summary does not carry evaluator or treatment hashes",
            ),
        ]
        if not events:
            issues.append(
                NormalizationIssue(
                    "maze.events_summary_only",
                    "native maze logs lack evaluation and insertion outcomes per slot",
                )
            )
        if bool(summary.get("llm_enabled")) and not llm_path.is_file():
            issues.append(
                NormalizationIssue(
                    "maze.llm_usage_missing",
                    "LLM token and latency usage is unavailable",
                )
            )
        return NormalizedRunBundle(
            summary=normalized_summary,
            checkpoints=checkpoints,
            events=events,
            artifacts=artifacts,
            issues=tuple(issues),
        )

    @staticmethod
    def _validate_manifest(manifest: RunManifest) -> None:
        capabilities = maze_capabilities()
        if manifest.domain_id != capabilities.domain_id:
            raise NormalizationError(
                f"maze adapter cannot normalize domain {manifest.domain_id!r}"
            )
        if (
            manifest.adapter_id != capabilities.adapter_id
            or manifest.adapter_version != capabilities.adapter_version
        ):
            raise NormalizationError(
                "run manifest does not identify the current maze adapter"
            )
        if manifest.treatment.initialization.kind != "empty":
            raise NormalizationError("native maze runner only supports empty start")
        if manifest.initial_archive_hash != canonical_sha256([]):
            raise NormalizationError("empty maze initial archive hash is not canonical")
        archive_type = manifest.treatment.replacement.parameters.get("archive_type")
        if archive_type != "grid":
            raise NormalizationError("native maze runner requires grid archive")

    @staticmethod
    def _validate_summary_identity(
        manifest: RunManifest,
        summary: Mapping[str, Any],
    ) -> None:
        if summary.get("schema_version") != "maze-1.0":
            raise NormalizationError(
                f"unsupported maze summary schema {summary.get('schema_version')!r}"
            )
        if summary.get("benchmark") != "maze" or not summary.get("maze_benchmark"):
            raise NormalizationError("native summary is not marked as maze benchmark")
        if _required_int(summary, "seed") != manifest.seed:
            raise NormalizationError("native maze seed does not match run manifest")
        if summary.get("archive_type") != "grid":
            raise NormalizationError("native maze archive must be grid")
        if summary.get("target_selection") != manifest.treatment.selector.kind:
            raise NormalizationError(
                "native maze target selection does not match treatment manifest"
            )
        condition = _required_condition(summary)
        expected_generator = (
            "llm"
            if condition.startswith("llm_")
            else "random" if condition == "random" else "genetic"
        )
        if manifest.treatment.generator.kind != expected_generator:
            raise NormalizationError(
                "native maze condition does not match generator treatment"
            )
        expected_gate = "filter" if condition.endswith("_filter") else "off"
        if manifest.treatment.gate.kind != expected_gate:
            raise NormalizationError(
                "native maze condition does not match gate treatment"
            )


def _maze_archive_state(
    rows: tuple[dict[str, Any], ...],
    *,
    capacity: int,
) -> ArchiveState:
    cells: set[tuple[int, int]] = set()
    fitnesses: list[float] = []
    for row in rows:
        if row.get("schema_version") != "maze-1.0":
            raise NormalizationError(
                f"unsupported maze archive schema {row.get('schema_version')!r}"
            )
        raw_bin = row.get("bin")
        if not isinstance(raw_bin, list | tuple) or len(raw_bin) != 2:
            raise NormalizationError("maze archive bin must have two coordinates")
        bin_ij = (
            _required_int({"value": raw_bin[0]}, "value"),
            _required_int({"value": raw_bin[1]}, "value"),
        )
        if bin_ij in cells:
            raise NormalizationError(f"duplicate maze archive bin {bin_ij!r}")
        cells.add(bin_ij)
        fitness = _optional_float(row.get("fitness"))
        if fitness is None:
            raise NormalizationError("maze archive fitness must be numeric")
        fitnesses.append(fitness)
    return archive_state_from_fitnesses(fitnesses, capacity=capacity)


def _validate_terminal_archive(
    summary: Mapping[str, Any],
    state: ArchiveState,
) -> None:
    if _required_int(summary, "filled_cells") != state.occupied_cells:
        raise NormalizationError("maze filled_cells disagrees with archive")
    assert_close(state.coverage, summary.get("coverage"), label="maze coverage")
    assert_close(
        state.raw_qd_score,
        summary.get("qd_score"),
        label="maze QD-score",
    )
    assert_close(
        state.occupied_mean_quality,
        summary.get("mean_best_fitness"),
        label="maze mean fitness",
    )


def _validate_trace_terminal(
    row: Mapping[str, Any],
    state: ArchiveState,
    summary: Mapping[str, Any],
) -> None:
    if _required_int(row, "filled_cells") != state.occupied_cells:
        raise NormalizationError("maze final trace occupancy disagrees with archive")
    if _required_int(row, "proposals") != _required_int(summary, "proposals"):
        raise NormalizationError("maze final trace proposals disagree with summary")
    if _required_int(row, "evaluations") != _required_int(summary, "evaluations"):
        raise NormalizationError("maze final trace evaluations disagree with summary")
    assert_close(state.coverage, row.get("coverage"), label="maze trace coverage")
    assert_close(
        state.raw_qd_score,
        row.get("qd_score"),
        label="maze trace QD-score",
    )


def _terminal_counters(
    summary: Mapping[str, Any],
    *,
    events: tuple[ProposalEvent, ...] = (),
) -> tuple[BudgetCounters, dict[BudgetAxis, SourceCompleteness]]:
    proposals = _required_int(summary, "proposals")
    evaluations = _required_int(summary, "evaluations")
    skipped = _required_int(summary, "skipped")
    if evaluations + skipped != proposals:
        raise NormalizationError(
            "maze proposals must equal evaluations plus skipped slots"
        )
    llm_enabled = bool(summary.get("llm_enabled"))
    # When LLM is disabled, leave counters null so completeness is unavailable
    # rather than marking synthetic zeros as observed.
    llm_attempts = _optional_int(summary.get("llm_calls")) if llm_enabled else None
    evaluated_events = tuple(event for event in events if event.evaluation.attempted)
    evaluator_seconds = (
        sum(
            float(event.resources.evaluator_seconds)
            for event in evaluated_events
            if event.resources.evaluator_seconds is not None
        )
        if evaluated_events
        and all(
            event.resources.evaluator_seconds is not None for event in evaluated_events
        )
        else None
    )
    counters = BudgetCounters(
        proposal_slots=proposals,
        # Skipped slots consumed a proposal budget but were never valid/evaluated.
        valid_proposals=evaluations,
        evaluator_attempts=evaluations,
        evaluator_completions=evaluations,
        llm_attempts=llm_attempts,
        llm_completions=None,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        evaluator_seconds=evaluator_seconds,
        llm_latency_seconds=None,
        wall_seconds=_optional_float(summary.get("elapsed_seconds")),
        monetary_cost=None,
    )
    observed = "observed"
    derived = "derived"
    unavailable = "unavailable"
    completeness: dict[BudgetAxis, SourceCompleteness] = {
        "proposal": observed,
        "valid_proposal": derived,
        "evaluation": observed,
        "llm_call_attempted": (observed if llm_attempts is not None else unavailable),
        "llm_call_completed": unavailable,
        "prompt_token": unavailable,
        "completion_token": unavailable,
        "token": unavailable,
        "evaluator_wall_time": (
            observed if evaluator_seconds is not None else unavailable
        ),
        "llm_latency": unavailable,
        "wall_time": (observed if counters.wall_seconds is not None else unavailable),
        "monetary": unavailable,
    }
    return counters, completeness


def _validate_prospective_events(
    events: tuple[ProposalEvent, ...],
    summary: Mapping[str, Any],
    final_state: ArchiveState,
) -> None:
    if not events:
        raise NormalizationError("prospective maze event log must not be empty")
    if len(events) != _required_int(summary, "proposals"):
        raise NormalizationError(
            "prospective maze event proposals disagree with native summary"
        )
    evaluations = sum(event.evaluation.completed for event in events)
    skipped = sum(not event.evaluation.attempted for event in events)
    if evaluations != _required_int(summary, "evaluations"):
        raise NormalizationError(
            "prospective maze event evaluations disagree with native summary"
        )
    if skipped != _required_int(summary, "skipped"):
        raise NormalizationError(
            "prospective maze skipped slots disagree with native summary"
        )
    terminal = events[-1].after
    if terminal.occupied_cells != final_state.occupied_cells:
        raise NormalizationError(
            "prospective maze terminal occupancy disagrees with native archive"
        )
    assert_close(
        terminal.raw_qd_score,
        final_state.raw_qd_score,
        label="prospective maze terminal QD-score",
    )


def _archive_completeness() -> dict[ArchiveMetric, SourceCompleteness]:
    return {
        "coverage": "observed",
        "raw_qd_score": "observed",
        "normalized_qd_score": "derived",
        "maximum_elite_quality": "derived",
        "occupied_mean_quality": "observed",
    }


def _required_condition(summary: Mapping[str, Any]) -> str:
    value = summary.get("condition")
    if not isinstance(value, str) or not value:
        raise NormalizationError("native maze condition must be a non-empty string")
    return value

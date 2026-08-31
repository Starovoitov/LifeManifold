"""Read-only normalizer for native cellular-automata MAP-Elites runs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

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
    terminal_checkpoint,
)
from worldspace.attribution.adapters.io import (
    ArtifactSource,
    build_artifact_manifest,
    existing_sources,
    read_json_object,
    read_jsonl_objects,
)
from worldspace.attribution.capabilities import ca_capabilities
from worldspace.attribution.hashing import canonical_sha256
from worldspace.attribution.manifest import BudgetAxis, RunManifest
from worldspace.attribution.records import (
    ArchiveMetric,
    ArchiveState,
    BudgetCounters,
    ProposalEvent,
    RunSummary,
    SourceCompleteness,
)
from worldspace.illuminators.archive import (
    merge_archives,
    load_and_collapse_jsonl,
)
from worldspace.illuminators.archive_protocol import ArchiveProtocol

SUMMARY_FILENAME = "nightly_run_summary.json"
ARCHIVE_FILENAME = "map_elites_archive.jsonl"
TRACE_FILENAME = "archive_trace.jsonl"
PROPOSAL_FILENAME = "proposal_log.jsonl"
LLM_CALL_FILENAME = "llm_call_log.jsonl"


class CaNormalizationAdapter:
    """Normalize existing CA artifacts without modifying them."""

    def capabilities(self):
        return ca_capabilities()

    def normalize(
        self,
        manifest: RunManifest,
        inputs: NativeRunInputs,
    ) -> NormalizedRunBundle:
        self._validate_manifest(manifest)
        summary_path = inputs.path(SUMMARY_FILENAME)
        archive_path = inputs.path(ARCHIVE_FILENAME)
        summary = read_json_object(summary_path)
        self._validate_summary_identity(manifest, summary)

        archive_type = _required_str(summary, "archive_type")
        resolution = _required_int(summary, "grid_resolution")
        capacity = _required_int(summary, "n_cells")
        centroids_path = inputs.centroids_path
        final_archive = _load_archive(
            archive_path,
            archive_type=archive_type,
            resolution=resolution,
            centroids_path=centroids_path,
        )
        initial_archive = self._initial_archive(
            manifest,
            inputs,
            archive_type=archive_type,
            resolution=resolution,
            centroids_path=centroids_path,
        )
        initial_cells = _archive_cells(initial_archive)
        if initial_archive is not None:
            final_archive = merge_archives(initial_archive, final_archive)

        final_state = _state_from_archive(final_archive, capacity=capacity)
        _validate_terminal_archive(summary, final_state)

        trace_path = inputs.path(TRACE_FILENAME)
        trace_rows = read_jsonl_objects(trace_path) if trace_path.is_file() else ()
        issues: list[NormalizationIssue] = [
            NormalizationIssue(
                "ca.native_identity_manifest_only",
                "native summary does not carry evaluator or treatment hashes",
            )
        ]
        if not trace_rows:
            issues.append(
                NormalizationIssue(
                    "ca.trace_missing",
                    "archive trace is unavailable; no anytime checkpoints emitted",
                )
            )
        else:
            _validate_trace_terminal(trace_rows[-1], final_state)

        proposal_path = inputs.path(PROPOSAL_FILENAME)
        proposal_rows = (
            read_jsonl_objects(proposal_path) if proposal_path.is_file() else ()
        )
        llm_path = inputs.path(LLM_CALL_FILENAME)
        llm_rows = read_jsonl_objects(llm_path) if llm_path.is_file() else ()
        events, event_issues = _normalize_proposals(
            manifest,
            summary,
            proposal_rows,
            llm_rows,
            capacity=capacity,
            resolution=resolution,
            initial_cells=initial_cells,
            final_state=final_state,
        )
        issues.extend(event_issues)
        if bool(summary.get("llm_enabled")) and not llm_rows:
            issues.append(
                NormalizationIssue(
                    "ca.llm_usage_missing",
                    "LLM call, token, and latency usage is unavailable",
                )
            )
        counters, counter_completeness = _terminal_counters(
            summary,
            proposal_rows=proposal_rows,
            events=events,
            llm_rows=llm_rows,
        )
        checkpoints = (
            checkpoints_from_trace(
                trace_rows,
                run_id=manifest.run_id,
                capacity=capacity,
            )
            if trace_rows
            else (
                terminal_checkpoint(
                    run_id=manifest.run_id,
                    counters=counters,
                    source_completeness=counter_completeness,
                    archive=final_state,
                ),
            )
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
            event_completeness="partial" if events else "summary_only",
            final_counters=counters,
            counter_completeness=counter_completeness,
            final_archive=final_state,
            archive_metric_completeness=_derived_archive_completeness(),
            completed=True,
            failure_reason=None,
        )
        artifacts = build_artifact_manifest(
            manifest,
            existing_sources(
                (
                    (
                        "native_summary",
                        ArtifactSource(
                            summary_path, str(summary.get("schema_version"))
                        ),
                    ),
                    (
                        "native_archive",
                        ArtifactSource(
                            archive_path, str(summary.get("schema_version"))
                        ),
                    ),
                    (
                        "native_trace",
                        ArtifactSource(trace_path, None),
                    ),
                    (
                        "native_proposal_log",
                        ArtifactSource(proposal_path, "2.0"),
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
                        (
                            "native_initial_archive",
                            ArtifactSource(inputs.initial_archive_path, None),
                        )
                        if inputs.initial_archive_path is not None
                        else (
                            "native_initial_archive",
                            ArtifactSource(
                                inputs.path("__absent_initial_archive__"), None
                            ),
                        )
                    ),
                    (
                        (
                            "native_cvt_centroids",
                            ArtifactSource(inputs.centroids_path, None),
                        )
                        if inputs.centroids_path is not None
                        else (
                            "native_cvt_centroids",
                            ArtifactSource(
                                inputs.path("__absent_cvt_centroids__"), None
                            ),
                        )
                    ),
                )
            ),
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
        capabilities = ca_capabilities()
        if manifest.domain_id != capabilities.domain_id:
            raise NormalizationError(
                f"CA adapter cannot normalize domain {manifest.domain_id!r}"
            )
        if (
            manifest.adapter_id != capabilities.adapter_id
            or manifest.adapter_version != capabilities.adapter_version
        ):
            raise NormalizationError(
                "run manifest does not identify the current CA adapter"
            )
        expected_type = manifest.treatment.replacement.parameters.get("archive_type")
        if expected_type not in capabilities.archive_types:
            raise NormalizationError(
                f"unsupported CA archive type in manifest: {expected_type!r}"
            )

    @staticmethod
    def _validate_summary_identity(
        manifest: RunManifest,
        summary: Mapping[str, Any],
    ) -> None:
        if str(summary.get("schema_version")) not in {"1.2", "1.3"}:
            raise NormalizationError(
                f"unsupported CA summary schema {summary.get('schema_version')!r}"
            )
        if _required_int(summary, "seed") != manifest.seed:
            raise NormalizationError("native CA seed does not match run manifest")
        expected_type = manifest.treatment.replacement.parameters.get("archive_type")
        if summary.get("archive_type") != expected_type:
            raise NormalizationError(
                "native CA archive type does not match treatment manifest"
            )

    @staticmethod
    def _initial_archive(
        manifest: RunManifest,
        inputs: NativeRunInputs,
        *,
        archive_type: str,
        resolution: int,
        centroids_path: Path | None,
    ) -> ArchiveProtocol | None:
        initialization = manifest.treatment.initialization.kind
        if initialization == "empty":
            observed_hash = canonical_sha256([])
            if manifest.initial_archive_hash != observed_hash:
                raise NormalizationError(
                    "empty initial archive hash does not match canonical empty hash"
                )
            return None
        if inputs.initial_archive_path is None:
            raise NormalizationError(
                "non-empty CA initialization requires initial_archive_path"
            )
        archive = _load_archive(
            inputs.initial_archive_path,
            archive_type=archive_type,
            resolution=resolution,
            centroids_path=centroids_path,
        )
        observed_hash = _archive_hash(
            archive,
            evaluator_hash=canonical_sha256(manifest.evaluator),
        )
        if observed_hash != manifest.initial_archive_hash:
            raise NormalizationError(
                "initial CA archive hash does not match run manifest"
            )
        return archive


def _load_archive(
    path: Path,
    *,
    archive_type: str,
    resolution: int,
    centroids_path: Path | None,
) -> ArchiveProtocol:
    try:
        return load_and_collapse_jsonl(
            path,
            archive_type=archive_type,
            resolution=resolution,
            centroids_path=centroids_path,
            on_invalid_line="raise",
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise NormalizationError(f"cannot load CA archive {path}: {exc}") from exc


def _archive_cells(archive: ArchiveProtocol | None) -> dict[int, float]:
    if archive is None:
        return {}
    return {
        cell_id: float(elite.fitness)
        for cell_id in range(archive.n_cells)
        if (elite := archive.get_cell(cell_id)) is not None
    }


def _state_from_archive(
    archive: ArchiveProtocol,
    *,
    capacity: int,
) -> ArchiveState:
    if archive.n_cells != capacity:
        raise NormalizationError(
            f"CA archive capacity {archive.n_cells} != summary capacity {capacity}"
        )
    return archive_state_from_fitnesses(
        (
            float(elite.fitness)
            for cell_id in range(archive.n_cells)
            if (elite := archive.get_cell(cell_id)) is not None
        ),
        capacity=capacity,
    )


def _archive_hash(
    archive: ArchiveProtocol,
    *,
    evaluator_hash: str,
) -> str:
    records: list[dict[str, object]] = []
    for cell_id in range(archive.n_cells):
        elite = archive.get_cell(cell_id)
        if elite is None:
            continue
        records.append(
            {
                "cell_id": cell_id,
                "genotype_hash": (
                    canonical_sha256(elite.world_spec.to_canonical_dict())
                    if elite.world_spec is not None
                    else None
                ),
                "measures": elite.measures,
                "fitness": float(elite.fitness),
                "evaluator_hash": evaluator_hash,
            }
        )
    return canonical_sha256(records)


def _validate_terminal_archive(
    summary: Mapping[str, Any],
    state: ArchiveState,
) -> None:
    if _required_int(summary, "filled_cells") != state.occupied_cells:
        raise NormalizationError("CA filled_cells disagrees with collapsed archive")
    assert_close(state.coverage, summary.get("coverage"), label="CA coverage")


def _validate_trace_terminal(
    row: Mapping[str, Any],
    state: ArchiveState,
) -> None:
    if _required_int(row, "filled_cells") != state.occupied_cells:
        raise NormalizationError("CA final trace occupancy disagrees with archive")
    assert_close(state.coverage, row.get("coverage"), label="CA trace coverage")
    assert_close(
        state.raw_qd_score,
        row.get("qd_score"),
        label="CA trace QD-score",
    )


def _terminal_counters(
    summary: Mapping[str, Any],
    *,
    proposal_rows: tuple[dict[str, Any], ...],
    events: tuple[ProposalEvent, ...],
    llm_rows: tuple[dict[str, Any], ...],
) -> tuple[BudgetCounters, dict[BudgetAxis, SourceCompleteness]]:
    evaluations = _required_int(summary, "evaluations")
    proposals = (
        len(proposal_rows) if events and len(proposal_rows) == evaluations else None
    )
    llm_enabled = bool(summary.get("llm_enabled"))
    if llm_enabled:
        llm_attempts = len(llm_rows) if llm_rows else None
        llm_completions = (
            sum(bool(row.get("ok")) for row in llm_rows) if llm_rows else None
        )
        token_counts = _sum_tokens(llm_rows)
        llm_latency = _sum_optional_milliseconds(llm_rows, "latency_ms")
    else:
        # Disabled LLM axes were never collected; keep null so completeness
        # reports unavailable instead of treating synthetic zeros as observed.
        llm_attempts = llm_completions = None
        token_counts = (None, None, None)
        llm_latency = None
    counters = BudgetCounters(
        proposal_slots=proposals,
        valid_proposals=None,
        evaluator_attempts=evaluations,
        evaluator_completions=evaluations,
        llm_attempts=llm_attempts,
        llm_completions=llm_completions,
        prompt_tokens=token_counts[0],
        completion_tokens=token_counts[1],
        total_tokens=token_counts[2],
        evaluator_seconds=_optional_float(summary.get("eval_seconds")),
        llm_latency_seconds=llm_latency,
        wall_seconds=_optional_float(summary.get("elapsed_seconds")),
        monetary_cost=None,
    )
    observed = "observed"
    unavailable = "unavailable"
    completeness: dict[BudgetAxis, SourceCompleteness] = {
        "proposal": observed if proposals is not None else unavailable,
        "valid_proposal": unavailable,
        "evaluation": observed,
        "llm_call_attempted": (observed if llm_attempts is not None else unavailable),
        "llm_call_completed": (
            observed if llm_completions is not None else unavailable
        ),
        "prompt_token": observed if token_counts[0] is not None else unavailable,
        "completion_token": (observed if token_counts[1] is not None else unavailable),
        "token": observed if token_counts[2] is not None else unavailable,
        "evaluator_wall_time": (
            observed if counters.evaluator_seconds is not None else unavailable
        ),
        "llm_latency": observed if llm_latency is not None else unavailable,
        "wall_time": observed if counters.wall_seconds is not None else unavailable,
        "monetary": unavailable,
    }
    return counters, completeness


def _normalize_proposals(
    manifest: RunManifest,
    summary: Mapping[str, Any],
    proposal_rows: tuple[dict[str, Any], ...],
    llm_rows: tuple[dict[str, Any], ...],
    *,
    capacity: int,
    resolution: int,
    initial_cells: dict[int, float],
    final_state: ArchiveState,
) -> tuple[tuple[ProposalEvent, ...], tuple[NormalizationIssue, ...]]:
    if not proposal_rows:
        return (), (
            NormalizationIssue(
                "ca.proposal_log_missing",
                "proposal log unavailable; event completeness is summary_only",
            ),
        )
    evaluations = _required_int(summary, "evaluations")
    gate_kind = manifest.treatment.gate.kind
    if len(proposal_rows) != evaluations or gate_kind not in {"off", "shadow"}:
        return (), (
            NormalizationIssue(
                "ca.proposal_log_partial",
                "proposal log cannot establish every proposal index safely",
            ),
        )
    llm_by_id = {
        str(row["call_id"]): row for row in llm_rows if row.get("call_id") is not None
    }
    ordered = sorted(
        proposal_rows,
        key=lambda row: (
            _required_int(row, "iteration"),
            _required_int(row, "candidate_id"),
        ),
    )
    cells = dict(initial_cells)
    events: list[ProposalEvent] = []
    for proposal_index, row in enumerate(ordered, start=1):
        before = archive_state_from_fitnesses(cells.values(), capacity=capacity)
        realized_bin = _bin(row.get("realized_bin"), label="realized_bin")
        cell_id = realized_bin[0] * resolution + realized_bin[1]
        fitness = _required_float(row, "fitness")
        insertion = _required_str(row, "outcome")
        if insertion not in {"fill_empty", "improve", "occupied_not_better"}:
            raise NormalizationError(f"unsupported CA insertion outcome {insertion!r}")
        incumbent = cells.get(cell_id)
        native_incumbent = _optional_float(row.get("incumbent_fitness"))
        assert_close(incumbent, native_incumbent, label="CA proposal incumbent")
        if insertion == "fill_empty":
            if incumbent is not None:
                raise NormalizationError("fill_empty proposal targets occupied cell")
            delta_qd = fitness
            cells[cell_id] = fitness
        elif insertion == "improve":
            if incumbent is None or fitness <= incumbent:
                raise NormalizationError("invalid strict-improvement proposal")
            delta_qd = fitness - incumbent
            cells[cell_id] = fitness
        else:
            delta_qd = 0.0
        after = archive_state_from_fitnesses(cells.values(), capacity=capacity)
        emitter = _required_str(row, "emitter_type")
        configured = _configured_operator(emitter)
        fallback = "fallback" in emitter
        call_id = row.get("llm_call_id")
        llm_row = llm_by_id.get(str(call_id)) if call_id is not None else None
        event = ProposalEvent.model_validate(
            {
                "run_id": manifest.run_id,
                "study_id": manifest.study_id,
                "arm_id": manifest.arm_id,
                "pair_id": manifest.pair_id,
                "proposal_index": proposal_index,
                "iteration": _required_int(row, "iteration"),
                "slot": _required_int(row, "candidate_id"),
                "timestamp_utc": _required_str(row, "ts_utc"),
                "configured_operator": configured,
                "realized_operator": emitter,
                "target_cell_id": str(_required_int(row, "target_cell_id")),
                "parent_id": row.get("parent_id"),
                "parent_genotype_hash": row.get("parent_world_spec_hash"),
                "candidate_id": (
                    f"{manifest.run_id}:"
                    f"{_required_int(row, 'iteration')}:"
                    f"{_required_int(row, 'candidate_id')}"
                ),
                "candidate_genotype_hash": _required_str(row, "world_spec_hash"),
                "before": before,
                "generation": {
                    "status": "generated",
                    "parse_valid": _parse_validity(row.get("llm_parse_outcome")),
                    "structurally_valid": True,
                    "duplicate": None,
                    "repair_attempts": 0,
                    "repair_outcome": None,
                    "fallback": fallback,
                    "fallback_cause": ("native_emitter_fallback" if fallback else None),
                    "step_metrics": {},
                },
                "gate": {
                    "mode": gate_kind,
                    "decision": "evaluate",
                    "reason": "native evaluated proposal",
                    "policy_version": manifest.treatment.gate.version,
                },
                "evaluation": {
                    "attempted": True,
                    "completed": True,
                    "evaluator_seed": _world_seed(row.get("world_spec")),
                    "fitness": fitness,
                    "descriptors": row.get("measures"),
                    "realized_cell_id": str(cell_id),
                    "incumbent_fitness": incumbent,
                    "insertion": insertion,
                    "delta_qd": delta_qd,
                },
                "resources": _event_resources(
                    manifest,
                    configured_operator=configured,
                    llm_row=llm_row,
                ),
                "after": after,
            }
        )
        events.append(event)
    replayed = events[-1].after
    if replayed.occupied_cells != final_state.occupied_cells:
        raise NormalizationError(
            "replayed CA proposal occupancy disagrees with archive"
        )
    assert_close(
        replayed.raw_qd_score,
        final_state.raw_qd_score,
        label="replayed CA proposal QD-score",
    )
    assert_close(
        replayed.maximum_elite_quality,
        final_state.maximum_elite_quality,
        label="replayed CA proposal maximum fitness",
    )
    return tuple(events), (
        NormalizationIssue(
            "ca.events_partial",
            "proposal events omit unrecorded generation and per-event timing details",
        ),
    )


def _event_resources(
    manifest: RunManifest,
    *,
    configured_operator: str,
    llm_row: Mapping[str, Any] | None,
) -> dict[str, object]:
    if configured_operator != "llm":
        return {
            "llm_calls_attempted": 0,
            "llm_calls_completed": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_latency_seconds": 0.0,
            "evaluator_seconds": None,
            "event_seconds": None,
            "monetary_cost": None,
            "price_table_id": manifest.price_table_id,
        }
    usage = llm_row.get("usage") if llm_row is not None else None
    prompt, completion, total = _usage_tokens(usage)
    latency_ms = (
        _optional_float(llm_row.get("latency_ms")) if llm_row is not None else None
    )
    return {
        "llm_calls_attempted": 1 if llm_row is not None else None,
        "llm_calls_completed": (
            int(bool(llm_row.get("ok"))) if llm_row is not None else None
        ),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "llm_latency_seconds": (
            latency_ms / 1000.0 if latency_ms is not None else None
        ),
        "evaluator_seconds": None,
        "event_seconds": None,
        "monetary_cost": None,
        "price_table_id": manifest.price_table_id,
    }


def _derived_archive_completeness() -> dict[ArchiveMetric, SourceCompleteness]:
    return {
        "coverage": "observed",
        "raw_qd_score": "derived",
        "normalized_qd_score": "derived",
        "maximum_elite_quality": "derived",
        "occupied_mean_quality": "derived",
    }


def _sum_tokens(
    rows: tuple[dict[str, Any], ...],
) -> tuple[int | None, int | None, int | None]:
    if not rows:
        return (None, None, None)
    usages = [_usage_tokens(row.get("usage")) for row in rows]
    if any(any(value is None for value in usage) for usage in usages):
        return (None, None, None)
    return tuple(sum(int(usage[index]) for usage in usages) for index in range(3))  # type: ignore[return-value]


def _usage_tokens(
    usage: object,
) -> tuple[int | None, int | None, int | None]:
    if not isinstance(usage, Mapping):
        return (None, None, None)
    return (
        _optional_int(usage.get("prompt_tokens")),
        _optional_int(usage.get("completion_tokens")),
        _optional_int(usage.get("total_tokens")),
    )


def _sum_optional_milliseconds(
    rows: tuple[dict[str, Any], ...],
    key: str,
) -> float | None:
    if not rows:
        return None
    values = [_optional_float(row.get(key)) for row in rows]
    if any(value is None for value in values):
        return None
    return sum(cast(float, value) for value in values) / 1000.0


def _configured_operator(emitter: str) -> str:
    if emitter.startswith("llm"):
        return "llm"
    if emitter.startswith("genetic"):
        return "genetic"
    if emitter.startswith("random"):
        return "random"
    raise NormalizationError(f"cannot derive configured operator from {emitter!r}")


def _parse_validity(value: object) -> bool | None:
    if value is None:
        return None
    normalized = str(value).lower()
    if normalized in {"ok", "success", "parsed", "valid"}:
        return True
    if normalized in {"failed", "invalid", "parse_error"}:
        return False
    return None


def _world_seed(value: object) -> int | None:
    if not isinstance(value, Mapping):
        return None
    return _optional_int(value.get("seed"))


def _bin(value: object, *, label: str) -> tuple[int, int]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise NormalizationError(f"{label} must contain two integer coordinates")
    return (
        _required_int({"value": value[0]}, "value"),
        _required_int({"value": value[1]}, "value"),
    )


def _required_str(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise NormalizationError(f"native field {key!r} must be a non-empty string")
    return value


def _required_int(row: Mapping[str, Any], key: str) -> int:
    value = _optional_int(row.get(key))
    if value is None:
        raise NormalizationError(f"native field {key!r} must be an integer")
    return value


def _required_float(row: Mapping[str, Any], key: str) -> float:
    value = _optional_float(row.get(key))
    if value is None:
        raise NormalizationError(f"native field {key!r} must be numeric")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise NormalizationError(f"expected integer, got {value!r}")
    try:
        result = int(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise NormalizationError(f"expected integer, got {value!r}") from exc
    if result < 0:
        raise NormalizationError(f"counter cannot be negative, got {result}")
    return result


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise NormalizationError(f"expected numeric value, got {value!r}") from exc

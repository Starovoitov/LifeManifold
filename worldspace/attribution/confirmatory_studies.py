"""Confirmatory study builders (emit jobs; never launch)."""

from __future__ import annotations

from typing import Any, Literal

from worldspace.attribution.capabilities import (
    nas201_capabilities,
    pcg_sokoban_capabilities,
)
from worldspace.attribution.hashing import canonical_sha256
from worldspace.attribution.job_builder import (
    InitialArchiveRef,
    JobBuildContext,
    build_factorial_job_plan,
)
from worldspace.attribution.manifest import StudyManifest

DomainId = Literal["nas201", "pcg_sokoban"]

CONFIRMATORY_SEEDS = tuple(range(301001, 301021))
CONFIRMATORY_PROTOCOL_ID = "ca-confirmatory-v0"
# Placeholder until the protocol hash is stamped; builders refuse launch.
PROTOCOL_HASH_PLACEHOLDER = "0" * 64


def _component(kind: str, **parameters: Any) -> dict[str, Any]:
    cleaned = {key: value for key, value in parameters.items() if value is not None}
    return {
        "kind": kind,
        "version": "1",
        "parameters": cleaned,
        "content_hashes": {},
    }


def _budget() -> dict[str, Any]:
    """Matched caps across arms; non-LLM arms simply leave LLM counters at zero."""
    return {
        **_component("matched_exact"),
        "caps": {
            "proposal_slots": 220,
            "valid_proposals": None,
            "evaluator_calls": 220,
            "llm_calls_attempted": 200,
            "llm_calls_completed": 200,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "evaluator_wall_seconds": None,
            "llm_latency_seconds": None,
            "wall_seconds": None,
            "monetary_cost": None,
        },
        "indexing_axes": ["proposal", "evaluation"],
        "stopping_precedence": ["proposal"],
    }


def _arm(
    domain: DomainId,
    *,
    arm_id: str,
    role: str,
    selector: str,
    generator: str,
    prompt_channel: str,
    allocation: str = "static",
    repair: str | None = None,
    reference_arm_id: str | None = None,
    expected_differences: list[str] | None = None,
) -> dict[str, Any]:
    if repair is None:
        repair = "structural_counts" if domain == "pcg_sokoban" else "identity"
    llm = generator == "llm"
    return {
        "arm_id": arm_id,
        "label": arm_id.replace("_", " "),
        "role": role,
        "treatment": {
            "initialization": _component(
                "generated_floor",
                floor_random=20,
                archive_capacity=400 if domain == "nas201" else 100,
            ),
            "selector": _component(selector),
            "generator": _component(generator),
            "prompt_channel": _component(prompt_channel),
            "repair_fallback": _component(repair),
            "gate": _component("off"),
            "replacement": _component(
                "strict_single_elite",
                archive_type="grid",
                strict=True,
            ),
            "allocation": _component(allocation),
            "budget": _budget(),
        },
        "representation": _component(f"{domain}_genotype"),
        "model": (
            _component("gpt-4o-mini", pinned_response="gpt-4o-mini-2024-07-18")
            if llm
            else _component("not_applicable")
        ),
        "evaluator": _component(f"{domain}_evaluator"),
        "reference_arm_id": reference_arm_id,
        "expected_differences": expected_differences or [],
    }


def selector_channel_study(domain: DomainId) -> StudyManifest:
    """Selector × channel 2×2 under matched repair."""
    arms = [
        _arm(
            domain,
            arm_id="A-const-uniform",
            role="baseline",
            selector="uniform_frontier",
            generator="llm",
            prompt_channel="constant",
        ),
        _arm(
            domain,
            arm_id="A-live-uniform",
            role="control",
            selector="uniform_frontier",
            generator="llm",
            prompt_channel="live",
            reference_arm_id="A-const-uniform",
            expected_differences=["prompt_channel"],
        ),
        _arm(
            domain,
            arm_id="A-const-minfit",
            role="control",
            selector="min_fitness_frontier",
            generator="llm",
            prompt_channel="constant",
            reference_arm_id="A-const-uniform",
            expected_differences=["selector"],
        ),
        _arm(
            domain,
            arm_id="A-live-minfit",
            role="focal",
            selector="min_fitness_frontier",
            generator="llm",
            prompt_channel="live",
            reference_arm_id="A-const-uniform",
            expected_differences=["selector", "prompt_channel"],
        ),
    ]
    return _study(
        domain,
        study_suffix="selector-channel",
        arms=arms,
        estimands=[
            {
                "estimand_id": "channel_x_selector",
                "endpoint": "qd_score_anytime_auc",
                "form": "interaction",
                "budget_axis": "proposal",
                "treatment_arm_ids": ["A-live-minfit", "A-live-uniform"],
                "control_arm_ids": ["A-const-minfit", "A-const-uniform"],
                "paired_by": ["seed", "domain_instance_id"],
                "alternative": "two_sided",
                "margin": None,
                "interaction_formula": (
                    "(A-live-minfit - A-const-minfit) - "
                    "(A-live-uniform - A-const-uniform)"
                ),
                "confirmatory_family": "F-A",
                "multiplicity_rule": "holm",
                "missing_policy": "complete_pairs_only",
                "minimum_complete_pairs": 20,
            }
        ],
    )


def generator_study(domain: DomainId) -> StudyManifest:
    """Generator contrast; B-llm matches A-const-uniform treatment."""
    arms = [
        _arm(
            domain,
            arm_id="B-random",
            role="baseline",
            selector="uniform_frontier",
            generator="random",
            prompt_channel="not_applicable",
        ),
        _arm(
            domain,
            arm_id="B-genetic",
            role="control",
            selector="uniform_frontier",
            generator="genetic",
            prompt_channel="not_applicable",
            reference_arm_id="B-random",
            expected_differences=["generator"],
        ),
        _arm(
            domain,
            arm_id="B-llm",
            role="focal",
            selector="uniform_frontier",
            generator="llm",
            prompt_channel="constant",
            reference_arm_id="B-random",
            expected_differences=["generator", "prompt_channel", "model"],
        ),
    ]
    return _study(
        domain,
        study_suffix="generator",
        arms=arms,
        estimands=[
            {
                "estimand_id": "generator",
                "endpoint": "qd_score_anytime_auc",
                "form": "anytime_auc",
                "budget_axis": "proposal",
                "treatment_arm_ids": ["B-llm", "B-genetic"],
                "control_arm_ids": ["B-random"],
                "paired_by": ["seed", "domain_instance_id"],
                "alternative": "two_sided",
                "margin": None,
                "interaction_formula": None,
                "confirmatory_family": "F-B",
                "multiplicity_rule": "holm",
                "missing_policy": "complete_pairs_only",
                "minimum_complete_pairs": 20,
            }
        ],
    )


def allocation_study(domain: DomainId) -> StudyManifest:
    """Allocation: state_aware_median vs static."""
    arms = [
        _arm(
            domain,
            arm_id="C-static",
            role="baseline",
            selector="uniform_frontier",
            generator="llm",
            prompt_channel="constant",
            allocation="static",
        ),
        _arm(
            domain,
            arm_id="C-state-aware-median",
            role="focal",
            selector="uniform_frontier",
            generator="llm",
            prompt_channel="constant",
            allocation="state_aware_median",
            reference_arm_id="C-static",
            expected_differences=["allocation"],
        ),
    ]
    return _study(
        domain,
        study_suffix="allocation",
        arms=arms,
        estimands=[
            {
                "estimand_id": "allocation",
                "endpoint": "qd_score_anytime_auc",
                "form": "anytime_auc",
                "budget_axis": "proposal",
                "treatment_arm_ids": ["C-state-aware-median"],
                "control_arm_ids": ["C-static"],
                "paired_by": ["seed", "domain_instance_id"],
                "alternative": "two_sided",
                "margin": None,
                "interaction_formula": None,
                "confirmatory_family": "F-C",
                "multiplicity_rule": "holm",
                "missing_policy": "complete_pairs_only",
                "minimum_complete_pairs": 20,
            }
        ],
    )


def confirmatory_job_context(
    study: StudyManifest, *, output_root: str
) -> JobBuildContext:
    empty_hash = canonical_sha256([])
    archives = {
        instance_id: InitialArchiveRef(
            archive_id=f"generated-floor-{instance_id}",
            archive_hash=empty_hash,
        )
        for instance_id in study.replication.domain_instance_ids
    }
    archive_name = (
        "nas201_archive.jsonl" if study.domain_id == "nas201" else "pcg_archive.jsonl"
    )
    return JobBuildContext(
        output_root=output_root,
        initial_archives=archives,
        dependency_hashes={"lock": empty_hash},
        expected_artifacts=(
            "nightly_run_summary.json",
            archive_name,
            "archive_trace.jsonl",
        ),
        unit_monetary_cost=0.0,
    )


def build_confirmatory_job_plan(study: StudyManifest, *, output_root: str):
    caps = (
        nas201_capabilities()
        if study.domain_id == "nas201"
        else pcg_sokoban_capabilities()
    )
    if study.domain_id not in {"nas201", "pcg_sokoban"}:
        raise ValueError(f"confirmatory builder rejects domain {study.domain_id!r}")
    return build_factorial_job_plan(
        study,
        caps,
        confirmatory_job_context(study, output_root=output_root),
    )


def _study(
    domain: DomainId,
    *,
    study_suffix: str,
    arms: list[dict[str, Any]],
    estimands: list[dict[str, Any]],
) -> StudyManifest:
    caps = nas201_capabilities() if domain == "nas201" else pcg_sokoban_capabilities()
    payload = {
        "schema_version": "attribution-1.0",
        "study_id": f"confirmatory-{domain}-{study_suffix}",
        "programme_id": "controlled-attribution",
        "protocol_id": CONFIRMATORY_PROTOCOL_ID,
        "protocol_hash": PROTOCOL_HASH_PLACEHOLDER,
        "evidence_tier": "confirmatory",
        "domain_id": domain,
        "domain_version": "confirmatory-draft-1",
        "adapter_id": caps.adapter_id,
        "adapter_version": caps.adapter_version,
        "task_instance_set": [f"{domain}-default"],
        "estimands": estimands,
        "arms": arms,
        "replication": {
            "seeds": list(CONFIRMATORY_SEEDS),
            "domain_instance_ids": [f"{domain}-default"],
            "api_block_ids": [],
            "paired_by": ["seed", "domain_instance_id"],
        },
        "cost_policy": {
            "currency": "USD",
            "price_table_id": "confirmatory-draft-unset",
            "price_table_hash": PROTOCOL_HASH_PLACEHOLDER,
            "approved_total_cost": 0.0,
            "missing_usage_policy": "reject",
        },
        "failure_policy": {
            "generation_failure": "consume_and_continue",
            "evaluator_failure": "abort_run",
            "maximum_evaluator_retries": 0,
            "incomplete_run": "exclude",
        },
        "privacy_policy": {
            "raw_prompts": "discard",
            "raw_responses": "discard",
            "publish_sanitized_events": True,
        },
    }
    return StudyManifest.model_validate(payload)

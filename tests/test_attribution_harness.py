from __future__ import annotations

import copy
import unittest
from typing import Any, Literal

from pydantic import ValidationError

from worldspace.attribution import (
    AdapterCapabilities,
    ArchiveState,
    AttributionAdmissionError,
    BudgetCounters,
    ProposalEvent,
    RunManifest,
    RunSummary,
    StudyManifest,
    TreatmentVector,
    admit_analysis_cohort,
    arm_treatment_hash,
    canonical_sha256,
    freeze_run_manifest,
    study_manifest_hash,
    validate_study_capabilities,
)
from worldspace.attribution.schemas import attribution_schema_bundle

HASH_A = "a" * 64
HASH_B = "b" * 64
DOMAINS = ("ca", "maze", "dungeon", "sphere")
EvidenceTier = Literal["feasibility", "design_pilot", "confirmatory", "robustness"]


def _component(kind: str, **parameters: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "version": "1",
        "parameters": parameters,
        "content_hashes": {},
    }


def _budget() -> dict[str, Any]:
    return {
        **_component("matched_exact"),
        "caps": {
            "proposal_slots": 100,
            "valid_proposals": None,
            "evaluator_calls": 100,
            "llm_calls_attempted": 0,
            "llm_calls_completed": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "evaluator_wall_seconds": None,
            "llm_latency_seconds": None,
            "wall_seconds": None,
            "monetary_cost": 0.0,
        },
        "indexing_axes": ["proposal", "evaluation"],
        "stopping_precedence": ["proposal"],
    }


def _arm(
    domain: str,
    *,
    arm_id: str,
    role: str,
    selector: str = "uniform_frontier",
    generator: str = "genetic",
    reference_arm_id: str | None = None,
    expected_differences: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "label": arm_id.replace("_", " "),
        "role": role,
        "treatment": {
            "initialization": _component("empty", archive_capacity=100),
            "selector": _component(selector),
            "generator": _component(generator, mutation_scale=0.1),
            "prompt_channel": _component("not_applicable"),
            "repair_fallback": _component("identity"),
            "gate": _component("off"),
            "replacement": _component(
                "strict_single_elite",
                archive_type="grid",
                strict=True,
            ),
            "allocation": _component("static"),
            "budget": _budget(),
        },
        "representation": _component(f"{domain}_genotype"),
        "model": _component("not_applicable"),
        "evaluator": _component(f"{domain}_evaluator"),
        "reference_arm_id": reference_arm_id,
        "expected_differences": expected_differences or [],
    }


def _study_payload(domain: str) -> dict[str, Any]:
    if domain == "dungeon":
        baseline = _arm(
            domain,
            arm_id="random",
            role="baseline",
            generator="random",
        )
        focal = _arm(
            domain,
            arm_id="genetic",
            role="focal",
            generator="genetic",
            reference_arm_id="random",
            expected_differences=["generator"],
        )
    else:
        baseline = _arm(
            domain,
            arm_id="uniform",
            role="baseline",
            selector="uniform_frontier",
        )
        focal = _arm(
            domain,
            arm_id="minfit",
            role="focal",
            selector="min_fitness_frontier",
            reference_arm_id="uniform",
            expected_differences=["selector"],
        )
    return {
        "schema_version": "attribution-1.0",
        "study_id": f"fixture-{domain}",
        "programme_id": "controlled-attribution",
        "protocol_id": "fixture-protocol",
        "protocol_hash": HASH_A,
        "evidence_tier": "feasibility",
        "domain_id": domain,
        "domain_version": "fixture-1",
        "adapter_id": f"{domain}-adapter",
        "adapter_version": "0.1",
        "task_instance_set": [f"{domain}-instance"],
        "estimands": [
            {
                "estimand_id": "primary",
                "endpoint": "normalized_qd_score",
                "form": "terminal",
                "budget_axis": "proposal",
                "treatment_arm_ids": [focal["arm_id"]],
                "control_arm_ids": [baseline["arm_id"]],
                "paired_by": ["seed", "domain_instance_id"],
                "alternative": "greater",
                "margin": None,
                "interaction_formula": None,
                "confirmatory_family": None,
                "multiplicity_rule": None,
                "missing_policy": "complete_pairs_only",
                "minimum_complete_pairs": 2,
            }
        ],
        "arms": [baseline, focal],
        "replication": {
            "seeds": [0, 1],
            "domain_instance_ids": [f"{domain}-instance"],
            "api_block_ids": [],
            "paired_by": ["seed", "domain_instance_id"],
        },
        "cost_policy": {
            "currency": "USD",
            "price_table_id": "fixture-free",
            "price_table_hash": HASH_B,
            "approved_total_cost": 0.0,
            "missing_usage_policy": "report_missing",
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


def _capabilities(domain: str) -> AdapterCapabilities:
    selectors = (
        ("uniform_frontier",)
        if domain == "dungeon"
        else ("uniform_frontier", "min_fitness_frontier")
    )
    return AdapterCapabilities(
        adapter_id=f"{domain}-adapter",
        adapter_version="0.1",
        domain_id=domain,
        initialization_kinds=("empty",),
        selectors=selectors,
        generators=("random", "genetic"),
        prompt_channels=("not_applicable",),
        repair_fallback_kinds=("identity",),
        gate_modes=("off",),
        replacement_kinds=("strict_single_elite",),
        allocation_kinds=("static",),
        budget_axes=("proposal", "evaluation"),
        archive_types=("grid",),
        supports_full_proposal_log=False,
        supports_warm_start=False,
        stochastic_evaluation=domain == "dungeon",
        native_fitness_min=0.0,
        native_fitness_max=100.0 if domain == "sphere" else 1.0,
        empty_cell_fitness=0.0,
    )


def _archive_state(*, occupied: int = 1, raw_qd: float = 0.5) -> dict[str, Any]:
    return {
        "occupied_cells": occupied,
        "capacity": 10,
        "coverage": occupied / 10,
        "raw_qd_score": raw_qd,
        "normalized_qd_score": raw_qd / 10,
        "maximum_elite_quality": 0.5 if occupied else None,
        "occupied_mean_quality": raw_qd / occupied if occupied else None,
    }


def _summary(
    *,
    run_id: str,
    arm_id: str,
    pair_id: str,
    tier: EvidenceTier = "feasibility",
) -> RunSummary:
    return RunSummary(
        run_id=run_id,
        study_id="study",
        arm_id=arm_id,
        pair_id=pair_id,
        evidence_tier=tier,
        domain_id="ca",
        domain_version="fixture-1",
        adapter_id="ca-adapter",
        adapter_version="0.1",
        evaluator_hash=HASH_A,
        seed=0,
        domain_instance_id="ca-instance",
        initial_archive_hash=HASH_A,
        protocol_hash=HASH_A,
        study_manifest_hash=HASH_A,
        run_manifest_hash=HASH_A,
        treatment_hash=HASH_A if arm_id == "control" else HASH_B,
        event_completeness="full",
        final_counters=BudgetCounters(
            proposal_slots=100,
            valid_proposals=100,
            evaluator_attempts=100,
            evaluator_completions=100,
            llm_attempts=0,
            llm_completions=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            evaluator_seconds=1.0,
            llm_latency_seconds=0.0,
            wall_seconds=1.0,
            monetary_cost=0.0,
        ),
        counter_completeness={
            "proposal": "observed",
            "valid_proposal": "observed",
            "evaluation": "observed",
            "llm_call_attempted": "observed",
            "llm_call_completed": "observed",
            "prompt_token": "observed",
            "completion_token": "observed",
            "token": "observed",
            "evaluator_wall_time": "observed",
            "llm_latency": "observed",
            "wall_time": "observed",
            "monetary": "observed",
        },
        final_archive=ArchiveState.model_validate(_archive_state()),
        archive_metric_completeness={
            "coverage": "observed",
            "raw_qd_score": "observed",
            "normalized_qd_score": "derived",
            "maximum_elite_quality": "observed",
            "occupied_mean_quality": "observed",
        },
        completed=True,
        failure_reason=None,
    )


class TestAttributionHarness(unittest.TestCase):
    def test_four_domain_fixture_manifests_validate(self) -> None:
        for domain in DOMAINS:
            with self.subTest(domain=domain):
                study = StudyManifest.model_validate(_study_payload(domain))
                self.assertEqual(study.domain_id, domain)
                self.assertEqual(len(study.arms), 2)
                self.assertEqual(len(study_manifest_hash(study)), 64)
                self.assertEqual(
                    validate_study_capabilities(study, _capabilities(domain)),
                    (),
                )

    def test_canonical_hash_is_mapping_order_independent(self) -> None:
        left = {"b": [2, 3], "a": {"z": 1}}
        right = {"a": {"z": 1}, "b": [2, 3]}
        self.assertEqual(canonical_sha256(left), canonical_sha256(right))

    def test_missing_treatment_component_is_rejected(self) -> None:
        payload = _study_payload("ca")
        del payload["arms"][0]["treatment"]["repair_fallback"]
        with self.assertRaisesRegex(ValidationError, "repair_fallback"):
            StudyManifest.model_validate(payload)

    def test_declared_component_diff_must_match_actual_diff(self) -> None:
        payload = _study_payload("maze")
        payload["arms"][1]["expected_differences"] = ["generator"]
        with self.assertRaisesRegex(ValidationError, "expected differences"):
            StudyManifest.model_validate(payload)

    def test_adapter_cannot_silently_substitute_selector(self) -> None:
        payload = _study_payload("dungeon")
        payload["arms"][1]["treatment"]["selector"] = _component("min_fitness_frontier")
        payload["arms"][1]["expected_differences"] = ["generator", "selector"]
        study = StudyManifest.model_validate(payload)
        issues = validate_study_capabilities(study, _capabilities("dungeon"))
        self.assertEqual({item.code for item in issues}, {"capability.selector"})

    def test_run_manifest_hash_detects_tampering(self) -> None:
        study = StudyManifest.model_validate(_study_payload("ca"))
        arm = study.arms[1]
        core = {
            "run_id": "fixture-ca-minfit-seed-0",
            "study_id": study.study_id,
            "arm_id": arm.arm_id,
            "pair_id": "fixture-ca-seed-0",
            "block_id": "local",
            "evidence_tier": study.evidence_tier,
            "protocol_id": study.protocol_id,
            "protocol_hash": study.protocol_hash,
            "domain_id": study.domain_id,
            "domain_version": study.domain_version,
            "adapter_id": study.adapter_id,
            "adapter_version": study.adapter_version,
            "seed": 0,
            "domain_instance_id": "ca-instance",
            "initial_archive_id": "empty-ca",
            "initial_archive_hash": HASH_A,
            "treatment": arm.treatment,
            "representation": arm.representation,
            "model": arm.model,
            "evaluator": arm.evaluator,
            "treatment_hash": arm_treatment_hash(arm),
            "study_manifest_hash": study_manifest_hash(study),
            "currency": "USD",
            "price_table_id": "fixture-free",
            "price_table_hash": HASH_B,
            "dependency_hashes": {"lock": HASH_B},
            "output_paths": {"run_dir": "output/fixture-ca/minfit/seed-0"},
            "expected_artifacts": ["run_summary.json"],
        }
        frozen = freeze_run_manifest(core)
        self.assertEqual(len(frozen.run_manifest_hash), 64)
        self.assertEqual(
            RunManifest.model_validate(frozen.model_dump(mode="json")),
            frozen,
        )
        tampered = frozen.model_dump(mode="json")
        tampered["seed"] = 99
        with self.assertRaisesRegex(ValidationError, "run_manifest_hash mismatch"):
            RunManifest.model_validate(tampered)

    def test_schema_bundle_contains_all_phase_1_records(self) -> None:
        bundle = attribution_schema_bundle()
        self.assertEqual(
            set(bundle),
            {
                "StudyManifest",
                "RunManifest",
                "AdapterCapabilities",
                "ProposalEvent",
                "BudgetCheckpoint",
                "RunSummary",
                "ArtifactManifest",
                "DesignMatrix",
                "JobPlan",
            },
        )
        self.assertIn("treatment", bundle["RunManifest"]["properties"])

    def test_proposal_event_checks_exact_qd_transition(self) -> None:
        payload = {
            "run_id": "run",
            "study_id": "study",
            "arm_id": "arm",
            "pair_id": "pair",
            "proposal_index": 1,
            "iteration": 0,
            "slot": 0,
            "timestamp_utc": "2026-08-31T00:00:00Z",
            "configured_operator": "genetic",
            "realized_operator": "genetic",
            "target_cell_id": "1",
            "parent_id": None,
            "parent_genotype_hash": None,
            "candidate_id": "candidate",
            "candidate_genotype_hash": HASH_A,
            "before": _archive_state(),
            "generation": {
                "status": "generated",
                "parse_valid": None,
                "structurally_valid": True,
                "duplicate": False,
                "repair_attempts": 0,
                "repair_outcome": None,
                "fallback": False,
                "fallback_cause": None,
                "step_metrics": {},
            },
            "gate": {
                "mode": "off",
                "decision": "evaluate",
                "reason": "gate disabled",
                "policy_version": "1",
            },
            "evaluation": {
                "attempted": True,
                "completed": True,
                "evaluator_seed": 0,
                "fitness": 0.75,
                "descriptors": {"x": 0.1, "y": 0.2},
                "realized_cell_id": "2",
                "incumbent_fitness": None,
                "insertion": "fill_empty",
                "delta_qd": 0.75,
            },
            "resources": {
                "llm_calls_attempted": 0,
                "llm_calls_completed": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "llm_latency_seconds": 0.0,
                "evaluator_seconds": 0.01,
                "event_seconds": 0.01,
                "monetary_cost": 0.0,
                "price_table_id": "fixture-free",
            },
            "after": _archive_state(occupied=2, raw_qd=1.25),
        }
        event = ProposalEvent.model_validate(payload)
        self.assertEqual(event.evaluation.delta_qd, 0.75)
        payload["evaluation"]["delta_qd"] = 0.5
        with self.assertRaisesRegex(ValidationError, "delta_qd mismatch"):
            ProposalEvent.model_validate(payload)

    def test_analysis_admission_rejects_mixed_tiers(self) -> None:
        rows = [
            _summary(run_id="a", arm_id="control", pair_id="pair"),
            _summary(
                run_id="b",
                arm_id="treatment",
                pair_id="pair",
                tier="confirmatory",
            ),
        ]
        with self.assertRaises(AttributionAdmissionError) as caught:
            admit_analysis_cohort(
                rows,
                expected_arm_ids=("control", "treatment"),
                minimum_complete_pairs=1,
            )
        self.assertIn("cohort.mixed_evidence_tier", str(caught.exception))

    def test_analysis_admission_requires_complete_pairs(self) -> None:
        rows = [_summary(run_id="a", arm_id="control", pair_id="pair")]
        with self.assertRaises(AttributionAdmissionError) as caught:
            admit_analysis_cohort(
                rows,
                expected_arm_ids=("control", "treatment"),
                minimum_complete_pairs=1,
            )
        self.assertIn("cohort.incomplete_pair", str(caught.exception))

    def test_analysis_admission_checks_pairing_identity(self) -> None:
        control = _summary(run_id="a", arm_id="control", pair_id="pair")
        treatment_payload = _summary(
            run_id="b",
            arm_id="treatment",
            pair_id="pair",
        ).model_dump(mode="json")
        treatment_payload["initial_archive_hash"] = HASH_B
        treatment = RunSummary.model_validate(treatment_payload)
        with self.assertRaises(AttributionAdmissionError) as caught:
            admit_analysis_cohort(
                (control, treatment),
                expected_arm_ids=("control", "treatment"),
                minimum_complete_pairs=1,
            )
        self.assertIn("cohort.unpaired_initial_archive_hash", str(caught.exception))

    def test_budget_counter_relations_are_validated(self) -> None:
        payload = BudgetCounters(
            proposal_slots=10,
            valid_proposals=10,
            evaluator_attempts=10,
            evaluator_completions=10,
            llm_attempts=0,
            llm_completions=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            evaluator_seconds=1.0,
            llm_latency_seconds=0.0,
            wall_seconds=1.0,
            monetary_cost=0.0,
        ).model_dump()
        invalid = copy.deepcopy(payload)
        invalid["evaluator_completions"] = 11
        with self.assertRaisesRegex(ValidationError, "evaluator completions"):
            BudgetCounters.model_validate(invalid)

    def test_treatment_vector_rejects_extra_components(self) -> None:
        payload = _arm("ca", arm_id="a", role="control")["treatment"]
        payload["secret_selector"] = _component("hidden")
        with self.assertRaisesRegex(ValidationError, "secret_selector"):
            TreatmentVector.model_validate(payload)


if __name__ == "__main__":
    unittest.main()

"""Tests for factorial job builder and design-matrix analysis admission."""

from __future__ import annotations

import copy
import unittest
from typing import Any

from worldspace.attribution import (
    AdapterCapabilities,
    ArchiveState,
    AttributionAdmissionError,
    BudgetCounters,
    InitialArchiveRef,
    JobBuildContext,
    RunSummary,
    StudyManifest,
    admit_design_matrix_cohort,
    arm_treatment_hash,
    build_factorial_job_plan,
    descriptive_contrast,
    study_manifest_hash,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def _component(kind: str, **parameters: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "version": "1",
        "parameters": parameters,
        "content_hashes": {},
    }


def _budget(*, monetary_cost: float | None = 0.0) -> dict[str, Any]:
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
            "monetary_cost": monetary_cost,
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
    generator_version: str = "1",
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
            "generator": {
                "kind": generator,
                "version": generator_version,
                "parameters": {"mutation_scale": 0.1},
                "content_hashes": {},
            },
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


def _block_a_study() -> dict[str, Any]:
    """Selector factorial (uniform vs min-fitness) — Block A."""
    baseline = _arm("ca", arm_id="uniform", role="baseline")
    focal = _arm(
        "ca",
        arm_id="minfit",
        role="focal",
        selector="min_fitness_frontier",
        reference_arm_id="uniform",
        expected_differences=["selector"],
    )
    return {
        "schema_version": "attribution-1.0",
        "study_id": "block-a-selector",
        "programme_id": "controlled-attribution",
        "protocol_id": "block-a-protocol",
        "protocol_hash": HASH_A,
        "evidence_tier": "feasibility",
        "domain_id": "ca",
        "domain_version": "fixture-1",
        "adapter_id": "ca-adapter",
        "adapter_version": "0.1",
        "task_instance_set": ["ca-instance"],
        "estimands": [
            {
                "estimand_id": "selector_main",
                "endpoint": "normalized_qd_score",
                "form": "terminal",
                "budget_axis": "proposal",
                "treatment_arm_ids": ["minfit"],
                "control_arm_ids": ["uniform"],
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
            "domain_instance_ids": ["ca-instance"],
            "api_block_ids": [],
            "paired_by": ["seed", "domain_instance_id"],
        },
        "cost_policy": {
            "currency": "USD",
            "price_table_id": "fixture-free",
            "price_table_hash": HASH_B,
            "approved_total_cost": 10.0,
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


def _block_b_study() -> dict[str, Any]:
    """Generator calibration contrast — Block B."""
    baseline = _arm(
        "maze",
        arm_id="genetic_uncalibrated",
        role="baseline",
        generator="genetic",
        generator_version="uncalibrated",
    )
    focal = _arm(
        "maze",
        arm_id="genetic_calibrated",
        role="focal",
        generator="genetic",
        generator_version="calibrated",
        reference_arm_id="genetic_uncalibrated",
        expected_differences=["generator"],
    )
    return {
        "schema_version": "attribution-1.0",
        "study_id": "block-b-generator",
        "programme_id": "controlled-attribution",
        "protocol_id": "block-b-protocol",
        "protocol_hash": HASH_A,
        "evidence_tier": "feasibility",
        "domain_id": "maze",
        "domain_version": "fixture-1",
        "adapter_id": "maze-adapter",
        "adapter_version": "0.1",
        "task_instance_set": ["maze-instance"],
        "estimands": [
            {
                "estimand_id": "generator_main",
                "endpoint": "normalized_qd_score",
                "form": "terminal",
                "budget_axis": "proposal",
                "treatment_arm_ids": ["genetic_calibrated"],
                "control_arm_ids": ["genetic_uncalibrated"],
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
            "seeds": [10, 11],
            "domain_instance_ids": ["maze-instance"],
            "api_block_ids": ["calendar-1"],
            "paired_by": ["seed", "domain_instance_id"],
        },
        "cost_policy": {
            "currency": "USD",
            "price_table_id": "fixture-free",
            "price_table_hash": HASH_B,
            "approved_total_cost": 5.0,
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
    return AdapterCapabilities(
        adapter_id=f"{domain}-adapter",
        adapter_version="0.1",
        domain_id=domain,
        initialization_kinds=("empty",),
        selectors=("uniform_frontier", "min_fitness_frontier"),
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
        stochastic_evaluation=False,
        native_fitness_min=0.0,
        native_fitness_max=1.0,
        empty_cell_fitness=0.0,
    )


def _context_for(
    study: StudyManifest,
    *,
    unit_cost: float = 0.0,
    seed_api_blocks: dict[int, str] | None = None,
    reserved_pilot_seeds: frozenset[int] = frozenset(),
    reserved_pilot_archives: frozenset[str] = frozenset(),
) -> JobBuildContext:
    archives = {
        instance_id: InitialArchiveRef(
            archive_id=f"empty-{instance_id}",
            archive_hash=HASH_A,
        )
        for instance_id in study.replication.domain_instance_ids
    }
    return JobBuildContext(
        output_root="artifacts/attribution-jobs",
        initial_archives=archives,
        dependency_hashes={"lock": HASH_B},
        expected_artifacts=("run_summary.json", "run_manifest.json"),
        unit_monetary_cost=unit_cost,
        order_seed=7,
        reserved_pilot_seeds=reserved_pilot_seeds,
        reserved_pilot_archive_hashes=reserved_pilot_archives,
        seed_api_blocks=seed_api_blocks,
    )


def _summary_from_cell(
    study: StudyManifest,
    *,
    run_id: str,
    arm_id: str,
    pair_id: str,
    seed: int,
    treatment_hash: str,
    run_manifest_hash: str,
    qd: float,
) -> RunSummary:
    return RunSummary(
        run_id=run_id,
        study_id=study.study_id,
        arm_id=arm_id,
        pair_id=pair_id,
        evidence_tier=study.evidence_tier,
        domain_id=study.domain_id,
        domain_version=study.domain_version,
        adapter_id=study.adapter_id,
        adapter_version=study.adapter_version,
        evaluator_hash=HASH_A,
        seed=seed,
        domain_instance_id=study.replication.domain_instance_ids[0],
        initial_archive_hash=HASH_A,
        protocol_hash=study.protocol_hash,
        study_manifest_hash=study_manifest_hash(study),
        run_manifest_hash=run_manifest_hash,
        treatment_hash=treatment_hash,
        event_completeness="summary_only",
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
        final_archive=ArchiveState(
            occupied_cells=1,
            capacity=10,
            coverage=0.1,
            raw_qd_score=qd,
            normalized_qd_score=qd / 10.0,
            maximum_elite_quality=qd,
            occupied_mean_quality=qd,
        ),
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


class TestFactorialJobBuilder(unittest.TestCase):
    def test_block_a_and_block_b_emit_without_launching(self) -> None:
        block_a = StudyManifest.model_validate(_block_a_study())
        plan_a = build_factorial_job_plan(
            block_a,
            _capabilities("ca"),
            _context_for(block_a),
        )
        self.assertFalse(plan_a.launched)
        self.assertEqual(len(plan_a.runs), 4)
        self.assertEqual(len(plan_a.design.cells), 4)
        self.assertEqual(plan_a.projection.run_count, 4)
        self.assertEqual(plan_a.design.contrasts[0].differing_axes, ("selector",))
        self.assertTrue(plan_a.design.contrasts[0].controlled_axes_match)
        self.assertIn("local", plan_a.execution_order)
        self.assertEqual(len(plan_a.execution_order["local"]), 4)

        block_b = StudyManifest.model_validate(_block_b_study())
        plan_b = build_factorial_job_plan(
            block_b,
            _capabilities("maze"),
            _context_for(
                block_b,
                seed_api_blocks={10: "calendar-1", 11: "calendar-1"},
            ),
        )
        self.assertFalse(plan_b.launched)
        self.assertEqual(len(plan_b.runs), 4)
        self.assertEqual(plan_b.design.contrasts[0].differing_axes, ("generator",))
        self.assertEqual(set(plan_b.execution_order), {"calendar-1"})
        for run in plan_b.runs:
            self.assertTrue(
                run.output_paths["run_dir"].startswith(
                    "artifacts/attribution-jobs/block-b-generator/feasibility/maze/"
                )
            )

    def test_rejects_duplicate_treatment_hash_without_alias(self) -> None:
        payload = _block_a_study()
        payload["arms"][1] = copy.deepcopy(payload["arms"][0])
        payload["arms"][1]["arm_id"] = "uniform_clone"
        payload["arms"][1]["label"] = "uniform clone"
        payload["arms"][1]["role"] = "sensitivity"
        payload["estimands"][0]["treatment_arm_ids"] = ["uniform_clone"]
        study = StudyManifest.model_validate(payload)
        with self.assertRaises(AttributionAdmissionError) as caught:
            build_factorial_job_plan(
                study,
                _capabilities("ca"),
                _context_for(study),
            )
        self.assertIn("builder.duplicate_treatment_hash", str(caught.exception))

    def test_rejects_confirmatory_reuse_of_pilot_seeds(self) -> None:
        payload = _block_a_study()
        payload["evidence_tier"] = "confirmatory"
        study = StudyManifest.model_validate(payload)
        with self.assertRaises(AttributionAdmissionError) as caught:
            build_factorial_job_plan(
                study,
                _capabilities("ca"),
                _context_for(study, reserved_pilot_seeds=frozenset({0})),
            )
        self.assertIn("builder.pilot_seed", str(caught.exception))

    def test_rejects_projected_cost_above_cap(self) -> None:
        study = StudyManifest.model_validate(_block_a_study())
        with self.assertRaises(AttributionAdmissionError) as caught:
            build_factorial_job_plan(
                study,
                _capabilities("ca"),
                _context_for(study, unit_cost=3.0),
            )
        self.assertIn("builder.budget_cap", str(caught.exception))

    def test_design_matrix_analysis_and_descriptive_contrasts(self) -> None:
        study = StudyManifest.model_validate(_block_a_study())
        plan = build_factorial_job_plan(
            study,
            _capabilities("ca"),
            _context_for(study),
        )
        qd_by_arm = {"uniform": 0.4, "minfit": 0.6}
        rows = []
        for cell in plan.design.cells:
            rows.append(
                _summary_from_cell(
                    study,
                    run_id=cell.run_id,
                    arm_id=cell.arm_id,
                    pair_id=cell.pair_id,
                    seed=cell.seed,
                    treatment_hash=cell.treatment_hash,
                    run_manifest_hash=cell.run_manifest_hash,
                    qd=qd_by_arm[cell.arm_id] + 0.01 * cell.seed,
                )
            )
        estimand = study.estimands[0]
        admitted = admit_design_matrix_cohort(
            rows,
            study=study,
            estimand=estimand,
            design=plan.design,
        )
        self.assertEqual(len(admitted), 4)
        contrast = descriptive_contrast(admitted, estimand=estimand)
        self.assertEqual(contrast.complete_pairs, 2)
        self.assertEqual(len(contrast.paired_differences), 2)
        self.assertAlmostEqual(contrast.mean_difference or 0.0, 0.02, places=9)
        means = {item.arm_id: item.mean for item in contrast.cell_means}
        self.assertAlmostEqual(means["uniform"] or 0.0, 0.0405, places=9)
        self.assertAlmostEqual(means["minfit"] or 0.0, 0.0605, places=9)

    def test_analysis_rejects_run_absent_from_design_matrix(self) -> None:
        study = StudyManifest.model_validate(_block_a_study())
        plan = build_factorial_job_plan(
            study,
            _capabilities("ca"),
            _context_for(study),
        )
        cell = plan.design.cells[0]
        row = _summary_from_cell(
            study,
            run_id="undeclared-run",
            arm_id=cell.arm_id,
            pair_id=cell.pair_id,
            seed=cell.seed,
            treatment_hash=cell.treatment_hash,
            run_manifest_hash=cell.run_manifest_hash,
            qd=0.5,
        )
        with self.assertRaises(AttributionAdmissionError) as caught:
            admit_design_matrix_cohort(
                [row],
                study=study,
                estimand=study.estimands[0],
                design=plan.design,
            )
        self.assertIn("analysis.undeclared_run", str(caught.exception))

    def test_analysis_rejects_pilot_rows_in_confirmatory_design(self) -> None:
        payload = _block_a_study()
        payload["evidence_tier"] = "confirmatory"
        study = StudyManifest.model_validate(payload)
        plan = build_factorial_job_plan(
            study,
            _capabilities("ca"),
            _context_for(study),
        )
        cell = plan.design.cells[0]
        row = _summary_from_cell(
            study,
            run_id=cell.run_id,
            arm_id=cell.arm_id,
            pair_id=cell.pair_id,
            seed=cell.seed,
            treatment_hash=cell.treatment_hash,
            run_manifest_hash=cell.run_manifest_hash,
            qd=0.5,
        )
        contaminated = row.model_dump(mode="json")
        contaminated["evidence_tier"] = "design_pilot"
        with self.assertRaises(AttributionAdmissionError) as caught:
            admit_design_matrix_cohort(
                [RunSummary.model_validate(contaminated)],
                study=study,
                estimand=study.estimands[0],
                design=plan.design,
            )
        self.assertIn("analysis.pilot_contamination", str(caught.exception))

    def test_arm_treatment_hashes_stable_across_builder(self) -> None:
        study = StudyManifest.model_validate(_block_a_study())
        plan = build_factorial_job_plan(
            study,
            _capabilities("ca"),
            _context_for(study),
        )
        expected = {arm.arm_id: arm_treatment_hash(arm) for arm in study.arms}
        for run in plan.runs:
            self.assertEqual(run.treatment_hash, expected[run.arm_id])


if __name__ == "__main__":
    unittest.main()

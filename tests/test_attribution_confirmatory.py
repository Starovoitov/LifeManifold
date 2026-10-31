"""Public-domain adapters, loops, live prompts, and confirmatory job emission."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from worldspace.attribution.analysis import (
    _endpoint_value,
    descriptive_contrast,
    family_holm,
    holm_step_down,
)
from worldspace.attribution.adapters import (
    Nas201NormalizationAdapter,
    NativeRunInputs,
    PcgSokobanNormalizationAdapter,
)
from worldspace.attribution.capabilities import current_domain_capabilities
from worldspace.attribution.confirmatory_admission import (
    refuse_contaminated_artifact_paths,
    refuse_missing_llm_usage,
    refuse_mixed_evidence_tier,
    refuse_response_model_mismatch,
)
from worldspace.attribution.confirmatory_studies import (
    allocation_study,
    build_confirmatory_job_plan,
    confirmatory_bin_edges,
    generator_study,
    selector_channel_study,
)
from worldspace.attribution.hashing import canonical_sha256
from worldspace.attribution.manifest import (
    ArmManifest,
    EstimandSpec,
    RunManifest,
    arm_treatment_hash,
    freeze_run_manifest,
)
from worldspace.attribution.public_loop import (
    PublicRunConfig,
    SUMMARY_FILENAME,
    SUMMARY_SCHEMA,
    TRACE_FILENAME,
    append_jsonl,
    floor_recipe_hash,
    occupied_genotype_hash,
    search_curve_auc,
    should_use_llm,
    write_json,
)
from worldspace.attribution.records import (
    ArchiveState,
    BudgetCheckpoint,
    BudgetCounters,
    RunSummary,
)
from worldspace.attribution.validation import (
    AttributionAdmissionError,
    admit_analysis_cohort,
)
from worldspace.nas201.descriptors import Nas201BinEdges
from worldspace.nas201.live_prompt_scan import assert_live_prompt_templates
from worldspace.nas201.llm_emitter import Nas201LlmEmitter, bind_nas201_llm
from worldspace.nas201.runner import ARCHIVE_FILENAME, run_nas201_qd
from worldspace.nas201.table import Nas201SearchRecord
from worldspace.pcg.descriptors import bin_edges_from_measures
from worldspace.pcg.live_prompt_scan import (
    assert_live_prompt_templates as assert_pcg_live,
)
from worldspace.pcg.llm_emitter import PcgSokobanLlmEmitter, bind_pcg_llm
from worldspace.pcg.runner import run_pcg_sokoban_qd
from worldspace.pcg.spec import SOKOBAN_V0

from tests.test_attribution_harness import HASH_A, HASH_B, _arm, _component


class _AlwaysHitNasLookup:
    def lookup_search(self, arch_str: str) -> Nas201SearchRecord | None:
        digest = int(hashlib.sha256(arch_str.encode()).hexdigest()[:8], 16)
        params = 1e4 + (digest % 9000)
        flops = 1e6 + (digest % 900000)
        accuracy = 40.0 + (digest % 5000) / 100.0
        return Nas201SearchRecord(
            index=digest % 100000,
            arch=arch_str,
            flops=flops,
            params=params,
            latency=None,
            valid_accuracy=accuracy,
            n_trials=3,
        )

    def __len__(self) -> int:
        return 1


class _ToyPcgEnv:
    def info(self, contents: object) -> dict[str, Any]:
        grid = cast(list[list[int]], contents)
        zeros = sum(tile == 0 for row in grid for tile in row)
        crates = sum(tile == 3 for row in grid for tile in row)
        players = sum(tile == 2 for row in grid for tile in row)
        return {
            "players": players,
            "crates": crates,
            "targets": crates,
            "content": grid,
            "heuristic": -1,
            "solution": [0] * zeros,
        }

    def quality(self, contents: object) -> tuple[float, float, dict[str, Any]]:
        info = self.info(contents)
        quality = min(1.0, 0.05 * (int(info["players"]) + 1))
        return 0.0, quality, info


def _public_run_manifest(
    domain: str,
    *,
    seed: int,
    selector: str,
    generator: str,
    repair: str,
    prompt_channel: str = "not_applicable",
    allocation: str = "static",
) -> RunManifest:
    arm_payload = _arm(
        domain,
        arm_id=f"{generator}-{selector}",
        role="baseline",
        selector=selector,
        generator=generator,
    )
    arm_payload["treatment"]["initialization"] = _component(
        "generated_floor",
        floor_random=20,
        archive_capacity=400 if domain == "nas201" else 100,
    )
    arm_payload["treatment"]["prompt_channel"] = _component(prompt_channel)
    arm_payload["treatment"]["repair_fallback"] = _component(repair)
    arm_payload["treatment"]["allocation"] = _component(allocation)
    arm = ArmManifest.model_validate(arm_payload)
    caps = current_domain_capabilities()[domain]
    return freeze_run_manifest(
        {
            "run_id": f"public-{domain}-seed-{seed}",
            "study_id": f"public-{domain}",
            "arm_id": arm.arm_id,
            "pair_id": f"public-{domain}-seed-{seed}",
            "block_id": "local",
            "evidence_tier": "feasibility",
            "protocol_id": "public-adapters",
            "protocol_hash": HASH_A,
            "domain_id": domain,
            "domain_version": "native-fixture-1",
            "adapter_id": caps.adapter_id,
            "adapter_version": caps.adapter_version,
            "seed": seed,
            "domain_instance_id": f"{domain}-default",
            "initial_archive_id": "generated-floor",
            "initial_archive_hash": canonical_sha256([]),
            "treatment": arm.treatment,
            "representation": arm.representation,
            "model": arm.model,
            "evaluator": arm.evaluator,
            "treatment_hash": arm_treatment_hash(arm),
            "study_manifest_hash": HASH_A,
            "currency": "USD",
            "price_table_id": "fixture-free",
            "price_table_hash": HASH_B,
            "dependency_hashes": {"lock": HASH_B},
            "output_paths": {"run_dir": f"output/{domain}/seed-{seed}"},
            "expected_artifacts": [
                "nightly_run_summary.json",
                "archive_trace.jsonl",
            ],
        }
    )


class TestPublicTaskCapabilities(unittest.TestCase):
    def test_registry_includes_public_tasks(self) -> None:
        declarations = current_domain_capabilities()
        self.assertEqual(
            set(declarations),
            {"ca", "maze", "dungeon", "sphere", "nas201", "pcg_sokoban"},
        )
        self.assertIn("state_aware_median", declarations["nas201"].allocation_kinds)
        self.assertIn(
            "structural_counts",
            declarations["pcg_sokoban"].repair_fallback_kinds,
        )
        self.assertIn("generated_floor", declarations["nas201"].initialization_kinds)
        self.assertEqual(declarations["nas201"].repair_fallback_kinds, ("identity",))
        self.assertNotIn(
            "genetic_fallback",
            declarations["pcg_sokoban"].repair_fallback_kinds,
        )


class TestStateAwareAllocation(unittest.TestCase):
    def test_median_rule(self) -> None:
        self.assertTrue(
            should_use_llm(
                allocation="state_aware_median",
                archive_fitnesses=[1.0, 2.0, 3.0],
                target_empty=False,
                target_fitness=1.5,
                completed_llm_calls=0,
                llm_call_cap=10,
            )
        )
        self.assertFalse(
            should_use_llm(
                allocation="state_aware_median",
                archive_fitnesses=[1.0, 2.0, 3.0],
                target_empty=False,
                target_fitness=2.5,
                completed_llm_calls=0,
                llm_call_cap=10,
            )
        )
        # Even-length archives use the average of the two middle elites.
        self.assertTrue(
            should_use_llm(
                allocation="state_aware_median",
                archive_fitnesses=[1.0, 2.0],
                target_empty=False,
                target_fitness=1.5,
                completed_llm_calls=0,
                llm_call_cap=10,
            )
        )
        self.assertFalse(
            should_use_llm(
                allocation="state_aware_median",
                archive_fitnesses=[1.0, 2.0],
                target_empty=False,
                target_fitness=1.6,
                completed_llm_calls=0,
                llm_call_cap=10,
            )
        )
        self.assertFalse(
            should_use_llm(
                allocation="state_aware_median",
                archive_fitnesses=[1.0, 2.0, 3.0, 4.0],
                target_empty=False,
                target_fitness=2.6,
                completed_llm_calls=0,
                llm_call_cap=10,
            )
        )


class TestLivePrompts(unittest.TestCase):
    def test_live_templates_pass_scanners(self) -> None:
        assert_live_prompt_templates()
        assert_pcg_live()


class TestNas201AdapterLoop(unittest.TestCase):
    def test_genetic_run_normalizes_read_only(self) -> None:
        edges = Nas201BinEdges(
            resolution=4,
            log_params_min=4.0,
            log_params_max=5.0,
            log_flops_min=6.0,
            log_flops_max=7.0,
            n_architectures=1,
            source_sha256="a" * 64,
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            config = PublicRunConfig(
                seed=11,
                generator="genetic",
                selector="uniform_frontier",
                floor_random=5,
                search_horizon=8,
            )
            result = run_nas201_qd(
                _AlwaysHitNasLookup(),
                edges,
                config,
                output_dir=run_dir,
            )
            before = {
                path.name: path.read_bytes()
                for path in run_dir.iterdir()
                if path.is_file()
            }
            self.assertGreater(result.filled_cells, 0)
            manifest = _public_run_manifest(
                "nas201",
                seed=11,
                selector="uniform_frontier",
                generator="genetic",
                repair="identity",
            )
            bundle = Nas201NormalizationAdapter().normalize(
                manifest,
                NativeRunInputs(run_dir),
            )
            after = {
                path.name: path.read_bytes()
                for path in run_dir.iterdir()
                if path.is_file()
            }
            self.assertEqual(before, after)
            self.assertEqual(
                bundle.summary.final_archive.occupied_cells, result.filled_cells
            )
            self.assertEqual(bundle.summary.event_completeness, "partial")
            self.assertEqual(bundle.summary.floor_proposals, config.floor_random)
            self.assertEqual(bundle.summary.qd_score_anytime_auc, result.anytime_auc)
            self.assertEqual(
                bundle.summary.final_counters.evaluator_attempts,
                result.summary["evaluations"],
            )
            self.assertEqual(
                bundle.summary.final_counters.evaluator_completions,
                result.summary["evaluations"],
            )
            self.assertEqual(
                bundle.summary.final_counters.valid_proposals,
                result.summary["valid_proposals"],
            )

    def test_evaluator_completions_count_failed_evaluations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_json(
                run_dir / SUMMARY_FILENAME,
                {
                    "schema_version": SUMMARY_SCHEMA,
                    "domain": "nas201",
                    "benchmark": "nas201",
                    "seed": 11,
                    "generator": "genetic",
                    "target_selection": "uniform_frontier",
                    "allocation": "static",
                    "prompt_channel": "not_applicable",
                    "repair": "identity",
                    "n_cells": 4,
                    "proposals": 5,
                    "evaluations": 5,
                    "valid_proposals": 3,
                    "filled_cells": 1,
                    "coverage": 0.25,
                    "qd_score": 0.5,
                    "completed": True,
                },
            )
            (run_dir / ARCHIVE_FILENAME).write_text(
                '{"fitness": 0.5}\n',
                encoding="utf-8",
            )
            append_jsonl(
                run_dir / TRACE_FILENAME,
                {
                    "proposals": 5,
                    "evaluations": 5,
                    "filled_cells": 1,
                    "coverage": 0.25,
                    "qd_score": 0.5,
                    "mean_best_fitness": 0.5,
                },
            )
            bundle = Nas201NormalizationAdapter().normalize(
                _public_run_manifest(
                    "nas201",
                    seed=11,
                    selector="uniform_frontier",
                    generator="genetic",
                    repair="identity",
                ),
                NativeRunInputs(run_dir),
            )
            counters = bundle.summary.final_counters
            self.assertEqual(counters.valid_proposals, 3)
            self.assertEqual(counters.evaluator_attempts, 5)
            self.assertEqual(counters.evaluator_completions, 5)

    def test_runner_rejects_genetic_fallback_repair(self) -> None:
        edges = Nas201BinEdges(
            resolution=4,
            log_params_min=0.0,
            log_params_max=8.0,
            log_flops_min=0.0,
            log_flops_max=10.0,
            n_architectures=1,
            source_sha256="a" * 64,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_nas201_qd(
                    _AlwaysHitNasLookup(),
                    edges,
                    PublicRunConfig(
                        seed=11,
                        generator="genetic",
                        selector="uniform_frontier",
                        repair_kind="genetic_fallback",
                        floor_random=1,
                        search_horizon=1,
                    ),
                    output_dir=Path(tmp),
                )

    def test_floor_hash_is_the_occupied_genotypes(self) -> None:
        edges = Nas201BinEdges(
            resolution=4,
            log_params_min=0.0,
            log_params_max=8.0,
            log_flops_min=0.0,
            log_flops_max=10.0,
            n_architectures=1,
            source_sha256="a" * 64,
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result = run_nas201_qd(
                _AlwaysHitNasLookup(),
                edges,
                PublicRunConfig(
                    seed=11,
                    generator="genetic",
                    selector="uniform_frontier",
                    floor_random=2,
                    search_horizon=0,
                ),
                output_dir=run_dir,
            )
            genotypes = [
                json.loads(line)["genotype_hash"]
                for line in (run_dir / ARCHIVE_FILENAME).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            realized = occupied_genotype_hash(genotypes)
            self.assertEqual(result.summary["initial_archive_hash"], realized)
            self.assertNotEqual(realized, canonical_sha256([]))
            bundle = Nas201NormalizationAdapter().normalize(
                _public_run_manifest(
                    "nas201",
                    seed=11,
                    selector="uniform_frontier",
                    generator="genetic",
                    repair="identity",
                ),
                NativeRunInputs(run_dir),
            )
            self.assertEqual(bundle.summary.initial_archive_hash, realized)
            self.assertEqual(
                bundle.summary.floor_recipe_hash,
                _public_run_manifest(
                    "nas201",
                    seed=11,
                    selector="uniform_frontier",
                    generator="genetic",
                    repair="identity",
                ).initial_archive_hash,
            )


class TestPcgSokobanAdapterLoop(unittest.TestCase):
    def test_genetic_structural_counts_normalizes(self) -> None:
        env = _ToyPcgEnv()
        edges = bin_edges_from_measures(
            [(float(i), float(j)) for i in range(0, 26, 5) for j in range(0, 26, 5)],
            measure_names=("solution_length", "crates"),
            problem_name=SOKOBAN_V0.problem_name,
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            config = PublicRunConfig(
                seed=13,
                generator="genetic",
                selector="min_fitness_frontier",
                repair_kind="structural_counts",
                floor_random=4,
                search_horizon=6,
            )
            result = run_pcg_sokoban_qd(
                env,
                edges,
                config,
                output_dir=run_dir,
            )
            before = {
                path.name: path.read_bytes()
                for path in run_dir.iterdir()
                if path.is_file()
            }
            self.assertGreater(result.filled_cells, 0)
            manifest = _public_run_manifest(
                "pcg_sokoban",
                seed=13,
                selector="min_fitness_frontier",
                generator="genetic",
                repair="structural_counts",
            )
            bundle = PcgSokobanNormalizationAdapter().normalize(
                manifest,
                NativeRunInputs(run_dir),
            )
            after = {
                path.name: path.read_bytes()
                for path in run_dir.iterdir()
                if path.is_file()
            }
            self.assertEqual(before, after)
            self.assertEqual(bundle.summary.domain_id, "pcg_sokoban")


class TestConfirmatoryJobBuilder(unittest.TestCase):
    def test_studies_emit_without_launch(self) -> None:
        for domain in ("nas201", "pcg_sokoban"):
            for builder in (
                selector_channel_study,
                generator_study,
                allocation_study,
            ):
                study = builder(domain)  # type: ignore[arg-type]
                plan = build_confirmatory_job_plan(
                    study,
                    output_root=f"artifacts/attribution-jobs/{study.study_id}",
                )
                self.assertFalse(plan.launched)
                self.assertEqual(len(plan.runs), len(study.arms) * 20)
                self.assertTrue(study.study_id.startswith(f"confirmatory-{domain}-"))
                empty_floor = canonical_sha256([])
                repair = "identity" if domain == "nas201" else "structural_counts"
                by_seed: dict[int, set[str]] = {}
                for run in plan.runs:
                    self.assertNotEqual(run.initial_archive_hash, empty_floor)
                    by_seed.setdefault(run.seed, set()).add(run.initial_archive_hash)
                    self.assertEqual(
                        run.initial_archive_hash,
                        floor_recipe_hash(
                            seed=run.seed,
                            floor_random=20,
                            repair=repair,
                            bin_edges=confirmatory_bin_edges(domain),
                        ),
                    )
                self.assertTrue(all(len(group) == 1 for group in by_seed.values()))
                self.assertGreater(len(by_seed), 1)


class TestConfirmatoryAdmission(unittest.TestCase):
    def test_refuse_feasibility_and_q1(self) -> None:
        overlapping = (
            "/tmp/artifacts/controlled_attribution/nas201/smoke/feasibility.jsonl"
        )
        with self.assertRaises(AttributionAdmissionError) as caught:
            refuse_contaminated_artifact_paths([overlapping])
        self.assertEqual(
            [issue.code for issue in caught.exception.issues],
            ["confirmatory.refuse_feasibility_jsonl"],
        )
        with self.assertRaises(AttributionAdmissionError) as caught:
            refuse_contaminated_artifact_paths(
                [
                    "/tmp/artifacts/controlled_attribution/pcg/isolated/run.json",
                ]
            )
        self.assertEqual(
            [issue.code for issue in caught.exception.issues],
            ["confirmatory.refuse_feasibility_design_data"],
        )
        with self.assertRaises(AttributionAdmissionError):
            refuse_contaminated_artifact_paths(
                ["artifacts/nightly/q1/nightly_run_summary.json"]
            )
        with self.assertRaises(AttributionAdmissionError):
            refuse_mixed_evidence_tier(["confirmatory", "feasibility"])

    def test_response_model_mismatch_rejects_the_arm(self) -> None:
        pinned = "gpt-4o-mini-2024-07-18"
        mismatch = {
            "expected_response_model": pinned,
            "response_model": "other-snapshot",
            "response_model_mismatch": True,
        }
        with self.assertRaises(AttributionAdmissionError) as caught:
            refuse_response_model_mismatch([mismatch])
        self.assertEqual(
            caught.exception.issues[0].code,
            "confirmatory.response_model_mismatch",
        )
        hidden = {
            "expected_response_model": pinned,
            "response_model": "other-snapshot",
            "response_model_mismatch": False,
        }
        with self.assertRaises(AttributionAdmissionError):
            refuse_response_model_mismatch([hidden])
        refuse_response_model_mismatch(
            [
                {
                    "expected_response_model": pinned,
                    "response_model": pinned,
                    "response_model_mismatch": False,
                }
            ]
        )

    def test_missing_tokens_reject_only_llm_arms(self) -> None:
        refuse_missing_llm_usage(
            [
                {
                    "generator": "genetic",
                    "llm_enabled": False,
                    "total_tokens": None,
                },
                {
                    "generator": "random",
                    "total_tokens": None,
                },
                {
                    "generator": "llm",
                    "llm_calls_completed": 0,
                    "llm_http_calls": 0,
                    "total_tokens": None,
                },
            ],
            missing_usage_policy="reject",
        )
        refuse_missing_llm_usage(
            [
                {
                    "generator": "llm",
                    "llm_calls_attempted": 1,
                    "llm_calls_completed": 1,
                    "llm_http_calls": 2,
                    "total_tokens": 12,
                }
            ],
            missing_usage_policy="reject",
        )
        with self.assertRaises(AttributionAdmissionError) as caught:
            refuse_missing_llm_usage(
                [
                    {
                        "generator": "llm",
                        "llm_calls_attempted": 1,
                        "llm_calls_completed": 1,
                        "llm_http_calls": 2,
                        "total_tokens": None,
                    }
                ],
                missing_usage_policy="reject",
            )
        self.assertEqual(
            caught.exception.issues[0].code,
            "confirmatory.missing_llm_usage",
        )

    def test_same_seed_with_different_floors_does_not_pair(self) -> None:
        left = _auc_summary("run-a", "live", [1.0, 2.0])
        right = _auc_summary("run-b", "constant", [1.0, 2.0])
        with self.assertRaises(AttributionAdmissionError) as caught:
            admit_analysis_cohort(
                (
                    left,
                    right.model_copy(update={"initial_archive_hash": HASH_B}),
                ),
                expected_arm_ids=("live", "constant"),
                minimum_complete_pairs=1,
            )
        self.assertTrue(
            any(
                issue.code == "cohort.unpaired_initial_archive_hash"
                for issue in caught.exception.issues
            )
        )
        admit_analysis_cohort(
            (left, right),
            expected_arm_ids=("live", "constant"),
            minimum_complete_pairs=1,
        )


class TestLiveChannelSummary(unittest.TestCase):
    def test_nas_summary_records_rendered_channel_and_http_retries(self) -> None:
        edges = Nas201BinEdges(
            resolution=4,
            log_params_min=0.0,
            log_params_max=8.0,
            log_flops_min=0.0,
            log_flops_max=10.0,
            n_architectures=1,
            source_sha256="a" * 64,
        )
        caller = _RetryCaller('{"ops":["none","none","none","none","none","none"]}')
        emitter = Nas201LlmEmitter(
            call_llm_text=caller,
            max_retries=1,
            channel="live",
        )
        seen: list[Any] = []
        inner = bind_nas201_llm(emitter)

        def emit(parent: Any, rng: Any, index: int, elite: Any) -> Any:
            seen.append(elite)
            return inner(parent, rng, index, elite)

        usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        pinned = "gpt-4o-mini-2024-07-18"
        config = PublicRunConfig(
            seed=11,
            generator="llm",
            selector="uniform_frontier",
            prompt_channel="constant",
            floor_random=1,
            search_horizon=1,
            expected_response_model=pinned,
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "worldspace.nas201.llm_emitter._usage_from_call_id",
            return_value=usage,
        ), patch(
            "worldspace.nas201.llm_emitter._response_model_from_call_id",
            return_value="other-snapshot",
        ), patch("worldspace.nas201.llm_emitter.time.sleep"):
            result = run_nas201_qd(
                _AlwaysHitNasLookup(),
                edges,
                config,
                output_dir=Path(tmp),
                llm_emit=emit,
            )
            on_disk = json.loads(
                (Path(tmp) / SUMMARY_FILENAME).read_text(encoding="utf-8")
            )
        self.assertEqual(result.summary["llm_calls_completed"], 1)
        self.assertEqual(on_disk["prompt_channel"], "live")
        self.assertEqual(on_disk["prompt_template_sha256"], emitter.prompt_template_sha256)
        self.assertEqual(on_disk["llm_calls_attempted"], 1)
        self.assertEqual(on_disk["llm_http_calls"], 2)
        self.assertEqual(on_disk["prompt_tokens"], 10)
        self.assertEqual(on_disk["completion_tokens"], 2)
        self.assertEqual(on_disk["total_tokens"], 12)
        self.assertEqual(on_disk["response_model"], "other-snapshot")
        self.assertTrue(on_disk["response_model_mismatch"])
        self.assertEqual(len(seen), 1)
        prompt = caller.prompts[-1]
        self.assertIn(
            f"parent_valid_accuracy={format(seen[0].fitness, '.8g')}",
            prompt,
        )
        self.assertIn(
            f"parent_log_params={format(seen[0].measures[0], '.8g')}",
            prompt,
        )
        self.assertIn(
            f"parent_log_flops={format(seen[0].measures[1], '.8g')}",
            prompt,
        )
        self.assertNotIn("test", prompt.lower())
        with self.assertRaises(AttributionAdmissionError):
            refuse_response_model_mismatch([on_disk])

    def test_pcg_summary_records_rendered_channel_and_http_retries(self) -> None:
        edges = bin_edges_from_measures(
            [(0.0, 0.0), (30.0, 30.0)],
            measure_names=("solution_length", "crates"),
            problem_name=SOKOBAN_V0.problem_name,
        )
        grid = json.dumps({"grid": [[0, 0, 0, 0, 0]] * 5})
        caller = _RetryCaller(grid)
        emitter = PcgSokobanLlmEmitter(
            call_llm_text=caller,
            max_retries=1,
            channel="live",
        )
        seen: list[Any] = []
        inner = bind_pcg_llm(emitter)

        def emit(parent: Any, rng: Any, index: int, elite: Any) -> Any:
            seen.append(elite)
            return inner(parent, rng, index, elite)

        usage = {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}
        pinned = "gpt-4o-mini-2024-07-18"
        config = PublicRunConfig(
            seed=13,
            generator="llm",
            selector="uniform_frontier",
            prompt_channel="constant",
            repair_kind="identity",
            floor_random=1,
            search_horizon=1,
            expected_response_model=pinned,
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "worldspace.pcg.llm_emitter._usage_and_model_from_call_id",
            return_value=(usage, "other-snapshot"),
        ), patch("worldspace.pcg.llm_emitter.time.sleep"):
            result = run_pcg_sokoban_qd(
                _ToyPcgEnv(),
                edges,
                config,
                output_dir=Path(tmp),
                llm_emit=emit,
            )
        self.assertEqual(result.summary["llm_calls_completed"], 1)
        self.assertEqual(result.summary["prompt_channel"], "live")
        self.assertEqual(
            result.summary["prompt_template_sha256"],
            emitter.prompt_template_sha256,
        )
        self.assertEqual(result.summary["llm_http_calls"], 2)
        self.assertEqual(result.summary["prompt_tokens"], 8)
        self.assertEqual(result.summary["total_tokens"], 11)
        self.assertTrue(result.summary["response_model_mismatch"])
        prompt = caller.prompts[-1]
        elite = seen[0]
        self.assertIn(f"parent_fitness={format(elite.fitness, '.8g')}", prompt)
        self.assertIn(
            f"parent_measure_0={format(elite.measures[0], '.8g')}",
            prompt,
        )
        self.assertIn(
            "parent_playable=" + ("true" if elite.playable else "false"),
            prompt,
        )
        self.assertNotRegex(prompt, r"\barchive\b")
        with self.assertRaises(AttributionAdmissionError):
            refuse_response_model_mismatch([result.summary])


class _RetryCaller:
    def __init__(self, good: str) -> None:
        self.good = good
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, **kwargs: object) -> str:
        self.prompts.append(str(kwargs["prompt"]))
        self.calls += 1
        if self.calls == 1:
            return "not json"
        return self.good


class TestSearchCurveFamilies(unittest.TestCase):
    def test_search_curve_auc_drops_floor(self) -> None:
        curve = [10.0, 10.0, 1.0, 3.0, 5.0]
        self.assertEqual(search_curve_auc(curve, 2), 3.0)

    def test_contrast_uses_post_floor_auc(self) -> None:
        estimand = EstimandSpec.model_validate(
            {
                "estimand_id": "channel_at_uniform",
                "endpoint": "qd_score_anytime_auc",
                "form": "anytime_auc",
                "budget_axis": "proposal",
                "treatment_arm_ids": ["live"],
                "control_arm_ids": ["constant"],
                "paired_by": ["seed"],
                "alternative": "two_sided",
                "margin": None,
                "missing_policy": "complete_pairs_only",
                "minimum_complete_pairs": 1,
            }
        )
        live = _auc_summary("live-run", "live", [10.0, 10.0, 1.0, 3.0, 5.0])
        constant = _auc_summary("constant-run", "constant", [10.0, 10.0, 1.0, 1.0, 1.0])
        contrast = descriptive_contrast(
            [live, constant],
            estimand=estimand,
            checkpoints_by_run={
                live.run_id: _qd_trace(live.run_id, [10.0, 10.0, 1.0, 3.0, 5.0]),
                constant.run_id: _qd_trace(
                    constant.run_id, [10.0, 10.0, 1.0, 1.0, 1.0]
                ),
            },
        )
        self.assertAlmostEqual(contrast.mean_difference or 0.0, 2.0, places=9)
        self.assertEqual(_endpoint_value(live, "qd_score_anytime_auc"), 3.0)

    def test_families_split_holm_contrasts(self) -> None:
        for domain in ("nas201", "pcg_sokoban"):
            selector = selector_channel_study(domain)  # type: ignore[arg-type]
            by_id = {item.estimand_id: item for item in selector.estimands}
            self.assertEqual(
                [item.estimand_id for item in selector.estimands if item.multiplicity_rule == "holm"],
                ["selector_at_constant", "channel_at_uniform"],
            )
            self.assertEqual(
                by_id["selector_at_constant"].treatment_arm_ids, ("A-const-minfit",)
            )
            self.assertEqual(
                by_id["selector_at_constant"].control_arm_ids, ("A-const-uniform",)
            )
            self.assertEqual(
                by_id["channel_at_uniform"].treatment_arm_ids, ("A-live-uniform",)
            )
            self.assertEqual(
                by_id["channel_at_uniform"].control_arm_ids, ("A-const-uniform",)
            )
            self.assertEqual(by_id["channel_x_selector"].form, "interaction")
            self.assertIsNone(by_id["channel_x_selector"].confirmatory_family)
            self.assertEqual(by_id["channel_x_selector"].multiplicity_rule, "descriptive")

            generator = generator_study(domain)  # type: ignore[arg-type]
            generated = {item.estimand_id: item for item in generator.estimands}
            self.assertEqual(
                generated["llm_minus_genetic"].treatment_arm_ids, ("B-llm",)
            )
            self.assertEqual(
                generated["llm_minus_genetic"].control_arm_ids, ("B-genetic",)
            )
            self.assertEqual(
                generated["llm_minus_random"].treatment_arm_ids, ("B-llm",)
            )
            self.assertEqual(
                generated["llm_minus_random"].control_arm_ids, ("B-random",)
            )
            self.assertTrue(
                all(item.confirmatory_family == "F-B" for item in generator.estimands)
            )

            allocation = allocation_study(domain)  # type: ignore[arg-type]
            self.assertEqual(len(allocation.estimands), 1)
            only = allocation.estimands[0]
            self.assertEqual(only.estimand_id, "allocation")
            self.assertEqual(only.confirmatory_family, "F-C")
            self.assertEqual(only.treatment_arm_ids, ("C-state-aware-median",))
            self.assertEqual(only.control_arm_ids, ("C-static",))

    def test_holm_excludes_interaction(self) -> None:
        study = selector_channel_study("nas201")
        decisions = family_holm(
            study.estimands,
            {
                "selector_at_constant": [1.0] * 10,
                "channel_at_uniform": [1.0, -1.0] * 5,
                "channel_x_selector": [5.0] * 10,
            },
            family="F-A",
        )
        self.assertEqual(
            [item.estimand_id for item in decisions],
            ["selector_at_constant", "channel_at_uniform"],
        )
        by_id = {item.estimand_id: item for item in decisions}
        self.assertTrue(by_id["selector_at_constant"].reject)
        self.assertFalse(by_id["channel_at_uniform"].reject)
        self.assertTrue(holm_step_down({"channel": 0.03})["channel"])
        self.assertFalse(
            holm_step_down({"channel": 0.03, "interaction": 0.9})["channel"]
        )


def _auc_summary(run_id: str, arm_id: str, curve: list[float]) -> RunSummary:
    observed = {axis: "observed" for axis in (
        "proposal",
        "valid_proposal",
        "evaluation",
        "llm_call_attempted",
        "llm_call_completed",
        "prompt_token",
        "completion_token",
        "token",
        "evaluator_wall_time",
        "llm_latency",
        "wall_time",
        "monetary",
    )}
    return RunSummary(
        run_id=run_id,
        study_id="auc-study",
        arm_id=arm_id,
        pair_id="pair-1",
        evidence_tier="confirmatory",
        domain_id="nas201",
        domain_version="test",
        adapter_id="nas201-native",
        adapter_version="0.1",
        evaluator_hash=HASH_A,
        seed=1,
        domain_instance_id="nas201-default",
        initial_archive_hash=HASH_A,
        protocol_hash=HASH_A,
        study_manifest_hash=HASH_A,
        run_manifest_hash=HASH_B,
        treatment_hash=HASH_A,
        event_completeness="partial",
        final_counters=BudgetCounters(
            proposal_slots=len(curve),
            valid_proposals=len(curve),
            evaluator_attempts=len(curve),
            evaluator_completions=len(curve),
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
        counter_completeness=observed,  # type: ignore[arg-type]
        final_archive=ArchiveState(
            occupied_cells=1,
            capacity=10,
            coverage=0.1,
            raw_qd_score=curve[-1],
            normalized_qd_score=curve[-1] / 10.0,
            maximum_elite_quality=curve[-1],
            occupied_mean_quality=curve[-1],
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
        qd_score_anytime_auc=search_curve_auc(curve, 2),
        floor_proposals=2,
    )


def _qd_trace(run_id: str, curve: list[float]) -> tuple[BudgetCheckpoint, ...]:
    completeness = {
        "proposal": "observed",
        "valid_proposal": "unavailable",
        "evaluation": "unavailable",
        "llm_call_attempted": "unavailable",
        "llm_call_completed": "unavailable",
        "prompt_token": "unavailable",
        "completion_token": "unavailable",
        "token": "unavailable",
        "evaluator_wall_time": "unavailable",
        "llm_latency": "unavailable",
        "wall_time": "unavailable",
        "monetary": "unavailable",
    }
    checkpoints: list[BudgetCheckpoint] = []
    for index, qd in enumerate(curve, start=1):
        checkpoints.append(
            BudgetCheckpoint(
                run_id=run_id,
                checkpoint_index=index - 1,
                indexed_by="proposal",
                counters=BudgetCounters(
                    proposal_slots=index,
                    valid_proposals=None,
                    evaluator_attempts=None,
                    evaluator_completions=None,
                    llm_attempts=None,
                    llm_completions=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    evaluator_seconds=None,
                    llm_latency_seconds=None,
                    wall_seconds=None,
                    monetary_cost=None,
                ),
                source_completeness=completeness,  # type: ignore[arg-type]
                calls_allocated_by_operator={},
                calls_used_by_operator={},
                calls_remaining_by_operator={},
                calls_forfeited_by_operator={},
                archive=ArchiveState(
                    occupied_cells=1,
                    capacity=10,
                    coverage=0.1,
                    raw_qd_score=qd,
                    normalized_qd_score=qd / 10.0,
                    maximum_elite_quality=qd,
                    occupied_mean_quality=qd,
                ),
            )
        )
    return tuple(checkpoints)


if __name__ == "__main__":
    unittest.main()

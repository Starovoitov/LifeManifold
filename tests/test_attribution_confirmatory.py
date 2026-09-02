"""Public-domain adapters, loops, live prompts, and confirmatory job emission."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

from worldspace.attribution.adapters import (
    Nas201NormalizationAdapter,
    NativeRunInputs,
    PcgSokobanNormalizationAdapter,
)
from worldspace.attribution.capabilities import current_domain_capabilities
from worldspace.attribution.confirmatory_admission import (
    refuse_contaminated_artifact_paths,
    refuse_mixed_evidence_tier,
)
from worldspace.attribution.confirmatory_studies import (
    allocation_study,
    build_confirmatory_job_plan,
    generator_study,
    selector_channel_study,
)
from worldspace.attribution.hashing import canonical_sha256
from worldspace.attribution.manifest import (
    ArmManifest,
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
    should_use_llm,
    write_json,
)
from worldspace.attribution.validation import AttributionAdmissionError
from worldspace.nas201.descriptors import Nas201BinEdges
from worldspace.nas201.live_prompt_scan import assert_live_prompt_templates
from worldspace.nas201.runner import ARCHIVE_FILENAME, run_nas201_qd
from worldspace.nas201.table import Nas201SearchRecord
from worldspace.pcg.descriptors import bin_edges_from_measures
from worldspace.pcg.live_prompt_scan import (
    assert_live_prompt_templates as assert_pcg_live,
)
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
                capture_events=False,
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
            self.assertEqual(bundle.summary.event_completeness, "summary_only")
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

    def test_genetic_fallback_summary_records_identity_repair(self) -> None:
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
            result = run_nas201_qd(
                _AlwaysHitNasLookup(),
                edges,
                PublicRunConfig(
                    seed=11,
                    generator="genetic",
                    selector="uniform_frontier",
                    repair_kind="genetic_fallback",
                    floor_random=5,
                    search_horizon=8,
                    capture_events=False,
                ),
                output_dir=run_dir,
            )
            self.assertEqual(result.summary["repair"], "identity")
            bundle = Nas201NormalizationAdapter().normalize(
                _public_run_manifest(
                    "nas201",
                    seed=11,
                    selector="uniform_frontier",
                    generator="genetic",
                    repair="genetic_fallback",
                ),
                NativeRunInputs(run_dir),
            )
            self.assertEqual(bundle.summary.domain_id, "nas201")


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


if __name__ == "__main__":
    unittest.main()

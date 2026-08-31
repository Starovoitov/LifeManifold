from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from worldspace.attribution.adapters import (
    CaNormalizationAdapter,
    MazeNormalizationAdapter,
    NativeRunInputs,
    NormalizationError,
)
from worldspace.attribution.capabilities import current_domain_capabilities
from worldspace.attribution.hashing import canonical_sha256
from worldspace.attribution.manifest import (
    ArmManifest,
    RunManifest,
    arm_treatment_hash,
    freeze_run_manifest,
)
from worldspace.illuminators.scheduler import DEFAULT_MINI_SCHEDULER_PATH
from worldspace.mazes.runner import MazeSchedulerConfig, run_maze_qd
from worldspace.scripts.run_map_elites_nightly import run_map_elites_nightly

from tests.test_attribution_harness import HASH_A, HASH_B, _arm, _component

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "attribution"


def _run_manifest(
    domain: str,
    *,
    seed: int,
    selector: str,
    generator: str,
    gate: str = "off",
) -> RunManifest:
    arm_payload = _arm(
        domain,
        arm_id=f"{generator}-{selector}",
        role="baseline",
        selector=selector,
        generator=generator,
    )
    arm_payload["treatment"]["gate"] = _component(gate)
    arm = ArmManifest.model_validate(arm_payload)
    adapter_id = "ca-native" if domain == "ca" else "maze-native"
    return freeze_run_manifest(
        {
            "run_id": f"p11-{domain}-seed-{seed}",
            "study_id": f"p11-{domain}",
            "arm_id": arm.arm_id,
            "pair_id": f"p11-{domain}-seed-{seed}",
            "block_id": "local",
            "evidence_tier": "feasibility",
            "protocol_id": "p11-read-only",
            "protocol_hash": HASH_A,
            "domain_id": domain,
            "domain_version": "native-fixture-1",
            "adapter_id": adapter_id,
            "adapter_version": "0.1",
            "seed": seed,
            "domain_instance_id": f"{domain}-default",
            "initial_archive_id": "empty",
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
                (
                    "map_elites_archive.jsonl"
                    if domain == "ca"
                    else "maze_archive.jsonl"
                ),
            ],
        }
    )


def _snapshot_tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class TestSyntheticNormalizationFixtures(unittest.TestCase):
    def test_ca_partial_fixture_preserves_missing_usage(self) -> None:
        run_dir = FIXTURES / "ca_partial"
        before = _snapshot_tree(run_dir)
        manifest = _run_manifest(
            "ca",
            seed=0,
            selector="min_fitness_frontier",
            generator="llm",
        )

        bundle = CaNormalizationAdapter().normalize(
            manifest,
            NativeRunInputs(run_dir),
        )

        self.assertEqual(bundle.summary.event_completeness, "partial")
        self.assertEqual(len(bundle.events), 2)
        self.assertEqual(bundle.summary.final_archive.raw_qd_score, 0.5)
        self.assertEqual(bundle.summary.final_archive.normalized_qd_score, 0.125)
        self.assertIsNone(bundle.summary.final_counters.total_tokens)
        self.assertEqual(
            bundle.summary.counter_completeness["token"],
            "unavailable",
        )
        self.assertEqual(_snapshot_tree(run_dir), before)

    def test_maze_summary_only_fixture_preserves_skip_accounting(self) -> None:
        run_dir = FIXTURES / "maze_summary_only"
        before = _snapshot_tree(run_dir)
        manifest = _run_manifest(
            "maze",
            seed=0,
            selector="uniform_frontier",
            generator="genetic",
            gate="filter",
        )

        bundle = MazeNormalizationAdapter().normalize(
            manifest,
            NativeRunInputs(run_dir),
        )

        self.assertEqual(bundle.summary.event_completeness, "summary_only")
        self.assertEqual(bundle.events, ())
        self.assertEqual(bundle.summary.final_counters.proposal_slots, 2)
        self.assertEqual(bundle.summary.final_counters.valid_proposals, 1)
        self.assertEqual(bundle.summary.final_counters.evaluator_completions, 1)
        self.assertEqual(bundle.summary.final_archive.raw_qd_score, 0.5)
        self.assertEqual(len(bundle.checkpoints), 4)
        counters = bundle.summary.final_counters
        completeness = bundle.summary.counter_completeness
        self.assertFalse(
            json.loads((run_dir / "nightly_run_summary.json").read_text())[
                "llm_enabled"
            ]
        )
        for axis in (
            "llm_call_attempted",
            "llm_call_completed",
            "prompt_token",
            "completion_token",
            "token",
            "llm_latency",
        ):
            self.assertEqual(completeness[axis], "unavailable")
        self.assertIsNone(counters.llm_attempts)
        self.assertIsNone(counters.llm_completions)
        self.assertIsNone(counters.prompt_tokens)
        self.assertIsNone(counters.completion_tokens)
        self.assertIsNone(counters.total_tokens)
        self.assertIsNone(counters.llm_latency_seconds)
        self.assertEqual(_snapshot_tree(run_dir), before)

    def test_ca_disabled_llm_counters_are_unavailable_not_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            shutil.copytree(FIXTURES / "ca_partial", run_dir)
            summary_path = run_dir / "nightly_run_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["llm_enabled"] = False
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            manifest = _run_manifest(
                "ca",
                seed=0,
                selector="min_fitness_frontier",
                generator="llm",
            )

            bundle = CaNormalizationAdapter().normalize(
                manifest,
                NativeRunInputs(run_dir),
            )

            counters = bundle.summary.final_counters
            completeness = bundle.summary.counter_completeness
            for axis in (
                "llm_call_attempted",
                "llm_call_completed",
                "prompt_token",
                "completion_token",
                "token",
                "llm_latency",
            ):
                self.assertEqual(completeness[axis], "unavailable")
            self.assertIsNone(counters.llm_attempts)
            self.assertIsNone(counters.llm_completions)
            self.assertIsNone(counters.prompt_tokens)
            self.assertIsNone(counters.completion_tokens)
            self.assertIsNone(counters.total_tokens)
            self.assertIsNone(counters.llm_latency_seconds)

    def test_maze_metric_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            shutil.copytree(FIXTURES / "maze_summary_only", run_dir)
            summary_path = run_dir / "nightly_run_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["qd_score"] = 99.0
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            manifest = _run_manifest(
                "maze",
                seed=0,
                selector="uniform_frontier",
                generator="genetic",
                gate="filter",
            )
            with self.assertRaisesRegex(NormalizationError, "QD-score mismatch"):
                MazeNormalizationAdapter().normalize(
                    manifest,
                    NativeRunInputs(run_dir),
                )

    def test_missing_required_native_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _run_manifest(
                "ca",
                seed=0,
                selector="min_fitness_frontier",
                generator="random",
            )
            with self.assertRaisesRegex(NormalizationError, "missing"):
                CaNormalizationAdapter().normalize(
                    manifest,
                    NativeRunInputs(Path(tmp)),
                )

    def test_all_current_domains_publish_capabilities(self) -> None:
        declarations = current_domain_capabilities()
        self.assertEqual(set(declarations), {"ca", "maze", "dungeon", "sphere"})
        self.assertEqual(declarations["dungeon"].selectors, ("uniform_frontier",))
        self.assertTrue(declarations["ca"].supports_warm_start)
        self.assertFalse(declarations["maze"].supports_warm_start)
        self.assertEqual(declarations["sphere"].native_fitness_max, 100.0)


class TestNormalizationSmokeRuns(unittest.TestCase):
    def test_ca_local_smoke_normalizes_without_metric_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            env = {
                "LIFEMANIFOLD_PROPOSAL_LOG_ALL_EMITTERS": "1",
                "LIFEMANIFOLD_LLM_CALL_LOG": "0",
            }
            with patch.dict(os.environ, env, clear=False):
                native = run_map_elites_nightly(
                    scheduler_path=DEFAULT_MINI_SCHEDULER_PATH,
                    output_dir=run_dir,
                    seed=7,
                    grid_size=8,
                    steps=200,
                    iterations=1,
                )
            before = _snapshot_tree(run_dir)
            manifest = _run_manifest(
                "ca",
                seed=7,
                selector="min_fitness_frontier",
                generator="random",
            )

            bundle = CaNormalizationAdapter().normalize(
                manifest,
                NativeRunInputs(run_dir),
            )

            self.assertEqual(
                bundle.summary.final_counters.evaluator_completions,
                native.evaluations,
            )
            self.assertEqual(
                bundle.summary.final_archive.occupied_cells,
                native.filled_cells,
            )
            self.assertAlmostEqual(
                bundle.summary.final_archive.coverage,
                native.coverage,
                places=6,
            )
            self.assertEqual(len(bundle.events), native.evaluations)
            self.assertEqual(bundle.summary.event_completeness, "partial")
            self.assertEqual(len(bundle.checkpoints), 1)
            self.assertEqual(bundle.checkpoints[0].indexed_by, "evaluation")
            self.assertEqual(_snapshot_tree(run_dir), before)

    def test_maze_local_smoke_normalizes_without_metric_drift(self) -> None:
        config = MazeSchedulerConfig(
            condition="genetic",
            iterations=2,
            batch_size=5,
            archive_resolution=8,
            initial_random_candidates=5,
            emitters=("random", "random", "genetic", "genetic", "genetic"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            native = run_maze_qd(config, seed=3, output_dir=run_dir)
            before = _snapshot_tree(run_dir)
            manifest = _run_manifest(
                "maze",
                seed=3,
                selector="uniform_frontier",
                generator="genetic",
            )

            bundle = MazeNormalizationAdapter().normalize(
                manifest,
                NativeRunInputs(run_dir),
            )

            self.assertEqual(
                bundle.summary.final_counters.proposal_slots,
                native.proposals,
            )
            self.assertEqual(
                bundle.summary.final_counters.evaluator_completions,
                native.evaluations,
            )
            self.assertEqual(
                bundle.summary.final_archive.occupied_cells,
                native.filled_cells,
            )
            self.assertAlmostEqual(
                bundle.summary.final_archive.raw_qd_score or 0.0,
                native.qd_score,
            )
            self.assertEqual(bundle.summary.event_completeness, "summary_only")
            self.assertEqual(_snapshot_tree(run_dir), before)


if __name__ == "__main__":
    unittest.main()

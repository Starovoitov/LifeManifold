"""Executable closing guards for Phase 1 acceptance criteria."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from typing import cast

from worldspace.attribution import (
    ArchiveState,
    AttributionAdmissionError,
    ProspectiveEventCapture,
    RunSummary,
    StudyManifest,
    admit_analysis_cohort,
    archive_fingerprint,
    build_factorial_job_plan,
    ca_genotype_hash,
    current_domain_capabilities,
    maze_genotype_hash,
    reconcile_event_ledger,
)
from worldspace.attribution.adapters import (
    CaNormalizationAdapter,
    MazeNormalizationAdapter,
    NativeRunInputs,
)
from worldspace.attribution.schemas import attribution_schema_bundle
from worldspace.mazes.spec import MazeSpec
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec

from tests.test_attribution_harness import _summary
from tests.test_attribution_job_builder import (
    _block_a_study,
    _capabilities,
    _context_for,
)
from tests.test_attribution_normalizers import FIXTURES, _run_manifest

ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TestPhase1Acceptance(unittest.TestCase):
    def test_authoritative_genotype_and_archive_fingerprint_goldens(self) -> None:
        ca = WorldSpec(
            birth=[1, 3],
            survival=[2, 3],
            noise=0.02,
            resource_regen=0.05,
            predation=0.1,
            cell_types=CANONICAL_CELL_TYPES.copy(),
            grid_size=8,
            steps=200,
            seed=123,
        )
        maze = MazeSpec(
            rows=(
                "#" * 16,
                "#S............G#",
                *("#..............#" for _ in range(13)),
                "#" * 16,
            )
        )
        ca_hash = ca_genotype_hash(ca)
        maze_hash = maze_genotype_hash(maze)
        self.assertEqual(
            ca_hash,
            "66d557c4cc240b59d3756a14e713471cf6067ed8f3535855b7b1db7adaf9186a",
        )
        self.assertEqual(
            maze_hash,
            "6c660f36a019984d2874541dd1a89d862b2393e3ac8d1b1ff39df4883f9e311c",
        )
        ca.seed = 999
        self.assertEqual(ca_genotype_hash(ca), ca_hash)
        entries = [
            {
                "cell_id": 2,
                "genotype_hash": maze_hash,
                "descriptors": {"x": 0.2, "y": 0.3},
                "fitness": 0.5,
            },
            {
                "cell_id": 1,
                "genotype_hash": ca_hash,
                "descriptors": {"x": 0.1, "y": 0.4},
                "fitness": 0.7,
            },
        ]
        expected_archive_hash = (
            "06f09613684c2a7fbd50511c028545c51473ddd73a4f857c175b7619710b0f93"
        )
        self.assertEqual(
            archive_fingerprint(entries, evaluator_hash="a" * 64),
            expected_archive_hash,
        )
        self.assertEqual(
            archive_fingerprint(reversed(entries), evaluator_hash="a" * 64),
            expected_archive_hash,
        )

    def test_raw_candidate_event_cannot_enter_primary_analysis(self) -> None:
        manifest = _run_manifest(
            "ca",
            seed=0,
            selector="min_fitness_frontier",
            generator="random",
        )
        state = ArchiveState(
            occupied_cells=0,
            capacity=4,
            coverage=0.0,
            raw_qd_score=0.0,
            normalized_qd_score=0.0,
            maximum_elite_quality=None,
            occupied_mean_quality=None,
        )
        event = ProspectiveEventCapture(manifest).append_slot(
            iteration=0,
            slot=0,
            configured_operator="random",
            realized_operator="random",
            target_cell_id="0",
            parent_id=None,
            parent_genotype_hash=None,
            candidate_id="candidate",
            candidate_genotype_hash="a" * 64,
            before=state,
            generation={
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
            gate={
                "mode": "filter",
                "decision": "skip",
                "reason": "fixture",
                "policy_version": "1",
            },
            evaluation={
                "attempted": False,
                "completed": False,
                "evaluator_seed": None,
                "fitness": None,
                "descriptors": None,
                "realized_cell_id": None,
                "incumbent_fitness": None,
                "insertion": "not_evaluated",
                "delta_qd": 0.0,
            },
            resources={
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
            },
            after=state,
        )

        with self.assertRaises(AttributionAdmissionError) as caught:
            admit_analysis_cohort(
                cast(tuple, (event,)),
                expected_arm_ids=(manifest.arm_id,),
                minimum_complete_pairs=1,
            )
        self.assertIn("cohort.invalid_record_type", str(caught.exception))

    def test_historical_summary_only_run_fails_exact_ledger_gate(self) -> None:
        manifest = _run_manifest(
            "maze",
            seed=0,
            selector="uniform_frontier",
            generator="genetic",
            gate="filter",
        )
        bundle = MazeNormalizationAdapter().normalize(
            manifest,
            NativeRunInputs(FIXTURES / "maze_summary_only"),
        )
        with self.assertRaisesRegex(
            ValueError,
            "event_completeness='full'",
        ):
            reconcile_event_ledger(
                bundle.events,
                bundle.summary,
                llm_applicable=False,
            )

    def test_analysis_rejects_mixed_protocol_domain_and_manifest(self) -> None:
        control = _summary(run_id="a", arm_id="control", pair_id="pair")
        for field_name, replacement, expected_code in (
            ("domain_id", "maze", "cohort.mixed_domain_id"),
            ("protocol_hash", "c" * 64, "cohort.mixed_protocol_hash"),
            (
                "study_manifest_hash",
                "d" * 64,
                "cohort.mixed_study_manifest_hash",
            ),
        ):
            with self.subTest(field_name=field_name):
                payload = _summary(
                    run_id="b",
                    arm_id="treatment",
                    pair_id="pair",
                ).model_dump(mode="json")
                payload[field_name] = replacement
                treatment = RunSummary.model_validate(payload)
                with self.assertRaises(AttributionAdmissionError) as caught:
                    admit_analysis_cohort(
                        (control, treatment),
                        expected_arm_ids=("control", "treatment"),
                        minimum_complete_pairs=1,
                    )
                self.assertIn(expected_code, str(caught.exception))

    def test_phase1_tooling_leaves_q1_configs_and_summaries_byte_unchanged(
        self,
    ) -> None:
        schedulers = sorted((ROOT / "worldspace" / "specs").glob("*scheduler*.yaml"))
        summaries = sorted(
            (ROOT / "artifacts" / "experiments").glob("**/nightly_run_summary.json")
        )
        self.assertTrue(schedulers)
        self.assertTrue(summaries)
        protected = tuple(schedulers + summaries)
        before = {path: _digest(path) for path in protected}

        attribution_schema_bundle()
        current_domain_capabilities()
        study = StudyManifest.model_validate(_block_a_study())
        plan = build_factorial_job_plan(
            study,
            _capabilities("ca"),
            _context_for(study),
        )
        self.assertFalse(plan.launched)
        CaNormalizationAdapter().normalize(
            _run_manifest(
                "ca",
                seed=0,
                selector="min_fitness_frontier",
                generator="llm",
            ),
            NativeRunInputs(FIXTURES / "ca_partial"),
        )

        after = {path: _digest(path) for path in protected}
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()

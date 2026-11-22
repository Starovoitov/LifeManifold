"""Confirmatory launcher stays dry until the explicit ack."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldspace.attribution.confirmatory_launch import (
    BLOCK_D_ACK,
    EXECUTE_ACK,
    REGISTERED_RESPONSE_PIN,
    ROBUSTNESS_ACK,
    ConfirmatoryLaunchError,
    build_block_d_plan,
    build_confirmatory_plan,
    build_robustness_plan,
    launch_block_d_plan,
    launch_plan,
    launch_robustness_plan,
    ordered_runs,
    public_config_from_block_d_run,
    public_config_from_robustness_run,
    public_config_from_run,
    run_disposition,
)
from worldspace.attribution.confirmatory_studies import confirmatory_bin_edges
from worldspace.attribution.manifest import RunManifest
from worldspace.attribution.public_loop import SUMMARY_FILENAME, floor_recipe_hash, write_json


class TestConfirmatoryLaunch(unittest.TestCase):
    def test_dry_run_interleaves_arms_inside_each_seed(self) -> None:
        plan = build_confirmatory_plan(
            "nas201",
            "selector-channel",
            output_root="artifacts/phase4",
        )
        ordered = ordered_runs(plan)
        self.assertEqual(len(ordered), 80)
        self.assertFalse(plan.launched)
        arm_ids = {run.arm_id for run in ordered}
        self.assertEqual(
            arm_ids,
            {
                "A-const-uniform",
                "A-live-uniform",
                "A-const-minfit",
                "A-live-minfit",
            },
        )
        for offset in range(0, len(ordered), 4):
            chunk = ordered[offset : offset + 4]
            self.assertEqual({run.seed for run in chunk}, {chunk[0].seed})
            self.assertEqual({run.arm_id for run in chunk}, arm_ids)
        seeds = [run.seed for run in ordered]
        self.assertEqual(seeds, sorted(seeds))
        rows = launch_plan(
            plan,
            execute=False,
            ack=None,
            output_root="artifacts/phase4",
        )
        self.assertEqual({row["action"] for row in rows}, {"dry-run"})

    def test_execute_requires_the_ack(self) -> None:
        plan = build_confirmatory_plan(
            "nas201",
            "selector-channel",
            output_root="artifacts/phase4-ack",
        )
        with self.assertRaises(ConfirmatoryLaunchError):
            launch_plan(
                plan,
                execute=True,
                ack=None,
                output_root="artifacts/phase4-ack",
                executor=lambda run: None,
            )

    def test_config_uses_the_registered_budget_and_response_pin(self) -> None:
        plan = build_confirmatory_plan(
            "nas201",
            "selector-channel",
            output_root="artifacts/phase4-config",
        )
        config = public_config_from_run(ordered_runs(plan)[0])
        self.assertEqual(config.floor_random, 20)
        self.assertEqual(config.search_horizon, 200)
        self.assertEqual(config.llm_call_cap, 200)
        self.assertEqual(config.repair_kind, "identity")
        self.assertEqual(config.generator, "llm")
        self.assertEqual(config.expected_response_model, REGISTERED_RESPONSE_PIN)

    def test_execute_skips_complete_and_stops_on_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = str(Path(tmp) / "phase4")
            plan = build_confirmatory_plan(
                "nas201",
                "selector-channel",
                output_root=root,
            )
            first = ordered_runs(plan)[0]
            partial = Path(first.output_paths["run_dir"])
            partial.mkdir(parents=True)
            (partial / "archive_trace.jsonl").write_text("{}\n", encoding="utf-8")
            called: list[str] = []
            with self.assertRaises(ConfirmatoryLaunchError):
                launch_plan(
                    plan,
                    execute=True,
                    ack=EXECUTE_ACK,
                    output_root=root,
                    executor=lambda run: called.append(run.run_id),
                )
            self.assertEqual(called, [])
            self.assertEqual(run_disposition(first), "incomplete")

    def test_execute_runs_pending_and_writes_the_clock_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "phase4"
            plan = build_confirmatory_plan(
                "pcg_sokoban",
                "generator",
                output_root=str(root),
            )
            pending = [
                run
                for run in ordered_runs(plan)
                if run.treatment.generator.kind == "llm"
            ][:1]
            genetic = next(
                run
                for run in ordered_runs(plan)
                if run.treatment.generator.kind == "genetic"
            )

            def _finish(run: RunManifest) -> None:
                write_json(
                    Path(run.output_paths["run_dir"]) / SUMMARY_FILENAME,
                    {"completed": True, "seed": run.seed},
                )

            _finish(genetic)
            rows = launch_plan(
                plan,
                execute=True,
                ack=EXECUTE_ACK,
                output_root=str(root),
                executor=_finish,
            )
            actions = {row["run_id"]: row["action"] for row in rows}
            self.assertEqual(actions[genetic.run_id], "skipped-complete")
            self.assertEqual(actions[pending[0].run_id], "executed")
            clock_path = root / "confirmatory_clock.json"
            first_clock = clock_path.read_text(encoding="utf-8")
            self.assertIn(pending[0].run_id, first_clock)
            launch_plan(
                plan,
                execute=True,
                ack=EXECUTE_ACK,
                output_root=str(root),
                executor=_finish,
            )
            self.assertEqual(clock_path.read_text(encoding="utf-8"), first_clock)
            payload = json.loads(first_clock)
            self.assertEqual(payload["protocol_id"], "ca-phase3-v0")


class TestRobustnessLaunch(unittest.TestCase):
    def test_dry_run_is_nas_block_a_on_the_robustness_seeds(self) -> None:
        plan = build_robustness_plan(output_root="artifacts/phase4-robustness")
        ordered = ordered_runs(plan)
        self.assertEqual(len(ordered), 40)
        self.assertFalse(plan.launched)
        self.assertEqual(plan.evidence_tier, "robustness")
        self.assertEqual({run.domain_id for run in ordered}, {"nas201"})
        self.assertEqual(
            {run.arm_id for run in ordered},
            {"A-const-uniform", "A-live-uniform", "A-const-minfit", "A-live-minfit"},
        )
        self.assertEqual([run.seed for run in ordered], sorted(run.seed for run in ordered))
        self.assertEqual(ordered[0].seed, 303001)
        self.assertEqual(ordered[-1].seed, 303010)
        self.assertTrue(all(run.model.kind == "gpt-4o" for run in ordered))
        self.assertTrue(all(not run.model.parameters.get("pinned_response") for run in ordered))
        config = public_config_from_robustness_run(ordered[0])
        self.assertIsNone(config.expected_response_model)
        self.assertEqual(config.floor_random, 20)
        self.assertEqual(config.search_horizon, 200)
        with self.assertRaises(ConfirmatoryLaunchError):
            public_config_from_run(ordered[0])
        with self.assertRaises(ConfirmatoryLaunchError):
            build_confirmatory_plan("nas201", "robustness", output_root="artifacts/phase4")

    def test_primary_ack_does_not_start_robustness(self) -> None:
        plan = build_robustness_plan(output_root="artifacts/phase4-robustness-ack")
        with self.assertRaises(ConfirmatoryLaunchError):
            launch_robustness_plan(
                plan,
                execute=True,
                ack=EXECUTE_ACK,
                output_root="artifacts/phase4-robustness-ack",
                executor=lambda run: None,
            )

    def test_snapshot_outside_the_stamped_list_stops_the_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "phase4"
            plan = build_robustness_plan(output_root=str(root))
            called: list[str] = []

            def _finish(run):  # type: ignore[no-untyped-def]
                called.append(run.run_id)
                write_json(
                    Path(run.output_paths["run_dir"]) / SUMMARY_FILENAME,
                    {"completed": True, "response_model": "gpt-4o-mini-2024-07-18"},
                )

            with self.assertRaises(ConfirmatoryLaunchError):
                launch_robustness_plan(
                    plan,
                    execute=True,
                    ack=ROBUSTNESS_ACK,
                    output_root=str(root),
                    executor=_finish,
                )
            self.assertEqual(len(called), 1)

    def test_accepted_snapshot_is_kept_for_the_next_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "phase4"
            plan = build_robustness_plan(output_root=str(root))
            calls = {"n": 0}

            def _finish(run):  # type: ignore[no-untyped-def]
                calls["n"] += 1
                model = "gpt-4o-2024-08-06" if calls["n"] == 1 else "gpt-4o-2024-11-20"
                write_json(
                    Path(run.output_paths["run_dir"]) / SUMMARY_FILENAME,
                    {"completed": True, "response_model": model},
                )

            with self.assertRaises(ConfirmatoryLaunchError):
                launch_robustness_plan(
                    plan,
                    execute=True,
                    ack=ROBUSTNESS_ACK,
                    output_root=str(root),
                    executor=_finish,
                )
            self.assertEqual(calls["n"], 2)


class TestBlockDLaunch(unittest.TestCase):
    def test_dry_run_is_two_nas_floors(self) -> None:
        plan = build_block_d_plan(output_root="artifacts/phase4-block-d")
        ordered = ordered_runs(plan)
        self.assertEqual(len(ordered), 20)
        self.assertFalse(plan.launched)
        self.assertEqual(plan.evidence_tier, "sensitivity")
        self.assertEqual(plan.study_id, "sensitivity-nas201-occupancy")
        self.assertEqual({run.domain_id for run in ordered}, {"nas201"})
        self.assertEqual({run.arm_id for run in ordered}, {"D-floor-random", "D-floor-genetic"})
        self.assertEqual(ordered[0].seed, 304001)
        self.assertEqual(ordered[-1].seed, 304010)
        by_seed: dict[int, dict[str, object]] = {}
        for run in ordered:
            self.assertEqual(run.model.kind, "gpt-4o-mini")
            self.assertEqual(run.model.parameters.get("pinned_response"), REGISTERED_RESPONSE_PIN)
            self.assertEqual(run.treatment.generator.kind, "llm")
            self.assertEqual(run.treatment.selector.kind, "uniform_frontier")
            self.assertEqual(run.treatment.prompt_channel.kind, "constant")
            self.assertEqual(run.treatment.repair_fallback.kind, "identity")
            config = public_config_from_block_d_run(run)
            self.assertEqual(config.search_horizon, 200)
            self.assertEqual(config.llm_call_cap, 200)
            self.assertEqual(config.expected_response_model, REGISTERED_RESPONSE_PIN)
            by_seed.setdefault(run.seed, {})[run.arm_id] = (config, run.initial_archive_hash)
            with self.assertRaises(ConfirmatoryLaunchError):
                public_config_from_run(run)
        self.assertEqual(set(by_seed), set(range(304001, 304011)))
        for seed, arms in by_seed.items():
            random_config, random_hash = arms["D-floor-random"]
            genetic_config, genetic_hash = arms["D-floor-genetic"]
            self.assertEqual(random_config.floor_random, 20)
            self.assertEqual(random_config.floor_genetic, 0)
            self.assertEqual(genetic_config.floor_random, 0)
            self.assertEqual(genetic_config.floor_genetic, 50)
            self.assertNotEqual(random_hash, genetic_hash)
        contrast = plan.design.contrasts[0]
        self.assertEqual(contrast.treatment_arm_id, "D-floor-genetic")
        self.assertEqual(contrast.control_arm_id, "D-floor-random")
        self.assertTrue(contrast.controlled_axes_match)
        edges = confirmatory_bin_edges("nas201")
        random_arm = next(run for run in ordered if run.seed == 304001 and run.arm_id == "D-floor-random")
        genetic_arm = next(run for run in ordered if run.seed == 304001 and run.arm_id == "D-floor-genetic")
        self.assertEqual(
            random_arm.initial_archive_hash,
            floor_recipe_hash(seed=304001, floor_random=20, repair="identity", bin_edges=edges),
        )
        self.assertEqual(
            random_arm.initial_archive_hash,
            floor_recipe_hash(
                seed=304001,
                floor_random=20,
                repair="identity",
                bin_edges=edges,
                floor_genetic=0,
            ),
        )
        self.assertEqual(
            genetic_arm.initial_archive_hash,
            floor_recipe_hash(
                seed=304001,
                floor_random=0,
                repair="identity",
                bin_edges=edges,
                floor_genetic=50,
            ),
        )
        with self.assertRaises(ConfirmatoryLaunchError):
            build_confirmatory_plan("nas201", "occupancy", output_root="artifacts/phase4")

    def test_other_acks_do_not_start_block_d(self) -> None:
        plan = build_block_d_plan(output_root="artifacts/phase4-block-d-ack")
        for ack in (EXECUTE_ACK, ROBUSTNESS_ACK):
            with self.assertRaises(ConfirmatoryLaunchError):
                launch_block_d_plan(
                    plan,
                    execute=True,
                    ack=ack,
                    output_root="artifacts/phase4-block-d-ack",
                    executor=lambda run: None,
                )
        with self.assertRaises(ConfirmatoryLaunchError):
            launch_plan(
                plan,
                execute=True,
                ack=EXECUTE_ACK,
                output_root="artifacts/phase4-block-d-ack",
                executor=lambda run: None,
            )

    def test_wrong_snapshot_stops_the_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "phase4"
            plan = build_block_d_plan(output_root=str(root))
            called: list[str] = []

            def _finish(run):  # type: ignore[no-untyped-def]
                called.append(run.run_id)
                write_json(
                    Path(run.output_paths["run_dir"]) / SUMMARY_FILENAME,
                    {"completed": True, "response_model": "gpt-4o-2024-08-06"},
                )

            with self.assertRaises(ConfirmatoryLaunchError):
                launch_block_d_plan(
                    plan,
                    execute=True,
                    ack=BLOCK_D_ACK,
                    output_root=str(root),
                    executor=_finish,
                )
            self.assertEqual(len(called), 1)


if __name__ == "__main__":
    unittest.main()

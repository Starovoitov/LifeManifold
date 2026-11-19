"""Confirmatory launcher stays dry until the explicit ack."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldspace.attribution.confirmatory_launch import (
    EXECUTE_ACK,
    REGISTERED_RESPONSE_PIN,
    ConfirmatoryLaunchError,
    build_confirmatory_plan,
    launch_plan,
    ordered_runs,
    public_config_from_run,
    run_disposition,
)
from worldspace.attribution.manifest import RunManifest
from worldspace.attribution.public_loop import SUMMARY_FILENAME, write_json


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


if __name__ == "__main__":
    unittest.main()

"""Infrastructure tests for the clean contemporaneous RQ1 factorial."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import scripts.run_rq1_contemporaneous_factorial as runner
from worldspace.generators.llm_config import load_llm_config
from worldspace.illuminators.archive import (
    ArchiveElite,
    elite_to_archive_record,
    new_elite_metadata,
)
from worldspace.specs.spec import WorldSpec


def _world() -> WorldSpec:
    return WorldSpec(
        birth=[3],
        survival=[2, 3],
        noise=0.01,
        resource_regen=0.1,
        predation=0.1,
        cell_types=["life", "food"],
        grid_size=50,
        steps=200,
        seed=0,
    )


class Rq1ContemporaneousFactorialTests(unittest.TestCase):
    def test_plan_contains_five_archives_two_continuations_and_all_arms(self) -> None:
        plan = runner.build_plan()
        self.assertEqual(plan["archive_count"], 5)
        self.assertEqual(plan["continuations_per_archive"], 2)
        self.assertEqual(len(plan["blocks"]), 10)
        for block in plan["blocks"]:
            self.assertEqual(set(block["arm_order"]), set(runner.ARM_SCHEDULERS))

    def test_arm_order_is_deterministic_and_varies_by_block(self) -> None:
        first = runner.arm_order(0, 0)
        self.assertEqual(first, runner.arm_order(0, 0))
        self.assertNotEqual(first, runner.arm_order(0, 1))

    def test_fixed_spec_uses_dated_model_and_seed(self) -> None:
        config = load_llm_config(runner.FIXED_LLM_SPEC)
        provider = config.providers["openai"]
        self.assertEqual(provider["model"], "gpt-4o-mini-2024-07-18")
        self.assertEqual(provider["chat_extra"]["seed"], 20260813)

    def test_model_probe_refuses_without_explicit_execution_ack(self) -> None:
        with self.assertRaises(SystemExit):
            runner.main(["probe-model"])

    def test_materialize_floor_stops_at_971_unique_bins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            destination = Path(tmp) / "floor.jsonl"
            lines: list[str] = []
            for cell_id in range(972):
                i, j = divmod(cell_id, 50)
                elite = ArchiveElite(
                    bin=(i, j),
                    fitness=cell_id / 2500.0,
                    world_spec=_world(),
                    measures={"stability": i / 50.0, "diversity": j / 50.0},
                    metadata=new_elite_metadata(
                        generated_by="test",
                        emitter_type="random",
                    ),
                )
                lines.append(json.dumps(elite_to_archive_record(elite)))
            source.write_text("\n".join(lines) + "\n", encoding="utf-8")

            result = runner.materialize_floor(source, destination)

            self.assertEqual(result["filled_cells"], 971)
            self.assertEqual(result["prefix_records"], 971)
            self.assertEqual(
                len(destination.read_text(encoding="utf-8").splitlines()),
                971,
            )


if __name__ == "__main__":
    unittest.main()

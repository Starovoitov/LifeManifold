"""Sokoban prompt scan, LLM parse/fallback, copy-rate, and isolated-batch gates."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from worldspace.pcg.copy_audit import README_SOKOBAN_V0_GRIDS, copy_readme_example
from worldspace.pcg.descriptors import PcgBinEdges
from worldspace.pcg.emitters import random_spec
from worldspace.pcg.isolated import run_isolated_batch
from worldspace.pcg.llm_emitter import PcgSokobanLlmEmitter, parse_sokoban_response
from worldspace.pcg.prompt_scan import (
    SokobanPromptError,
    assert_prompt_templates,
    prompt_violations,
)
from worldspace.pcg.spec import PcgSpec, SOKOBAN_V0, hamming_tiles


class _Caller:
    def __init__(self, response: object) -> None:
        self.response = response
        self.prompts: list[str] = []

    def __call__(self, **kwargs: object) -> str:
        self.prompts.append(str(kwargs["prompt"]))
        if isinstance(self.response, BaseException):
            raise self.response
        return str(self.response)


class _ToyEnv:
    def __init__(self) -> None:
        self.calls = 0

    def quality(self, contents: object) -> tuple[float, float, dict[str, object]]:
        self.calls += 1
        grid = contents
        crates = sum(tile == 3 for row in grid for tile in row)
        players = sum(tile == 2 for row in grid for tile in row)
        info = {
            "players": players,
            "crates": crates,
            "targets": crates,
            "content": grid,
            "heuristic": -1,
            "solution": [],
        }
        quality = min(1.0, 0.05 * (players + 1))
        return 0.0, quality, info


def _parent() -> PcgSpec:
    return random_spec(SOKOBAN_V0, np.random.default_rng(7))


def _edges() -> PcgBinEdges:
    return PcgBinEdges(
        resolution=10,
        measure_names=("solution_length", "crates"),
        axis0_min=0.0,
        axis0_max=0.0,
        axis1_min=0.0,
        axis1_max=10.0,
        n_samples=275,
        problem_name="sokoban-v0",
    )


def _child_from_parent(parent: PcgSpec) -> PcgSpec:
    grid = [list(row) for row in parent.grid]
    grid[0][0] = 0 if grid[0][0] != 0 else 1
    return PcgSpec.from_task_grid(SOKOBAN_V0, grid)


class TestSokobanPromptScan(unittest.TestCase):
    def test_committed_templates_pass_prompt_scan(self) -> None:
        templates = assert_prompt_templates()
        self.assertIn("{parent_json}", templates["user"])
        self.assertEqual(prompt_violations(templates["system"]), [])
        self.assertEqual(prompt_violations(templates["user"]), [])
        self.assertNotIn("[[0,1,4,0,0]", templates["system"].replace(" ", ""))

    def test_prompt_scan_rejects_fitness_and_readme_grid(self) -> None:
        self.assertIn(
            "fitness_or_qd", prompt_violations("maximize fitness of the grid")
        )
        self.assertIn(
            "few_shot_language", prompt_violations("here is an example level")
        )
        example = json.dumps([list(row) for row in README_SOKOBAN_V0_GRIDS[0]])
        self.assertIn("few_shot_grid_example", prompt_violations(example))
        self.assertIn("readme_example_grid", prompt_violations(example))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.txt"
            path.write_text("few-shot Microban grid\n", encoding="utf-8")
            with self.assertRaises(SokobanPromptError):
                assert_prompt_templates(path, path)


class TestPcgLlmParse(unittest.TestCase):
    def test_parse_accepts_fenced_grid_object(self) -> None:
        parent = _parent()
        payload = json.dumps({"grid": parent.to_nested_list()})
        parsed = parse_sokoban_response(f"```json\n{payload}\n```")
        self.assertEqual(parsed, parent)

    def test_parse_accepts_bare_array(self) -> None:
        parent = _parent()
        parsed = parse_sokoban_response(parent.canonical_json())
        self.assertEqual(parsed, parent)

    def test_parse_rejects_extra_keys_and_shape(self) -> None:
        with self.assertRaises(ValueError):
            parse_sokoban_response(
                '{"grid":[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],"score":1}'
            )
        with self.assertRaises(ValueError):
            parse_sokoban_response(json.dumps({"grid": [[9] * 5] * 5}))
        with self.assertRaises(json.JSONDecodeError):
            parse_sokoban_response("not json at all")

    def test_parse_fail_uses_genetic_fallback_without_repair(self) -> None:
        parent = _parent()
        emitter = PcgSokobanLlmEmitter(
            call_llm_text=_Caller("not json at all"),
            max_retries=0,
        )
        rng = np.random.default_rng(3)
        proposal = emitter.emit(parent, rng, proposal_index=0)
        self.assertTrue(proposal.used_fallback)
        self.assertFalse(proposal.schema_valid)
        self.assertEqual(proposal.emitter_type, "llm_fallback_genetic")
        self.assertEqual(hamming_tiles(parent, proposal.child), 1)

    def test_valid_json_is_not_rewritten(self) -> None:
        parent = _parent()
        child = _child_from_parent(parent)
        emitter = PcgSokobanLlmEmitter(
            call_llm_text=_Caller(json.dumps({"grid": child.to_nested_list()})),
            max_retries=0,
        )
        proposal = emitter.emit(parent, np.random.default_rng(1), proposal_index=1)
        self.assertTrue(proposal.schema_valid)
        self.assertFalse(proposal.used_fallback)
        self.assertEqual(proposal.child, child)
        self.assertEqual(proposal.hamming, hamming_tiles(parent, child))

    def test_copy_readme_flag_without_few_shot_in_prompt(self) -> None:
        parent = _parent()
        example = PcgSpec.from_task_grid(SOKOBAN_V0, README_SOKOBAN_V0_GRIDS[0])
        self.assertTrue(copy_readme_example(example))
        emitter = PcgSokobanLlmEmitter(
            call_llm_text=_Caller(json.dumps({"grid": example.to_nested_list()})),
            max_retries=0,
        )
        proposal = emitter.emit(parent, np.random.default_rng(1), proposal_index=2)
        self.assertTrue(proposal.copy_readme_example)
        self.assertNotIn("microban", emitter.system_prompt.lower())
        self.assertNotIn("example level", "".join(emitter.user_prompt.lower()))

    def test_retry_tokens_count_only_the_successful_parse(self) -> None:
        parent = _parent()
        child = _child_from_parent(parent)

        class _RetryCaller:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, **kwargs: object) -> str:
                self.calls += 1
                if self.calls == 1:
                    return "not json at all"
                return json.dumps({"grid": child.to_nested_list()})

        usages_seen: list[str] = []

        def fake_usage(call_id: str) -> tuple[dict[str, int], str]:
            usages_seen.append(call_id)
            n = len(usages_seen)
            return (
                {
                    "prompt_tokens": 10 * n,
                    "completion_tokens": n,
                    "total_tokens": 11 * n,
                },
                "mock-model",
            )

        emitter = PcgSokobanLlmEmitter(call_llm_text=_RetryCaller(), max_retries=2)
        with (
            patch(
                "worldspace.pcg.llm_emitter._usage_and_model_from_call_id",
                fake_usage,
            ),
            patch("worldspace.pcg.llm_emitter.time.sleep"),
        ):
            proposal = emitter.emit(parent, np.random.default_rng(1), proposal_index=0)

        self.assertTrue(proposal.schema_valid)
        self.assertEqual(proposal.retries, 1)
        self.assertEqual(proposal.api_calls, 2)
        self.assertEqual(len(usages_seen), 1)
        self.assertEqual(proposal.prompt_tokens, 10)
        self.assertEqual(proposal.completion_tokens, 1)
        self.assertEqual(proposal.total_tokens, 11)


class TestPcgIsolatedGates(unittest.TestCase):
    def test_g3_g4_on_mock_batch(self) -> None:
        parent = _parent()
        child = _child_from_parent(parent)
        emitter = PcgSokobanLlmEmitter(
            call_llm_text=_Caller(json.dumps({"grid": child.to_nested_list()})),
            max_retries=0,
        )
        records, summary = run_isolated_batch(
            _ToyEnv(),
            _edges(),
            emitter,
            n_proposals=8,
            seed=201401,
        )
        self.assertEqual(len(records), 8)
        gates = summary["gates"]
        hamming = summary["mean_hamming_parse_valid"]
        if not isinstance(gates, dict):
            self.fail("isolated summary gates must be a dict")
        if not isinstance(hamming, float):
            self.fail("mean_hamming_parse_valid must be a float")
        self.assertTrue(gates["parse"])
        self.assertTrue(gates["emitter_not_fallback"])
        self.assertGreater(hamming, 0.0)
        self.assertEqual(summary["copy_readme"], 0)
        self.assertTrue(gates["zero_shot_prompt"])


if __name__ == "__main__":
    unittest.main()

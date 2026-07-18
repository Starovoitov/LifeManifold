"""Strict parser, prompt, fallback, and audit tests for dungeon LLM emission."""

from __future__ import annotations

import json
import unittest

import numpy as np

from worldspace.dungeons.archive import DungeonArchive, DungeonElite
from worldspace.dungeons.emitters import select_uniform_frontier
from worldspace.dungeons.genetics import random_dungeon
from worldspace.dungeons.llm_emitter import (
    DungeonLlmEmitter,
    parse_dungeon_response,
    parse_dungeon_responses_ordered,
    repair_solvable_mutation,
    tile_distance,
)
from worldspace.dungeons.spec import DungeonSpec
from worldspace.dungeons.surrogate import DungeonPrediction


class _Caller:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def __call__(self, **kwargs: object) -> str:
        self.prompts.append(str(kwargs["prompt"]))
        return self.response


class _ErrorCaller(_Caller):
    def __init__(self, message: str) -> None:
        super().__init__("")
        self.message = message

    def __call__(self, **kwargs: object) -> str:
        self.prompts.append(str(kwargs["prompt"]))
        raise RuntimeError(self.message)


class _SequenceCaller(_Caller):
    def __init__(self, responses: list[str]) -> None:
        super().__init__("")
        self.responses = iter(responses)

    def __call__(self, **kwargs: object) -> str:
        self.prompts.append(str(kwargs["prompt"]))
        item = next(self.responses)
        if isinstance(item, BaseException):
            raise item
        return str(item)


def _target() -> tuple[DungeonArchive, object]:
    rng = np.random.default_rng(4)
    spec = random_dungeon(rng)
    archive = DungeonArchive(10)
    elite = DungeonElite(
        bin=(5, 5),
        fitness=0.8,
        measures=(0.55, 0.55),
        spec=spec,
        candidate_id="parent",
        parent_id=None,
        emitter_type="random",
    )
    archive.try_insert(elite)
    return archive, select_uniform_frontier(archive, rng)


class TestDungeonLlmEmitter(unittest.TestCase):
    def test_strict_parser_accepts_fenced_json_and_rejects_extra_fields(self) -> None:
        spec = random_dungeon(np.random.default_rng(1))
        parsed = parse_dungeon_response(f"```json\n{spec.canonical_json()}\n```")
        self.assertEqual(parsed, spec)
        with self.assertRaises(ValueError):
            parse_dungeon_response(
                json.dumps({"rows": list(spec.rows), "explanation": "no"})
            )
        ordinary = [
            (row, column)
            for row in range(1, 15)
            for column in range(1, 15)
            if spec.rows[row][column] not in ("S", "G", "K", "D")
        ][:4]
        edits = [
            {
                "row": row,
                "col": column,
                "tile": "." if spec.rows[row][column] != "." else "#",
            }
            for row, column in ordinary
        ]
        patched = parse_dungeon_response(
            json.dumps({"edits": edits}),
            parent=spec,
        )
        self.assertGreater(tile_distance(spec, patched), 0)

    def test_hints_render_and_valid_child_audit(self) -> None:
        archive, target = _target()
        child = random_dungeon(np.random.default_rng(8))
        caller = _Caller(child.canonical_json())
        emitter = DungeonLlmEmitter(
            prompt_mode="hints",
            call_llm_text=caller,  # type: ignore[arg-type]
        )
        result = emitter.emit(
            target=target,  # type: ignore[arg-type]
            archive=archive,
            rng=np.random.default_rng(5),
            prediction=DungeonPrediction(
                {},
                {"path_length": 0.3, "branching": 0.7},
                0.8123,
                0.0456,
            ),
        )
        self.assertEqual(result.emitter_type, "llm")
        self.assertIn("0.8123", caller.prompts[0])
        self.assertIn("0.0456", caller.prompts[0])
        self.assertEqual(emitter.audit.parse_successes, 1)
        self.assertEqual(emitter.audit.fallbacks, 0)
        self.assertGreater(tile_distance(target.parent.spec, child), 0)  # type: ignore[union-attr]

    def test_invalid_response_uses_logged_genetic_fallback(self) -> None:
        archive, target = _target()
        emitter = DungeonLlmEmitter(
            prompt_mode="stub",
            call_llm_text=_Caller("not json"),  # type: ignore[arg-type]
        )
        result = emitter.emit(
            target=target,  # type: ignore[arg-type]
            archive=archive,
            rng=np.random.default_rng(6),
            prediction=None,
        )
        self.assertEqual(result.emitter_type, "llm_fallback_genetic")
        self.assertEqual(emitter.audit.fallbacks, 1)
        self.assertEqual(emitter.audit.api_calls, 2)
        self.assertEqual(emitter.audit.retries, 1)
        self.assertIsNotNone(result.spec)

    def test_retry_recovers_before_fallback(self) -> None:
        archive, target = _target()
        child = random_dungeon(np.random.default_rng(19))
        emitter = DungeonLlmEmitter(
            prompt_mode="stub",
            call_llm_text=_SequenceCaller(["bad", child.canonical_json()]),  # type: ignore[arg-type]
        )
        result = emitter.emit(
            target=target,  # type: ignore[arg-type]
            archive=archive,
            rng=np.random.default_rng(6),
            prediction=None,
        )
        self.assertEqual(result.emitter_type, "llm")
        self.assertEqual(emitter.audit.retries, 1)
        self.assertEqual(emitter.audit.fallbacks, 0)

    def test_transient_network_error_recovers_before_fallback(self) -> None:
        archive, target = _target()
        child = random_dungeon(np.random.default_rng(21))
        emitter = DungeonLlmEmitter(
            prompt_mode="stub",
            call_llm_text=_SequenceCaller(  # type: ignore[arg-type]
                [
                    RuntimeError(
                        "LLM request failed: [SSL: UNEXPECTED_EOF_WHILE_READING]"
                    ),
                    child.canonical_json(),
                ]
            ),
        )
        result = emitter.emit(
            target=target,  # type: ignore[arg-type]
            archive=archive,
            rng=np.random.default_rng(6),
            prediction=None,
        )
        self.assertEqual(result.emitter_type, "llm")
        self.assertEqual(emitter.audit.retries, 1)
        self.assertEqual(emitter.audit.fallbacks, 0)
        self.assertEqual(emitter.audit.invalid_response_reasons.get("network"), 1)

    def test_persistent_network_error_uses_genetic_fallback(self) -> None:
        archive, target = _target()
        emitter = DungeonLlmEmitter(
            prompt_mode="stub",
            call_llm_text=_ErrorCaller("LLM request failed: connection reset"),  # type: ignore[arg-type]
        )
        result = emitter.emit(
            target=target,  # type: ignore[arg-type]
            archive=archive,
            rng=np.random.default_rng(6),
            prediction=None,
        )
        self.assertEqual(result.emitter_type, "llm_fallback_genetic")
        self.assertEqual(emitter.audit.fallbacks, 1)
        self.assertEqual(emitter.audit.failure_reasons.get("network"), 1)

    def test_unsolvable_patch_can_retain_a_solvable_subset(self) -> None:
        parent_rows = ["#" * 16, "#S............G#"] + ["#" * 16] * 14
        parent = DungeonSpec(rows=tuple(parent_rows))
        proposed_rows = list(parent.rows)
        row = list(proposed_rows[1])
        row[2], row[3], row[4] = "#", "#", "H"
        proposed_rows[1] = "".join(row)
        proposed = DungeonSpec(rows=tuple(proposed_rows))
        repaired = repair_solvable_mutation(parent, proposed)
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertGreater(tile_distance(parent, repaired), 0)

    def test_prompt_hash_is_reproducible(self) -> None:
        first = DungeonLlmEmitter(prompt_mode="stub", call_llm_text=_Caller("{}"))  # type: ignore[arg-type]
        second = DungeonLlmEmitter(prompt_mode="stub", call_llm_text=_Caller("{}"))  # type: ignore[arg-type]
        self.assertEqual(first.prompt_version, second.prompt_version)

    def test_parallel_parser_preserves_slot_order(self) -> None:
        parents = [random_dungeon(np.random.default_rng(seed)) for seed in (11, 12, 13)]
        parsed = parse_dungeon_responses_ordered(
            [parent.canonical_json() for parent in parents],
            parents,
            max_workers=3,
        )
        self.assertEqual(parsed, parents)


if __name__ == "__main__":
    unittest.main()

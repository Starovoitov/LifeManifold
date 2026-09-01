"""NAS prompt scan, LLM parse/fallback, and isolated-batch gates."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from worldspace.nas201.descriptors import Nas201BinEdges, load_frozen_bin_edges
from worldspace.nas201.isolated import run_isolated_batch
from worldspace.nas201.llm_emitter import Nas201LlmEmitter, parse_nas201_response
from worldspace.nas201.prompt_scan import (
    Nas201PromptError,
    assert_prompt_templates,
    prompt_violations,
)
from worldspace.nas201.spec import Nas201Spec, hamming_ops
from worldspace.nas201.table import Nas201SearchRecord


class _Caller:
    def __init__(self, response: object) -> None:
        self.response = response
        self.prompts: list[str] = []

    def __call__(self, **kwargs: object) -> str:
        self.prompts.append(str(kwargs["prompt"]))
        if isinstance(self.response, BaseException):
            raise self.response
        return str(self.response)


def _parent() -> Nas201Spec:
    return Nas201Spec(
        ops=(
            "none",
            "skip_connect",
            "nor_conv_1x1",
            "nor_conv_3x3",
            "avg_pool_3x3",
            "none",
        )
    )


def _edges() -> Nas201BinEdges:
    return Nas201BinEdges(
        resolution=20,
        log_params_min=-2.0,
        log_params_max=1.0,
        log_flops_min=0.0,
        log_flops_max=3.0,
        n_architectures=8,
        source_sha256="a" * 64,
    )


class TestNas201PromptScan(unittest.TestCase):
    def test_committed_templates_pass_prompt_scan(self) -> None:
        templates = assert_prompt_templates()
        self.assertIn("{parent_json}", templates["user"])
        self.assertEqual(prompt_violations(templates["system"]), [])
        self.assertEqual(prompt_violations(templates["user"]), [])

    def test_prompt_scan_rejects_accuracy_and_example_cell(self) -> None:
        self.assertIn("accuracy", prompt_violations("maximize accuracy of the cell"))
        self.assertIn("dataset_goal", prompt_violations("optimize CIFAR-10"))
        self.assertIn("leaderboard", prompt_violations("beat the leaderboard"))
        example = (
            '["nor_conv_3x3","nor_conv_3x3","avg_pool_3x3",'
            '"skip_connect","nor_conv_3x3","skip_connect"]'
        )
        self.assertIn("few_shot_cell_example", prompt_violations(example))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.txt"
            path.write_text("optimize CIFAR accuracy\n", encoding="utf-8")
            with self.assertRaises(Nas201PromptError):
                assert_prompt_templates(path, path)


class TestNas201LlmParse(unittest.TestCase):
    def test_parse_accepts_fenced_ops_json(self) -> None:
        parent = _parent()
        payload = parent.canonical_json()
        parsed = parse_nas201_response(f"```json\n{payload}\n```")
        self.assertEqual(parsed, parent)

    def test_parse_rejects_arch_string_and_extra_keys(self) -> None:
        parent = _parent()
        with self.assertRaises(json.JSONDecodeError):
            parse_nas201_response(parent.arch_str)
        with self.assertRaises(ValueError):
            parse_nas201_response(
                '{"ops":["none","none","none","none","none","none"],"score":90}'
            )
        with self.assertRaises(ValueError):
            parse_nas201_response(json.dumps({"ops": ["conv"] * 6}))

    def test_parse_fail_uses_genetic_fallback_without_repair(self) -> None:
        parent = _parent()
        emitter = Nas201LlmEmitter(
            call_llm_text=_Caller("not json at all"),
            max_retries=0,
        )
        rng = np.random.default_rng(3)
        proposal = emitter.emit(parent, rng, proposal_index=0)
        self.assertTrue(proposal.used_fallback)
        self.assertFalse(proposal.schema_valid)
        self.assertEqual(proposal.emitter_type, "llm_fallback_genetic")
        self.assertEqual(hamming_ops(parent, proposal.child), 1)

    def test_valid_json_is_not_rewritten(self) -> None:
        parent = _parent()
        child = Nas201Spec(ops=("skip_connect",) + parent.ops[1:])
        emitter = Nas201LlmEmitter(
            call_llm_text=_Caller(child.canonical_json()),
            max_retries=0,
        )
        proposal = emitter.emit(parent, np.random.default_rng(1), proposal_index=1)
        self.assertTrue(proposal.schema_valid)
        self.assertFalse(proposal.used_fallback)
        self.assertEqual(proposal.child, child)
        self.assertEqual(proposal.hamming, 1)

    def test_retry_tokens_count_only_the_successful_parse(self) -> None:
        parent = _parent()
        child = Nas201Spec(ops=("skip_connect",) + parent.ops[1:])

        class _RetryCaller:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, **kwargs: object) -> str:
                self.calls += 1
                if self.calls == 1:
                    return "not json at all"
                return child.canonical_json()

        usages_seen: list[str] = []

        def fake_usage(call_id: str) -> dict[str, int]:
            usages_seen.append(call_id)
            n = len(usages_seen)
            return {
                "prompt_tokens": 10 * n,
                "completion_tokens": n,
                "total_tokens": 11 * n,
            }

        emitter = Nas201LlmEmitter(call_llm_text=_RetryCaller(), max_retries=2)
        with (
            patch("worldspace.nas201.llm_emitter._usage_from_call_id", fake_usage),
            patch("worldspace.nas201.llm_emitter.time.sleep"),
        ):
            proposal = emitter.emit(parent, np.random.default_rng(1), proposal_index=0)

        self.assertTrue(proposal.schema_valid)
        self.assertEqual(proposal.retries, 1)
        self.assertEqual(proposal.api_calls, 2)
        self.assertEqual(len(usages_seen), 1)
        self.assertEqual(proposal.prompt_tokens, 10)
        self.assertEqual(proposal.completion_tokens, 1)
        self.assertEqual(proposal.total_tokens, 11)


class TestNas201IsolatedGates(unittest.TestCase):
    def test_g3_g4_on_mock_batch(self) -> None:
        parent_ops = (
            "none",
            "skip_connect",
            "nor_conv_1x1",
            "nor_conv_3x3",
            "avg_pool_3x3",
            "none",
        )
        child = Nas201Spec(ops=("skip_connect",) + parent_ops[1:])
        emitter = Nas201LlmEmitter(
            call_llm_text=_Caller(child.canonical_json()),
            max_retries=0,
        )

        class _AllHit:
            def lookup_search(self, arch_str: str) -> Nas201SearchRecord | None:
                spec = Nas201Spec.from_arch_str(arch_str)
                return Nas201SearchRecord(
                    index=0,
                    arch=spec.arch_str,
                    flops=12.0,
                    params=0.2,
                    latency=None,
                    valid_accuracy=70.0,
                    n_trials=3,
                )

            def __len__(self) -> int:
                return 15625

        records, summary = run_isolated_batch(
            _AllHit(),
            _edges(),
            emitter,
            n_proposals=8,
            seed=201101,
        )
        self.assertEqual(len(records), 8)
        self.assertTrue(summary["gates"]["parse"])
        self.assertTrue(summary["gates"]["emitter_not_fallback"])
        self.assertGreater(summary["mean_hamming_parse_valid"], 0.0)

    def test_frozen_edges_loader_does_not_recompute(self) -> None:
        payload = {
            "resolution": 20,
            "log_params_min": -1.1,
            "log_params_max": 0.2,
            "log_flops_min": 0.8,
            "log_flops_max": 2.3,
            "n_architectures": 15625,
            "nats_meta_sha256": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edges.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            edges = load_frozen_bin_edges(path)
        self.assertEqual(edges.log_params_min, -1.1)
        self.assertEqual(edges.source_sha256, "b" * 64)


if __name__ == "__main__":
    unittest.main()

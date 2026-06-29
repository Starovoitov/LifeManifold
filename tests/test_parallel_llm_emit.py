"""Tests for parallel LLM HTTP batching."""

from __future__ import annotations

import time
import unittest

from worldspace.illuminators.emitters.llm_emitter import LlmEmitter, LlmPreparedSlot
from worldspace.illuminators.parallel_llm_emit import (
    ParallelLlmPool,
    request_llm_batch,
)
from worldspace.illuminators.scheduler import TargetCell
from worldspace.specs.spec import WorldSpec

_PREPARED = LlmPreparedSlot(
    target=TargetCell(
        cell_id=0,
        target_stability=0.5,
        target_diversity=0.5,
        bin_ij=(0, 0),
    ),
    parent_spec=WorldSpec(
        birth=[1],
        survival=[2],
        noise=0.02,
        resource_regen=0.05,
        predation=0.1,
        cell_types=["life", "food"],
        grid_size=8,
        steps=200,
        seed=0,
    ),
    parent_id=None,
    system_prompt="sys",
    user_prompt="user",
    prompt_version="abc:def",
    grid_size=8,
    steps=200,
)


class TestParallelLlmEmitModule(unittest.TestCase):
    def test_request_llm_batch_reuses_pool(self) -> None:
        calls = 0

        class _Llm:
            def request_llm(self, prepared: LlmPreparedSlot) -> str:
                del prepared
                nonlocal calls
                calls += 1
                time.sleep(0.03)
                return "{}"

        llm = _Llm()
        pool = ParallelLlmPool(2)
        try:
            t0 = time.monotonic()
            request_llm_batch(
                llm,  # type: ignore[arg-type]
                [_PREPARED, _PREPARED],
                max_workers=2,
                llm_pool=pool,
            )
            parallel_s = time.monotonic() - t0

            t1 = time.monotonic()
            request_llm_batch(
                llm,  # type: ignore[arg-type]
                [_PREPARED, _PREPARED],
                max_workers=1,
                llm_pool=None,
            )
            sequential_s = time.monotonic() - t1
        finally:
            pool.shutdown()

        self.assertEqual(calls, 4)
        self.assertLess(parallel_s, sequential_s * 0.85)

    def test_request_llm_batch_empty_on_failure(self) -> None:
        emitter = LlmEmitter(
            grid_resolution=5,
            call_llm_text=lambda **_: (_ for _ in ()).throw(RuntimeError("fail")),
        )
        out = request_llm_batch(emitter, [_PREPARED], max_workers=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].response, "")
        self.assertIsInstance(out[0].request_error, RuntimeError)
        self.assertEqual(str(out[0].request_error), "fail")


if __name__ == "__main__":
    unittest.main()

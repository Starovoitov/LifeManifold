"""Tests for surrogate checkpoint path helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from worldspace.surrogate.checkpoint_paths import (
    STUB_CHECKPOINT_SENTINEL,
    is_stub_checkpoint,
    resolve_runtime_checkpoint_path,
)


class TestCheckpointPaths(unittest.TestCase):
    def test_stub_sentinel_is_detected(self) -> None:
        self.assertTrue(is_stub_checkpoint(STUB_CHECKPOINT_SENTINEL))
        self.assertFalse(
            is_stub_checkpoint("artifacts/surrogate/checkpoints/nightly_v2.pkl")
        )

    def test_resolve_runtime_checkpoint_path_rejects_stub(self) -> None:
        self.assertIsNone(resolve_runtime_checkpoint_path(STUB_CHECKPOINT_SENTINEL))
        self.assertIsNone(resolve_runtime_checkpoint_path(None))
        self.assertIsNone(resolve_runtime_checkpoint_path(""))

    def test_resolve_runtime_checkpoint_path_expands_real_paths(self) -> None:
        resolved = resolve_runtime_checkpoint_path(
            "artifacts/surrogate/checkpoints/micro.pkl"
        )
        self.assertEqual(resolved, Path("artifacts/surrogate/checkpoints/micro.pkl"))


if __name__ == "__main__":
    unittest.main()

"""Tests for GitHub LLM MAP-Elites scheduler and provider resolution."""

from __future__ import annotations

import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


class TestGithubLlmMapElites(unittest.TestCase):
    def test_github_scheduler_loads_with_llm_and_surrogate(self) -> None:
        from worldspace.illuminators.scheduler import (
            DEFAULT_GITHUB_LLM_SCHEDULER_PATH,
            load_scheduler,
        )

        config = load_scheduler(DEFAULT_GITHUB_LLM_SCHEDULER_PATH)
        self.assertTrue(config.llm_enabled)
        self.assertTrue(config.surrogate_enabled)
        self.assertIn("nightly.pkl", config.surrogate_checkpoint or "")

    def test_resolve_llm_spec_qwen(self) -> None:
        from scripts.run_github_llm_map_elites import resolve_llm_spec_path
        from worldspace.illuminators.scheduler import DEFAULT_QWEN_LLM_SPEC_PATH

        self.assertEqual(resolve_llm_spec_path("qwen"), DEFAULT_QWEN_LLM_SPEC_PATH)
        self.assertTrue(DEFAULT_QWEN_LLM_SPEC_PATH.is_file())

    def test_qwen_yaml_uses_qwen_provider(self) -> None:
        from worldspace.generators.llm_config import load_llm_config
        from worldspace.illuminators.scheduler import DEFAULT_QWEN_LLM_SPEC_PATH

        cfg = load_llm_config(DEFAULT_QWEN_LLM_SPEC_PATH)
        self.assertEqual(cfg.active_provider, "qwen")


if __name__ == "__main__":
    unittest.main()

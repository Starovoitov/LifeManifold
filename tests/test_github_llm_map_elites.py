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
        self.assertEqual(config.iterations, 650)
        self.assertEqual(config.batch_size, 50)
        self.assertEqual(config.batch_emitters.count("llm"), 20)

    def test_resolve_nightly_grid_resolution_from_summary(self) -> None:
        import json
        import tempfile

        from scripts.run_github_llm_map_elites import resolve_nightly_grid_resolution

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            archive = run_dir / "map_elites_archive.jsonl"
            archive.write_text("{}\n", encoding="utf-8")
            summary = run_dir / "nightly_run_summary.json"
            summary.write_text(
                json.dumps({"grid_resolution": 50}),
                encoding="utf-8",
            )
            self.assertEqual(resolve_nightly_grid_resolution(archive), 50)

    def test_resolve_nightly_grid_resolution_invalid_returns_none(self) -> None:
        import json
        import tempfile

        from scripts.run_github_llm_map_elites import resolve_nightly_grid_resolution

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            archive = run_dir / "map_elites_archive.jsonl"
            archive.write_text("{}\n", encoding="utf-8")
            summary = run_dir / "nightly_run_summary.json"
            for bad in ("fifty", [], {}):
                summary.write_text(
                    json.dumps({"grid_resolution": bad}),
                    encoding="utf-8",
                )
                self.assertIsNone(
                    resolve_nightly_grid_resolution(archive),
                    msg=f"expected None for {bad!r}",
                )

    def test_resolve_llm_spec_qwen(self) -> None:
        from scripts.run_github_llm_map_elites import resolve_llm_spec_path
        from worldspace.illuminators.scheduler import DEFAULT_QWEN_LLM_SPEC_PATH

        self.assertEqual(resolve_llm_spec_path("qwen"), DEFAULT_QWEN_LLM_SPEC_PATH)
        self.assertTrue(DEFAULT_QWEN_LLM_SPEC_PATH.is_file())

    def test_backfill_buffer_from_baseline_when_present(self) -> None:
        import tempfile

        import scripts.run_github_llm_map_elites as mod
        from scripts.run_github_llm_map_elites import _BASELINE_SUBDIR

        baseline = _REPO_ROOT / "artifacts" / "baseline" / "map_elites_archive.jsonl"
        if not baseline.is_file():
            self.skipTest("no local baseline archive for backfill smoke test")
        with tempfile.TemporaryDirectory() as tmp:
            buffer = Path(tmp) / "buffer_nightly.jsonl"
            orig_root = mod._NIGHTLY_ROOT
            orig_buffer = mod._NIGHTLY_BUFFER_PATH
            try:
                mod._NIGHTLY_ROOT = Path(tmp) / "map_elites_nightly"
                (mod._NIGHTLY_ROOT / _BASELINE_SUBDIR).mkdir(parents=True)
                archive = (
                    mod._NIGHTLY_ROOT / _BASELINE_SUBDIR / "map_elites_archive.jsonl"
                )
                archive.write_text(
                    baseline.read_text(encoding="utf-8"), encoding="utf-8"
                )
                mod._NIGHTLY_BUFFER_PATH = buffer
                self.assertTrue(mod._ensure_nightly_buffer_from_baseline())
                self.assertTrue(mod._nightly_buffer_has_rows())
            finally:
                mod._NIGHTLY_ROOT = orig_root
                mod._NIGHTLY_BUFFER_PATH = orig_buffer

    def test_empty_buffer_file_triggers_backfill(self) -> None:
        import tempfile

        import scripts.run_github_llm_map_elites as mod
        from scripts.run_github_llm_map_elites import _BASELINE_SUBDIR

        baseline = _REPO_ROOT / "artifacts" / "baseline" / "map_elites_archive.jsonl"
        if not baseline.is_file():
            self.skipTest("no local baseline archive for backfill smoke test")
        with tempfile.TemporaryDirectory() as tmp:
            buffer = Path(tmp) / "buffer_nightly.jsonl"
            buffer.write_text("", encoding="utf-8")
            orig_root = mod._NIGHTLY_ROOT
            orig_buffer = mod._NIGHTLY_BUFFER_PATH
            try:
                mod._NIGHTLY_ROOT = Path(tmp) / "map_elites_nightly"
                (mod._NIGHTLY_ROOT / _BASELINE_SUBDIR).mkdir(parents=True)
                archive = (
                    mod._NIGHTLY_ROOT / _BASELINE_SUBDIR / "map_elites_archive.jsonl"
                )
                archive.write_text(
                    baseline.read_text(encoding="utf-8"), encoding="utf-8"
                )
                mod._NIGHTLY_BUFFER_PATH = buffer

                self.assertFalse(mod._nightly_buffer_has_rows())
                self.assertTrue(mod._ensure_nightly_buffer_from_baseline())
                self.assertTrue(mod._nightly_buffer_has_rows())
            finally:
                mod._NIGHTLY_ROOT = orig_root
                mod._NIGHTLY_BUFFER_PATH = orig_buffer

    def test_qwen_yaml_uses_qwen_provider(self) -> None:
        from worldspace.generators.llm_config import load_llm_config
        from worldspace.illuminators.scheduler import DEFAULT_QWEN_LLM_SPEC_PATH

        cfg = load_llm_config(DEFAULT_QWEN_LLM_SPEC_PATH)
        self.assertEqual(cfg.active_provider, "qwen")


if __name__ == "__main__":
    unittest.main()

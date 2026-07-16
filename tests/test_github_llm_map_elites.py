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
        self.assertEqual(config.archive_type, "cvt")
        self.assertTrue(config.llm_enabled)
        self.assertTrue(config.surrogate_enabled)
        self.assertIn("nightly_v3_mc_d005.pkl", config.surrogate_checkpoint or "")
        self.assertEqual(config.iterations, 120)
        self.assertEqual(config.batch_size, 50)
        self.assertEqual(config.batch_emitters.count("llm"), 20)

    def test_resolve_nightly_baseline_archive_grid_legacy(self) -> None:
        import tempfile

        import scripts.run_github_llm_map_elites as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "map_elites_nightly"
            legacy = root / "baseline" / "map_elites_archive.jsonl"
            legacy.parent.mkdir(parents=True)
            legacy.write_text("{}\n", encoding="utf-8")
            orig_root = mod._NIGHTLY_ROOT
            try:
                mod._NIGHTLY_ROOT = root
                resolved = mod.resolve_nightly_baseline_archive("grid")
                self.assertEqual(resolved, legacy)
                self.assertIsNone(mod.resolve_nightly_baseline_archive("cvt"))
            finally:
                mod._NIGHTLY_ROOT = orig_root

    def test_resolve_baseline_archive_for_scheduler_grid(self) -> None:
        import tempfile

        import scripts.run_github_llm_map_elites as mod

        scheduler = (
            _REPO_ROOT / "worldspace/specs/map_elites_scheduler_nightly_llm_stub.yaml"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "map_elites_nightly"
            legacy = root / "baseline" / "map_elites_archive.jsonl"
            legacy.parent.mkdir(parents=True)
            legacy.write_text("{}\n", encoding="utf-8")
            orig_root = mod._NIGHTLY_ROOT
            try:
                mod._NIGHTLY_ROOT = root
                resolved = mod.resolve_baseline_archive_for_scheduler(scheduler)
                self.assertEqual(resolved, legacy)
            finally:
                mod._NIGHTLY_ROOT = orig_root

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
        from scripts.run_github_llm_map_elites import _CVT_BASELINE_SUBDIR

        baseline = _REPO_ROOT / "artifacts" / "baseline" / "map_elites_archive.jsonl"
        if not baseline.is_file():
            self.skipTest("no local baseline archive for backfill smoke test")
        with tempfile.TemporaryDirectory() as tmp:
            buffer = Path(tmp) / "buffer_nightly.jsonl"
            orig_root = mod._NIGHTLY_ROOT
            orig_buffer = mod._NIGHTLY_BUFFER_PATH
            try:
                mod._NIGHTLY_ROOT = Path(tmp) / "map_elites_nightly"
                (mod._NIGHTLY_ROOT / _CVT_BASELINE_SUBDIR).mkdir(parents=True)
                archive = (
                    mod._NIGHTLY_ROOT
                    / _CVT_BASELINE_SUBDIR
                    / "map_elites_archive.jsonl"
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
        from scripts.run_github_llm_map_elites import _CVT_BASELINE_SUBDIR

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
                (mod._NIGHTLY_ROOT / _CVT_BASELINE_SUBDIR).mkdir(parents=True)
                archive = (
                    mod._NIGHTLY_ROOT
                    / _CVT_BASELINE_SUBDIR
                    / "map_elites_archive.jsonl"
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

    def test_resolve_llm_spec_deepseek(self) -> None:
        from scripts.run_github_llm_map_elites import resolve_llm_spec_path
        from worldspace.generators.llm_config import load_llm_config

        path = resolve_llm_spec_path("deepseek")
        self.assertTrue(path.is_file())
        self.assertEqual(path.name, "llm_world_generator_deepseek.yaml")
        cfg = load_llm_config(path)
        self.assertEqual(cfg.active_provider, "deepseek")
        provider = cfg.providers["deepseek"]
        self.assertEqual(provider["model"], "deepseek-v4-pro")
        self.assertEqual(provider.get("thinking"), {"type": "disabled"})
        self.assertIn("api.deepseek.com", str(provider["api_base"]))
        self.assertEqual(provider["api_key_env"], "DEEPSEEK_API_KEY")

    def test_resolve_llm_spec_openai(self) -> None:
        from scripts.run_github_llm_map_elites import resolve_llm_spec_path
        from worldspace.generators.llm_config import load_llm_config

        path = resolve_llm_spec_path("openai")
        self.assertTrue(path.is_file())
        self.assertEqual(path.name, "llm_world_generator_openai.yaml")
        cfg = load_llm_config(path)
        self.assertEqual(cfg.active_provider, "openai")
        provider = cfg.providers["openai"]
        self.assertEqual(provider["model"], "gpt-4o-mini")
        self.assertIn("api.openai.com", str(provider["api_base"]))
        self.assertEqual(provider["api_key_env"], "OPENAI_API_KEY")

    def test_resolve_llm_spec_weak(self) -> None:
        from scripts.run_github_llm_map_elites import resolve_llm_spec_path
        from worldspace.generators.llm_config import load_llm_config

        path = resolve_llm_spec_path("weak")
        self.assertTrue(path.is_file())
        self.assertEqual(path.name, "llm_world_generator_weak.yaml")
        cfg = load_llm_config(path)
        self.assertEqual(cfg.active_provider, "weak")
        provider = cfg.providers["weak"]
        self.assertEqual(provider["model"], "qwen2.5-omni-7b")
        self.assertEqual(provider["api_key_env"], "QWEN_API_KEY")

    def test_call_llm_messages_sends_thinking_disabled(self) -> None:
        import json
        import os
        from unittest.mock import MagicMock, patch

        from worldspace.generators import call_llm_messages

        providers = {
            "deepseek": {
                "provider": "openai",
                "model": "deepseek-v4-pro",
                "api_base": "https://api.deepseek.com/v1/chat/completions",
                "api_key_env": "DEEPSEEK_API_KEY",
                "thinking": {"type": "disabled"},
            }
        }
        llm_body = {"choices": [{"message": {"content": "{}"}}]}
        fake_cm = MagicMock()
        fake_cm.__enter__.return_value.read.return_value = json.dumps(
            llm_body, ensure_ascii=True
        ).encode("utf-8")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-deepseek-key"}):
            with patch(
                "worldspace.generators.request.urlopen", return_value=fake_cm
            ) as m_open:
                out = call_llm_messages(
                    mode="remote",
                    provider_name="deepseek",
                    providers=providers,
                    messages=[{"role": "user", "content": "ping"}],
                )
        self.assertEqual(out, "{}")
        payload = json.loads(m_open.call_args[0][0].data.decode("utf-8"))
        self.assertEqual(payload["model"], "deepseek-v4-pro")
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_resolve_surrogate_quality_gate_from_env(self) -> None:
        import os
        from unittest import mock

        from scripts.run_github_llm_map_elites import resolve_surrogate_quality_gate

        with mock.patch.dict(os.environ, {"SURROGATE_REQUIRE_QUALITY_GATE": "true"}):
            self.assertTrue(resolve_surrogate_quality_gate(cli_flag=None))
        with mock.patch.dict(os.environ, {"SURROGATE_REQUIRE_QUALITY_GATE": "0"}):
            self.assertFalse(resolve_surrogate_quality_gate(cli_flag=None))

    def test_resolve_effective_checkpoint_stubs_when_gate_fails(self) -> None:
        import json
        import tempfile

        from scripts.run_github_llm_map_elites import (
            resolve_effective_surrogate_checkpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "nightly_v2.pkl"
            checkpoint.write_bytes(b"placeholder")
            summary = checkpoint.with_name("nightly_v2.summary.json")
            summary.write_text(
                json.dumps(
                    {
                        "quality_passed": False,
                        "feature_schema_version": "2.1",
                        "feature_dim": 24,
                    }
                ),
                encoding="utf-8",
            )
            effective = resolve_effective_surrogate_checkpoint(
                checkpoint,
                override=None,
                require_quality_gate=True,
                allow_ungated=False,
            )
            self.assertIsNone(effective)

    def test_resolve_effective_checkpoint_allows_override_when_gate_passes(
        self,
    ) -> None:
        import json
        import tempfile

        from scripts.run_github_llm_map_elites import (
            resolve_effective_surrogate_checkpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            default_ckpt = Path(tmpdir) / "default.pkl"
            default_ckpt.write_bytes(b"default")
            override = Path(tmpdir) / "override.pkl"
            override.write_bytes(b"override")
            summary = override.with_name("override.summary.json")
            summary.write_text(
                json.dumps(
                    {
                        "quality_passed": True,
                        "feature_schema_version": "2.1",
                        "feature_dim": 24,
                    }
                ),
                encoding="utf-8",
            )
            effective = resolve_effective_surrogate_checkpoint(
                default_ckpt,
                override=override,
                require_quality_gate=True,
                allow_ungated=False,
            )
            self.assertEqual(effective, override)


if __name__ == "__main__":
    unittest.main()

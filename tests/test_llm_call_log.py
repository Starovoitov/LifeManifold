"""Unit tests for per-call LLM JSONL archival."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from worldspace.generators import call_llm_messages
from worldspace.generators.llm_call_log import (
    append_llm_call_record,
    configure_llm_call_log,
    force_llm_call_log,
    messages_for_log,
    private_llm_call_log,
    resolve_llm_call_log_path,
)
from worldspace.generators.llm_config import load_llm_generator_yaml


class LlmCallLogTests(unittest.TestCase):
    def tearDown(self) -> None:
        force_llm_call_log(None)
        configure_llm_call_log(None)

    def test_messages_for_log_redacts_images(self) -> None:
        logged = messages_for_log(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAAA"},
                        },
                    ],
                }
            ]
        )
        self.assertEqual(
            logged[0]["content"][1]["image_url"]["url"], "[redacted:image]"
        )

    def test_env_zero_disables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"LIFEMANIFOLD_LLM_CALL_LOG": "0"}):
                self.assertIsNone(resolve_llm_call_log_path(output_dir=tmp))

    def test_call_llm_messages_appends_jsonl(self) -> None:
        spec_path = (
            Path(__file__).resolve().parent.parent
            / "worldspace/specs/llm_world_generator.yaml"
        )
        providers = load_llm_generator_yaml(spec_path)["llm"]["providers"]
        llm_body = {
            "id": "chatcmpl-test",
            "model": "qwen-plus",
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14,
            },
            "choices": [{"message": {"content": '{"ok": true}'}}],
        }
        fake_cm = MagicMock()
        fake_cm.__enter__.return_value.read.return_value = json.dumps(
            llm_body, ensure_ascii=True
        ).encode("utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "llm_call_log.jsonl"
            configure_llm_call_log(log_path)
            with patch.dict(os.environ, {"QWEN_API_KEY": "test-qwen-key"}, clear=False):
                # Ensure env does not force-disable.
                os.environ.pop("LIFEMANIFOLD_LLM_CALL_LOG", None)
                with patch(
                    "worldspace.generators.request.urlopen", return_value=fake_cm
                ):
                    out = call_llm_messages(
                        mode="remote",
                        provider_name="qwen",
                        providers=providers,
                        messages=[{"role": "user", "content": "ping"}],
                        audit_context={"llm_call_id": "fixed-call"},
                    )
            self.assertEqual(out, '{"ok": true}')
            rows = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertTrue(row["ok"])
            self.assertEqual(row["requested_model"], "qwen-plus")
            self.assertEqual(row["response_model"], "qwen-plus")
            self.assertEqual(row["usage"]["total_tokens"], 14)
            self.assertEqual(row["response_content"], '{"ok": true}')
            self.assertEqual(row["attempts"], 1)
            self.assertEqual(row["llm_call_log_schema"], 2)
            self.assertEqual(row["call_id"], "fixed-call")
            self.assertIn("request_body_sha256", row)
            self.assertEqual(
                row["audit_context"]["llm_call_id"],
                "fixed-call",
            )
            self.assertNotIn("Authorization", json.dumps(row))

    def test_private_log_keeps_prompt_and_scrubs_secrets_when_env_disables(self) -> None:
        secret = "sk-" + ("a" * 20)
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "llm_call_log.jsonl"
            with patch.dict(
                os.environ,
                {
                    "LIFEMANIFOLD_LLM_CALL_LOG": "0",
                    "OPENAI_API_KEY": secret,
                },
                clear=False,
            ):
                with private_llm_call_log(log_path, enabled=True):
                    self.assertEqual(resolve_llm_call_log_path(), log_path)
                    append_llm_call_record(
                        {
                            "call_id": "probe",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": f"header Bearer {secret} and {secret}",
                                }
                            ],
                            "response_content": "ops-json",
                            "ok": True,
                        }
                    )
                self.assertIsNone(resolve_llm_call_log_path(output_dir=tmp))
            text = log_path.read_text(encoding="utf-8")
            self.assertNotIn(secret, text)
            row = json.loads(text)
            self.assertEqual(row["messages"][0]["content"].count("[redacted]"), 2)
            self.assertIn("Bearer [redacted]", row["messages"][0]["content"])
            self.assertEqual(row["response_content"], "ops-json")
            self.assertEqual(row["call_id"], "probe")


if __name__ == "__main__":
    unittest.main()

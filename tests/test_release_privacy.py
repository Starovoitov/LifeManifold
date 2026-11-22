"""Sanitized call records keep metadata and drop provider text."""

from __future__ import annotations

import unittest

from worldspace.attribution.release_privacy import sanitize_llm_call_record


class TestSanitizeLlmCall(unittest.TestCase):
    def test_drops_messages_and_response_text(self) -> None:
        raw = {
            "ts_utc": "2026-11-21T00:00:00+00:00",
            "call_id": "abc",
            "ok": True,
            "attempts": 1,
            "latency_ms": 120.0,
            "requested_model": "gpt-4o-mini",
            "response_model": "gpt-4o-mini-2024-07-18",
            "finish_reason": "stop",
            "request_body_sha256": "a" * 64,
            "messages": [{"role": "user", "content": "sk-SECRET prompt"}],
            "response_content": "private answer",
            "error": "RuntimeError: sk-SECRET",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "prompt_tokens_details": {"cached_tokens": 3, "audio_tokens": 0},
            },
        }
        cleaned = sanitize_llm_call_record(raw, prompt_template_sha256="b" * 64)
        text = str(cleaned)
        self.assertNotIn("sk-SECRET", text)
        self.assertNotIn("private answer", text)
        self.assertNotIn("messages", cleaned)
        self.assertEqual(cleaned["prompt_tokens"], 10)
        self.assertEqual(cleaned["cached_prompt_tokens"], 3)
        self.assertEqual(cleaned["latency_ms"], 120.0)
        self.assertEqual(cleaned["response_model"], "gpt-4o-mini-2024-07-18")
        self.assertEqual(cleaned["prompt_template_sha256"], "b" * 64)

    def test_boolean_token_counts_are_dropped(self) -> None:
        raw = {
            "ok": True,
            "usage": {
                "prompt_tokens": True,
                "completion_tokens": False,
                "total_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": False},
            },
        }
        cleaned = sanitize_llm_call_record(raw, prompt_template_sha256=None)
        self.assertIsNone(cleaned["prompt_tokens"])
        self.assertIsNone(cleaned["completion_tokens"])
        self.assertIsNone(cleaned["cached_prompt_tokens"])
        self.assertEqual(cleaned["total_tokens"], 1)


if __name__ == "__main__":
    unittest.main()

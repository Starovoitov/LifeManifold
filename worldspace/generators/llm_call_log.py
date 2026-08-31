"""Thread-safe per-call LLM request/response JSONL archive.

Enabled for MAP-Elites runs under ``output_dir/llm_call_log.jsonl`` unless
``LIFEMANIFOLD_LLM_CALL_LOG=0``. Override path with ``LIFEMANIFOLD_LLM_CALL_LOG=/path``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_LLM_CALL_LOG_NAME",
    "LLM_CALL_LOG_SCHEMA",
    "append_llm_call_record",
    "configure_llm_call_log",
    "get_llm_call_record",
    "llm_call_log_path",
    "messages_for_log",
    "resolve_llm_call_log_path",
]

DEFAULT_LLM_CALL_LOG_NAME = "llm_call_log.jsonl"
LLM_CALL_LOG_SCHEMA = 2

_lock = threading.Lock()
_configured_path: Path | None = None
_records_by_id: dict[str, dict[str, Any]] = {}


def configure_llm_call_log(path: str | Path | None) -> Path | None:
    """Set the process-wide call-log path (or disable with ``None``)."""
    global _configured_path
    with _lock:
        _records_by_id.clear()
    if path is None:
        _configured_path = None
        return None
    dest = Path(path).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    _configured_path = dest
    return dest


def llm_call_log_path() -> Path | None:
    """Return the active log path after env + configure resolution."""
    return resolve_llm_call_log_path()


def resolve_llm_call_log_path(
    *,
    output_dir: str | Path | None = None,
) -> Path | None:
    """Resolve log path from env and optional ``output_dir`` default."""
    raw = os.environ.get("LIFEMANIFOLD_LLM_CALL_LOG")
    if raw is not None:
        value = raw.strip()
        if value in {"0", "false", "False", "off", "OFF"}:
            return None
        if value in {"1", "true", "True", "on", "ON"}:
            if _configured_path is not None:
                return _configured_path
            if output_dir is not None:
                return Path(output_dir).expanduser() / DEFAULT_LLM_CALL_LOG_NAME
            return _configured_path
        return Path(value).expanduser()
    if _configured_path is not None:
        return _configured_path
    if output_dir is not None:
        return Path(output_dir).expanduser() / DEFAULT_LLM_CALL_LOG_NAME
    return None


def messages_for_log(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy messages for logging; redact multimodal image payloads."""
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, list):
            parts: list[Any] = []
            for part in content:
                if not isinstance(part, dict):
                    parts.append(part)
                    continue
                if part.get("type") == "image_url":
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": "[redacted:image]"},
                        }
                    )
                else:
                    parts.append(part)
            out.append({"role": role, "content": parts})
        else:
            out.append({"role": role, "content": content})
    return out


def get_llm_call_record(call_id: str) -> dict[str, Any] | None:
    """Return a shallow copy of one call record captured in this process."""
    with _lock:
        record = _records_by_id.get(call_id)
        return dict(record) if record is not None else None


def append_llm_call_record(record: dict[str, Any]) -> None:
    """Append one JSONL record if a log path is active."""
    path = resolve_llm_call_log_path()
    if path is None:
        return
    payload = {
        "llm_call_log_schema": LLM_CALL_LOG_SCHEMA,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "call_id": str(uuid.uuid4()),
        "python_version": sys.version.split()[0],
        "http_client": "stdlib urllib",
        **record,
    }
    line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
    with _lock:
        _records_by_id[str(payload["call_id"])] = payload
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)

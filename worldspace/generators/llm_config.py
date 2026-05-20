"""YAML-backed configuration for LLM world generators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml


class LlmVisionCaller(Protocol):
    """Callable shape for multimodal simulation captioning."""

    def __call__(
        self,
        *,
        mode: str,
        provider_name: str,
        providers: dict[str, Any],
        system_content: str,
        user_text: str,
        image_png_bytes: bytes,
        temperature: float = 0.1,
        max_tokens: int = 300,
    ) -> str: ...


class LlmTextCaller(Protocol):
    """Callable shape for text-only patch / chat completions."""

    def __call__(
        self,
        *,
        mode: str,
        provider_name: str,
        providers: dict[str, Any],
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 350,
        system_content: str | None = None,
    ) -> str: ...


_DEFAULT_LLM_SPEC_PATH = (
    Path(__file__).resolve().parent.parent / "specs" / "llm_world_generator.yaml"
)


@dataclass(frozen=True)
class LLMGeneratorConfig:
    """Shared LLM provider settings for local and global-search generators."""

    mode: str
    active_provider: str
    providers: dict[str, Any]
    temperature: float
    max_tokens: int
    fallback_scale: float
    initial_generator: str
    global_search: bool
    vision_provider: str
    vision_max_tokens: int
    vision_temperature: float
    descriptive_max_side: int

    @classmethod
    def from_llm_dict(cls, llm: dict[str, Any]) -> LLMGeneratorConfig:
        active = str(llm["active_provider"]).strip()
        vision_raw = llm.get("vision_provider")
        vision_provider = str(vision_raw).strip() if vision_raw else active
        return cls(
            mode=str(llm["mode"]).strip().lower(),
            active_provider=active,
            providers=dict(llm["providers"]),
            temperature=float(llm.get("temperature", 0.2)),
            max_tokens=int(llm.get("max_tokens", 350)),
            fallback_scale=float(llm.get("fallback_scale", 0.02)),
            initial_generator=str(llm.get("initial_generator", "random")),
            global_search=bool(llm.get("global_search", False)),
            vision_provider=vision_provider,
            vision_max_tokens=int(llm.get("vision_max_tokens", 300)),
            vision_temperature=float(llm.get("vision_temperature", 0.1)),
            descriptive_max_side=int(llm.get("descriptive_max_side", 128)),
        )


def load_llm_generator_yaml(path: str | Path) -> dict[str, Any]:
    """Load and validate llm world generator YAML."""
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(
            f"LLM generator YAML not found: {src.resolve()}. "
            "Pass --generator-spec in CLI or place default llm_world_generator.yaml in worldspace/specs/."
        )
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {src}")
    if raw.get("version") != 1:
        raise ValueError(f"{src}: expected version: 1")
    llm = raw.get("llm")
    if not isinstance(llm, dict):
        raise ValueError(f"{src}: expected top-level key 'llm'")
    for key in ("mode", "active_provider", "providers"):
        if key not in llm:
            raise ValueError(f"{src}: llm.{key} is required")
    if not isinstance(llm["providers"], dict):
        raise ValueError(f"{src}: llm.providers must be a mapping")
    return raw


def load_llm_config(path: str | Path | None = None) -> LLMGeneratorConfig:
    """Load ``LLMGeneratorConfig`` from the default or given spec path."""
    raw = load_llm_generator_yaml(path or _DEFAULT_LLM_SPEC_PATH)
    return LLMGeneratorConfig.from_llm_dict(raw["llm"])

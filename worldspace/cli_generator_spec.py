"""Pydantic validation of generator YAML paths for ``worldspace`` CLI."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _GeneticGeneratorYaml(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(..., ge=1, le=1)
    genetic: dict[str, Any]
    pygad: dict[str, Any]


class _LlmOnlyGeneratorYaml(BaseModel):
    """LLM standalone config (no ``evolution`` block — rejects hybrid YAML)."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(..., ge=1, le=1)
    llm: dict[str, Any]


class _HybridGeneratorYaml(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(..., ge=1, le=1)
    evolution: dict[str, Any]
    llm: dict[str, Any]


class _NeuralGeneratorYaml(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(..., ge=1, le=1)
    torch: dict[str, Any]
    model: dict[str, Any]
    decoder: dict[str, Any]
    world_defaults: dict[str, Any]
    weights_path: Any = None
    base_seed: int | str | None = 0


_SPEC_MODELS: dict[str, type[BaseModel]] = {
    "genetic": _GeneticGeneratorYaml,
    "llm": _LlmOnlyGeneratorYaml,
    "hybrid": _HybridGeneratorYaml,
    "neural": _NeuralGeneratorYaml,
}


def parse_generator_spec_path(raw: str) -> Path | None:
    """Return expanded path or ``None`` when the argument is empty."""
    s = raw.strip()
    if not s:
        return None
    return Path(s).expanduser()


def validate_generator_spec_yaml(generator: str, spec_path: Path) -> None:
    """
    Load ``spec_path`` and ensure the document matches the YAML shape expected
    for ``generator`` (``genetic`` | ``llm`` | ``hybrid`` | ``neural``).

    Raises:
        FileNotFoundError: if the path is not a file.
        ValueError: on invalid YAML root or Pydantic validation failure.
    """
    model = _SPEC_MODELS.get(generator)
    if model is None:
        return
    src = spec_path
    if not src.is_file():
        raise FileNotFoundError(f"Generator spec YAML not found: {src.resolve()}")
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {src}")
    try:
        model.model_validate(raw)
    except ValidationError as exc:
        hint = (
            f"The file {src} is not a valid {generator} generator spec "
            f"(wrong top-level keys or structure for --generator {generator})."
        )
        raise ValueError(f"{hint}\n{exc}") from exc

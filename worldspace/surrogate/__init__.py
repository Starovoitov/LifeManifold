"""Surrogate factory and MVP-compatible implementations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from worldspace.surrogate.checkpoint_io import load_surrogate_checkpoint
from worldspace.surrogate.types import (
    SurrogateConfig,
    SurrogatePrediction,
    SurrogateProtocol,
)

if TYPE_CHECKING:
    from worldspace.surrogate.surrogate import StubSurrogate, SurrogateFacade

__all__ = [
    "StubSurrogate",
    "SurrogateFacade",
    "SurrogateConfig",
    "SurrogatePrediction",
    "SurrogateProtocol",
    "get_surrogate",
]


def get_surrogate(config: SurrogateConfig) -> SurrogateProtocol:
    """Create real surrogate when enabled and checkpoint exists, else stub."""
    from worldspace.surrogate.surrogate import StubSurrogate, build_surrogate_facade

    if not config.enabled:
        return StubSurrogate(mean=config.stub_mean, uncertainty=config.stub_uncertainty)
    checkpoint = _checkpoint_path(config.checkpoint)
    if checkpoint is None or not checkpoint.is_file():
        return StubSurrogate(mean=config.stub_mean, uncertainty=config.stub_uncertainty)
    model = load_surrogate_checkpoint(checkpoint)
    return build_surrogate_facade(
        model,
        uncertainty_fallback=config.stub_uncertainty,
        calibration_path=config.calibration,
    )


def __getattr__(name: str) -> object:
    if name == "StubSurrogate":
        from worldspace.surrogate.surrogate import StubSurrogate

        return StubSurrogate
    if name == "SurrogateFacade":
        from worldspace.surrogate.surrogate import SurrogateFacade

        return SurrogateFacade
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def _checkpoint_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser()

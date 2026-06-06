"""Surrogate factory and MVP-compatible implementations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from worldspace.surrogate.checkpoint_io import (
    CHECKPOINT_LOAD_ERRORS,
    load_surrogate_checkpoint,
)
from worldspace.surrogate.checkpoint_paths import resolve_runtime_checkpoint_path
from worldspace.surrogate.checkpoint_quality import checkpoint_quality_allows_hints
from worldspace.surrogate.model import (
    EXPECTED_FEATURE_DIM,
    checkpoint_feature_dim,
    checkpoint_matches_extractor,
)
from worldspace.surrogate.types import (
    SurrogateConfig,
    SurrogatePrediction,
    SurrogateProtocol,
)

if TYPE_CHECKING:
    from worldspace.surrogate.surrogate import StubSurrogate, SurrogateFacade

logger = logging.getLogger(__name__)

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
    try:
        model = load_surrogate_checkpoint(checkpoint)
    except CHECKPOINT_LOAD_ERRORS as exc:
        logger.warning(
            "Surrogate checkpoint load failed (%s): %s; using stub",
            checkpoint,
            exc,
        )
        return StubSurrogate(mean=config.stub_mean, uncertainty=config.stub_uncertainty)
    try:
        if not checkpoint_matches_extractor(model):
            trained_dim = checkpoint_feature_dim(model)
            logger.warning(
                "Surrogate checkpoint feature_dim=%s, expected %s (%s); using stub",
                trained_dim,
                EXPECTED_FEATURE_DIM,
                checkpoint,
            )
            return StubSurrogate(
                mean=config.stub_mean, uncertainty=config.stub_uncertainty
            )
    except ValueError as exc:
        logger.warning(
            "Surrogate checkpoint validation failed (%s): %s; using stub",
            checkpoint,
            exc,
        )
        return StubSurrogate(mean=config.stub_mean, uncertainty=config.stub_uncertainty)
    if config.require_quality_gate and not checkpoint_quality_allows_hints(checkpoint):
        logger.warning(
            "Surrogate checkpoint failed quality gate (%s); using stub",
            checkpoint,
        )
        return StubSurrogate(mean=config.stub_mean, uncertainty=config.stub_uncertainty)
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
    return resolve_runtime_checkpoint_path(value)

"""Deterministic feature extraction from canonicalized world specs."""

from __future__ import annotations

import numpy as np

from worldspace.specs.spec import WorldSpec
from worldspace.surrogate.genome_features import (
    FEATURE_DIM,
    FEATURE_DIM_V21,
    encode_world_spec_features,
    encode_world_spec_features_v21,
)

FEATURE_SCHEMA_VERSION_20 = "2.0"
FEATURE_SCHEMA_VERSION = "2.1"

FEATURE_NAMES_V20: tuple[str, ...] = (
    tuple(f"birth_{index}" for index in range(9))
    + tuple(f"survival_{index}" for index in range(9))
    + (
        "noise",
        "resource_regen",
        "predation",
    )
)
FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES_V20 + (
    "birth_count",
    "survival_count",
    "rule_overlap",
)

SUPPORTED_FEATURE_SCHEMA_VERSIONS: frozenset[str] = frozenset(
    {FEATURE_SCHEMA_VERSION_20, FEATURE_SCHEMA_VERSION}
)
SCHEMA_FEATURE_DIMS: dict[str, int] = {
    FEATURE_SCHEMA_VERSION_20: FEATURE_DIM,
    FEATURE_SCHEMA_VERSION: FEATURE_DIM_V21,
}

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_NAMES_V20",
    "FEATURE_SCHEMA_VERSION",
    "FEATURE_SCHEMA_VERSION_20",
    "SCHEMA_FEATURE_DIMS",
    "SUPPORTED_FEATURE_SCHEMA_VERSIONS",
    "extract",
    "feature_dim_for_schema",
    "feature_names_for_dim",
    "feature_names_for_schema",
]


def feature_dim_for_schema(schema_version: str) -> int:
    """Return expected feature width for one buffer schema version."""
    try:
        return SCHEMA_FEATURE_DIMS[schema_version]
    except KeyError as exc:
        msg = f"Unsupported feature_schema_version: {schema_version!r}"
        raise ValueError(msg) from exc


def feature_names_for_schema(schema_version: str) -> tuple[str, ...]:
    """Return stable feature column names for one schema version."""
    if schema_version == FEATURE_SCHEMA_VERSION_20:
        return FEATURE_NAMES_V20
    if schema_version == FEATURE_SCHEMA_VERSION:
        return FEATURE_NAMES
    msg = f"Unsupported feature_schema_version: {schema_version!r}"
    raise ValueError(msg)


def feature_names_for_dim(feature_dim: int) -> tuple[str, ...]:
    """Return LightGBM column names for a buffer or matrix feature width."""
    if feature_dim == FEATURE_DIM:
        return FEATURE_NAMES_V20
    if feature_dim == FEATURE_DIM_V21:
        return FEATURE_NAMES
    from worldspace.surrogate.emitter_features import EMITTER_ONEHOT_DIM

    if feature_dim == FEATURE_DIM_V21 + EMITTER_ONEHOT_DIM:
        return FEATURE_NAMES + (
            "emitter_random",
            "emitter_genetic",
            "emitter_llm",
        )
    return tuple(f"feature_{index}" for index in range(feature_dim))


def extract(
    spec: WorldSpec,
    *,
    schema_version: str = FEATURE_SCHEMA_VERSION,
) -> np.ndarray:
    """Return deterministic numeric features from a canonicalized ``WorldSpec``."""
    _require_canonical_seed(spec)
    if schema_version == FEATURE_SCHEMA_VERSION_20:
        return encode_world_spec_features(spec)
    if schema_version == FEATURE_SCHEMA_VERSION:
        return encode_world_spec_features_v21(spec)
    msg = f"Unsupported feature_schema_version: {schema_version!r}"
    raise ValueError(msg)


def _require_canonical_seed(spec: WorldSpec) -> None:
    from worldspace.illuminators.evaluation import canonical_seed

    expected = canonical_seed(spec)
    if spec.seed != expected:
        msg = (
            "feature_extractor.extract requires canonicalized WorldSpec: "
            f"seed={spec.seed}, expected={expected}"
        )
        raise ValueError(msg)

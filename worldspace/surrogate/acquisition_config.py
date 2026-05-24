"""Surrogate acquisition and nested retrain settings from scheduler YAML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AcquisitionMode = Literal["off", "shadow", "filter"]
AcquisitionPolicyName = Literal["threshold_gate", "ucb_promote"]

DEFAULT_SURROGATE_ARCHIVE_PATH = "artifacts/surrogate/surrogate_archive.jsonl"

__all__ = [
    "AcquisitionConfig",
    "AcquisitionMode",
    "AcquisitionPolicyName",
    "DEFAULT_SURROGATE_ARCHIVE_PATH",
    "RetrainConfig",
]


@dataclass(frozen=True)
class AcquisitionConfig:
    """Rule-based eval/skip policy settings (Surrogate Acquisition)."""

    mode: AcquisitionMode = "off"
    policy: AcquisitionPolicyName = "threshold_gate"
    min_predicted_fitness: float = 0.25
    max_uncertainty_to_skip: float = 0.40
    never_skip_empty_bin: bool = True
    exploration_weight: float = 0.15


@dataclass(frozen=True)
class RetrainConfig:
    """Optional in-run surrogate retrain loop (release 2.1)."""

    enabled: bool = False
    every_iterations: int = 50
    min_new_buffer_rows: int = 500

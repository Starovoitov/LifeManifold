"""Acquisition decision types and reason codes (policy logic in SA-2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AcquisitionAction = Literal["eval", "skip"]

REASON_ACCEPTED_FOR_EVAL = "accepted_for_eval"
REASON_BELOW_FITNESS_THRESHOLD = "below_fitness_threshold"
REASON_HIGH_UNCERTAINTY_FORCE_EVAL = "high_uncertainty_force_eval"
REASON_EMPTY_BIN_EXPLORE = "empty_bin_explore"
REASON_POLICY_UNSUPPORTED = "policy_unsupported"
REASON_ACQUISITION_DISABLED = "acquisition_disabled"

THRESHOLD_GATE_POLICY_VERSION = "threshold_gate_v1"
UCB_PROMOTE_POLICY_VERSION = "ucb_promote_v1"

__all__ = [
    "REASON_ACCEPTED_FOR_EVAL",
    "REASON_ACQUISITION_DISABLED",
    "REASON_BELOW_FITNESS_THRESHOLD",
    "REASON_EMPTY_BIN_EXPLORE",
    "REASON_HIGH_UNCERTAINTY_FORCE_EVAL",
    "REASON_POLICY_UNSUPPORTED",
    "THRESHOLD_GATE_POLICY_VERSION",
    "UCB_PROMOTE_POLICY_VERSION",
    "AcquisitionAction",
    "AcquisitionDecision",
]


@dataclass(frozen=True)
class AcquisitionDecision:
    """Outcome of one acquisition policy evaluation for a candidate slot."""

    action: AcquisitionAction
    reason: str
    policy_version: str

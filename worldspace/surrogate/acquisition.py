"""Acquisition decision types and rule-based eval/skip policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from worldspace.illuminators.archive_protocol import ArchiveProtocol
from worldspace.illuminators.scheduler import TargetBin
from worldspace.surrogate.acquisition_config import (
    AcquisitionConfig,
    AcquisitionMode,
    AcquisitionPolicyName,
)
from worldspace.surrogate.types import SurrogatePrediction

AcquisitionAction = Literal["eval", "skip"]

REASON_ACCEPTED_FOR_EVAL = "accepted_for_eval"
REASON_BELOW_FITNESS_THRESHOLD = "below_fitness_threshold"
REASON_HIGH_UNCERTAINTY_FORCE_EVAL = "high_uncertainty_force_eval"
REASON_EMPTY_BIN_EXPLORE = "empty_bin_explore"
REASON_POLICY_UNSUPPORTED = "policy_unsupported"
REASON_ACQUISITION_DISABLED = "acquisition_disabled"

OFF_POLICY_VERSION = "off_v1"
UNSUPPORTED_POLICY_VERSION = "unsupported_v1"
THRESHOLD_GATE_POLICY_VERSION = "threshold_gate_v1"
UCB_PROMOTE_POLICY_VERSION = "ucb_promote_v1"

__all__ = [
    "OFF_POLICY_VERSION",
    "REASON_ACCEPTED_FOR_EVAL",
    "REASON_ACQUISITION_DISABLED",
    "REASON_BELOW_FITNESS_THRESHOLD",
    "REASON_EMPTY_BIN_EXPLORE",
    "REASON_HIGH_UNCERTAINTY_FORCE_EVAL",
    "REASON_POLICY_UNSUPPORTED",
    "THRESHOLD_GATE_POLICY_VERSION",
    "UCB_PROMOTE_POLICY_VERSION",
    "UNSUPPORTED_POLICY_VERSION",
    "AcquisitionAction",
    "AcquisitionDecision",
    "decide",
    "effective_action",
    "policy_recommends_skip",
]


@dataclass(frozen=True)
class AcquisitionDecision:
    """Outcome of one acquisition policy evaluation for a candidate slot."""

    action: AcquisitionAction
    reason: str
    policy_version: str


PolicyFn = Callable[
    [AcquisitionConfig, SurrogatePrediction, TargetBin, ArchiveProtocol],
    AcquisitionDecision,
]


def decide(
    config: AcquisitionConfig,
    prediction: SurrogatePrediction,
    target: TargetBin,
    archive: ArchiveProtocol,
) -> AcquisitionDecision:
    """Return the policy-layer recommendation for one candidate slot."""
    if config.mode == "off":
        return _eval_decision(REASON_ACQUISITION_DISABLED, OFF_POLICY_VERSION)

    policy_fn = _POLICY_REGISTRY.get(config.policy)
    if policy_fn is None:
        return _eval_decision(REASON_POLICY_UNSUPPORTED, UNSUPPORTED_POLICY_VERSION)
    return policy_fn(config, prediction, target, archive)


def effective_action(
    mode: AcquisitionMode,
    decision: AcquisitionDecision,
) -> AcquisitionAction:
    """Return the action the illuminator loop must take for this acquisition mode."""
    if mode == "filter":
        return decision.action
    return "eval"


def policy_recommends_skip(decision: AcquisitionDecision) -> bool:
    """True when the policy layer chose skip (for shadow_would_skip metrics)."""
    return decision.action == "skip"


def _decide_threshold_gate(
    config: AcquisitionConfig,
    prediction: SurrogatePrediction,
    target: TargetBin,
    archive: ArchiveProtocol,
) -> AcquisitionDecision:
    cell_id = archive.cell_id_from_bin(target.bin)
    if config.never_skip_empty_bin and archive.is_empty_cell(cell_id):
        return _eval_decision(REASON_EMPTY_BIN_EXPLORE, THRESHOLD_GATE_POLICY_VERSION)

    low_fitness = prediction.fitness < config.min_predicted_fitness
    low_uncertainty = prediction.uncertainty <= config.max_uncertainty_to_skip
    if low_fitness and low_uncertainty:
        return _skip_decision(
            REASON_BELOW_FITNESS_THRESHOLD,
            THRESHOLD_GATE_POLICY_VERSION,
        )
    if low_fitness:
        return _eval_decision(
            REASON_HIGH_UNCERTAINTY_FORCE_EVAL,
            THRESHOLD_GATE_POLICY_VERSION,
        )
    return _eval_decision(REASON_ACCEPTED_FOR_EVAL, THRESHOLD_GATE_POLICY_VERSION)


_POLICY_REGISTRY: dict[AcquisitionPolicyName, PolicyFn] = {
    "threshold_gate": _decide_threshold_gate,
}


def _skip_decision(reason: str, policy_version: str) -> AcquisitionDecision:
    return AcquisitionDecision(
        action="skip", reason=reason, policy_version=policy_version
    )


def _eval_decision(reason: str, policy_version: str) -> AcquisitionDecision:
    return AcquisitionDecision(
        action="eval", reason=reason, policy_version=policy_version
    )

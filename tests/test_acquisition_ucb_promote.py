"""Unit tests for threshold_gate vs ucb_promote acquisition policies."""

from __future__ import annotations

import unittest

from worldspace.illuminators.archive_factory import ArchiveFactoryConfig, create_archive
from worldspace.illuminators.scheduler import TargetBin
from worldspace.surrogate.acquisition import (
    REASON_ACCEPTED_FOR_EVAL,
    REASON_BELOW_UCB_THRESHOLD,
    decide,
)
from worldspace.surrogate.acquisition_config import AcquisitionConfig
from worldspace.surrogate.types import SurrogatePrediction


def _target(archive, cell_id: int = 0) -> TargetBin:
    return TargetBin(
        bin=archive.bin_from_cell_id(cell_id),
        target_stability=0.5,
        target_diversity=0.5,
    )


class TestUcbPromotePolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.archive = create_archive(
            ArchiveFactoryConfig(archive_type="grid", resolution=10)
        )
        self.policy = AcquisitionConfig(
            mode="filter",
            policy="ucb_promote",
            min_predicted_fitness=0.45,
            exploration_weight=1.0,
            never_skip_empty_bin=False,
        )
        self.target = _target(self.archive, 0)

    def test_low_ucb_skips(self) -> None:
        pred = SurrogatePrediction(
            components={}, measures={}, fitness=0.2, uncertainty=0.05
        )
        # UCB = 0.2 + 1.0*0.05 = 0.25 < 0.45
        decision = decide(self.policy, pred, self.target, self.archive)
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, REASON_BELOW_UCB_THRESHOLD)

    def test_high_uncertainty_promotes_eval(self) -> None:
        pred = SurrogatePrediction(
            components={}, measures={}, fitness=0.2, uncertainty=0.40
        )
        # UCB = 0.2 + 0.40 = 0.60 >= 0.45
        decision = decide(self.policy, pred, self.target, self.archive)
        self.assertEqual(decision.action, "eval")
        self.assertEqual(decision.reason, REASON_ACCEPTED_FOR_EVAL)

    def test_threshold_gate_ignores_sigma_when_max_u_one(self) -> None:
        policy = AcquisitionConfig(
            mode="filter",
            policy="threshold_gate",
            min_predicted_fitness=0.45,
            max_uncertainty_to_skip=1.0,
            never_skip_empty_bin=False,
        )
        low = SurrogatePrediction(
            components={}, measures={}, fitness=0.2, uncertainty=0.01
        )
        high_u = SurrogatePrediction(
            components={}, measures={}, fitness=0.2, uncertainty=0.99
        )
        self.assertEqual(decide(policy, low, self.target, self.archive).action, "skip")
        self.assertEqual(
            decide(policy, high_u, self.target, self.archive).action, "skip"
        )


if __name__ == "__main__":
    unittest.main()

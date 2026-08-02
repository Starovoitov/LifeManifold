"""Unit tests for surrogate acquisition policies."""

from __future__ import annotations

import unittest

from worldspace.illuminators.archive import ArchiveElite, GridArchive
from worldspace.illuminators.scheduler import TargetBin
import numpy as np

from worldspace.surrogate.acquisition import (
    OFF_POLICY_VERSION,
    RANDOM_SKIP_POLICY_VERSION,
    REASON_ACCEPTED_FOR_EVAL,
    REASON_ACQUISITION_DISABLED,
    REASON_BELOW_FITNESS_THRESHOLD,
    REASON_EMPTY_BIN_EXPLORE,
    REASON_HIGH_UNCERTAINTY_FORCE_EVAL,
    REASON_RANDOM_SKIP,
    THRESHOLD_GATE_POLICY_VERSION,
    AcquisitionDecision,
    decide,
    effective_action,
    policy_recommends_skip,
)
from worldspace.surrogate.acquisition_config import (
    AcquisitionConfig,
    AcquisitionMode,
)
from worldspace.surrogate.types import SurrogatePrediction


def _prediction(fitness: float, uncertainty: float) -> SurrogatePrediction:
    return SurrogatePrediction(
        components={},
        measures={},
        fitness=fitness,
        uncertainty=uncertainty,
    )


def _target(bin_ij: tuple[int, int] = (0, 0)) -> TargetBin:
    return TargetBin(bin=bin_ij, target_stability=0.5, target_diversity=0.5)


def _archive_filled_at(bin_ij: tuple[int, int], *, resolution: int = 5) -> GridArchive:
    archive = GridArchive(resolution=resolution)
    archive.try_insert(ArchiveElite(bin=bin_ij, fitness=0.5))
    return archive


def _config(
    *,
    mode: AcquisitionMode = "filter",
    never_skip_empty_bin: bool = True,
    min_predicted_fitness: float = 0.25,
    max_uncertainty_to_skip: float = 0.40,
) -> AcquisitionConfig:
    return AcquisitionConfig(
        mode=mode,
        never_skip_empty_bin=never_skip_empty_bin,
        min_predicted_fitness=min_predicted_fitness,
        max_uncertainty_to_skip=max_uncertainty_to_skip,
    )


class TestSurrogateAcquisition(unittest.TestCase):
    def test_mode_off_returns_eval_disabled(self) -> None:
        decision = decide(
            _config(mode="off"),
            _prediction(0.0, 0.0),
            _target(),
            GridArchive(resolution=5),
        )
        self.assertEqual(decision.action, "eval")
        self.assertEqual(decision.reason, REASON_ACQUISITION_DISABLED)
        self.assertEqual(decision.policy_version, OFF_POLICY_VERSION)

    def test_threshold_skip_low_fitness_low_uncertainty(self) -> None:
        bin_ij = (0, 0)
        decision = decide(
            _config(),
            _prediction(0.1, 0.1),
            _target(bin_ij),
            _archive_filled_at(bin_ij),
        )
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, REASON_BELOW_FITNESS_THRESHOLD)
        self.assertEqual(decision.policy_version, THRESHOLD_GATE_POLICY_VERSION)

    def test_threshold_eval_low_fitness_high_uncertainty(self) -> None:
        bin_ij = (0, 0)
        decision = decide(
            _config(),
            _prediction(0.1, 0.9),
            _target(bin_ij),
            _archive_filled_at(bin_ij),
        )
        self.assertEqual(decision.action, "eval")
        self.assertEqual(decision.reason, REASON_HIGH_UNCERTAINTY_FORCE_EVAL)

    def test_threshold_eval_high_fitness(self) -> None:
        bin_ij = (0, 0)
        decision = decide(
            _config(),
            _prediction(0.9, 0.1),
            _target(bin_ij),
            _archive_filled_at(bin_ij),
        )
        self.assertEqual(decision.action, "eval")
        self.assertEqual(decision.reason, REASON_ACCEPTED_FOR_EVAL)

    def test_empty_bin_forces_eval_despite_skip_signals(self) -> None:
        decision = decide(
            _config(),
            _prediction(0.0, 0.0),
            _target((2, 2)),
            GridArchive(resolution=5),
        )
        self.assertEqual(decision.action, "eval")
        self.assertEqual(decision.reason, REASON_EMPTY_BIN_EXPLORE)

    def test_never_skip_empty_bin_false_allows_skip(self) -> None:
        decision = decide(
            _config(never_skip_empty_bin=False),
            _prediction(0.0, 0.0),
            _target((2, 2)),
            GridArchive(resolution=5),
        )
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, REASON_BELOW_FITNESS_THRESHOLD)

    def test_effective_action_filter_honors_skip(self) -> None:
        decision = AcquisitionDecision(
            action="skip",
            reason=REASON_BELOW_FITNESS_THRESHOLD,
            policy_version=THRESHOLD_GATE_POLICY_VERSION,
        )
        self.assertEqual(effective_action("filter", decision), "skip")

    def test_effective_action_shadow_always_eval(self) -> None:
        decision = AcquisitionDecision(
            action="skip",
            reason=REASON_BELOW_FITNESS_THRESHOLD,
            policy_version=THRESHOLD_GATE_POLICY_VERSION,
        )
        self.assertEqual(effective_action("shadow", decision), "eval")
        self.assertEqual(effective_action("off", decision), "eval")

    def test_policy_recommends_skip(self) -> None:
        skip = AcquisitionDecision(
            action="skip",
            reason=REASON_BELOW_FITNESS_THRESHOLD,
            policy_version=THRESHOLD_GATE_POLICY_VERSION,
        )
        eval_decision = AcquisitionDecision(
            action="eval",
            reason=REASON_ACCEPTED_FOR_EVAL,
            policy_version=THRESHOLD_GATE_POLICY_VERSION,
        )
        self.assertTrue(policy_recommends_skip(skip))
        self.assertFalse(policy_recommends_skip(eval_decision))

    def test_decide_is_deterministic(self) -> None:
        config = _config()
        prediction = _prediction(0.1, 0.2)
        bin_ij = (1, 3)
        target = _target(bin_ij)
        archive = _archive_filled_at(bin_ij)
        first = decide(config, prediction, target, archive)
        second = decide(config, prediction, target, archive)
        self.assertEqual(first, second)

    def test_gray_zone_forces_eval_despite_low_fitness(self) -> None:
        from worldspace.surrogate.acquisition import (
            REASON_EXTINCTION_GRAY_ZONE_FORCE_EVAL,
            THRESHOLD_GATE_GRAY_ZONE_POLICY_VERSION,
        )

        bin_ij = (0, 0)
        prediction = SurrogatePrediction(
            components={"early_extinction_prob": 0.7},
            measures={},
            fitness=0.1,
            uncertainty=0.1,
        )
        decision = decide(
            AcquisitionConfig(
                mode="filter",
                min_predicted_fitness=0.45,
                max_uncertainty_to_skip=1.0,
                force_eval_extinction_gray_zone=True,
            ),
            prediction,
            _target(bin_ij),
            _archive_filled_at(bin_ij),
        )
        self.assertEqual(decision.action, "eval")
        self.assertEqual(decision.reason, REASON_EXTINCTION_GRAY_ZONE_FORCE_EVAL)
        self.assertEqual(
            decision.policy_version, THRESHOLD_GATE_GRAY_ZONE_POLICY_VERSION
        )

    def test_gray_zone_disabled_allows_skip(self) -> None:
        bin_ij = (0, 0)
        prediction = SurrogatePrediction(
            components={"early_extinction_prob": 0.7},
            measures={},
            fitness=0.1,
            uncertainty=0.1,
        )
        decision = decide(
            _config(min_predicted_fitness=0.45, max_uncertainty_to_skip=1.0),
            prediction,
            _target(bin_ij),
            _archive_filled_at(bin_ij),
        )
        self.assertEqual(decision.action, "skip")


class TestSurrogateAcquisitionCvt(unittest.TestCase):
    """E5.5 CVT acquisition policy contract."""

    def test_empty_cell_forces_eval(self) -> None:
        from worldspace.illuminators.cvt import generate_centroids
        from worldspace.illuminators.cvt_archive import CvtArchive

        archive = CvtArchive(generate_centroids(9, seed=0, lloyd_iterations=5))
        decision = decide(
            _config(),
            _prediction(0.0, 0.0),
            _target((3, 0)),
            archive,
        )
        self.assertEqual(decision.action, "eval")
        self.assertEqual(decision.reason, REASON_EMPTY_BIN_EXPLORE)

    def test_threshold_skip_on_filled_cvt_cell(self) -> None:
        from worldspace.illuminators.cvt import generate_centroids
        from worldspace.illuminators.cvt_archive import CvtArchive

        archive = CvtArchive(generate_centroids(9, seed=0, lloyd_iterations=5))
        archive.try_insert(ArchiveElite(bin=(2, 0), fitness=0.5))
        decision = decide(
            _config(),
            _prediction(0.1, 0.1),
            _target((2, 0)),
            archive,
        )
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, REASON_BELOW_FITNESS_THRESHOLD)

    def test_never_skip_empty_cell_false_allows_skip(self) -> None:
        from worldspace.illuminators.cvt import generate_centroids
        from worldspace.illuminators.cvt_archive import CvtArchive

        archive = CvtArchive(generate_centroids(9, seed=0, lloyd_iterations=5))
        decision = decide(
            _config(never_skip_empty_bin=False),
            _prediction(0.0, 0.0),
            _target((4, 0)),
            archive,
        )
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, REASON_BELOW_FITNESS_THRESHOLD)

    def test_decide_is_deterministic_on_cvt(self) -> None:
        from worldspace.illuminators.cvt import generate_centroids
        from worldspace.illuminators.cvt_archive import CvtArchive

        archive = CvtArchive(generate_centroids(9, seed=0, lloyd_iterations=5))
        archive.try_insert(ArchiveElite(bin=(1, 0), fitness=0.5))
        config = _config()
        prediction = _prediction(0.1, 0.2)
        target = _target((1, 0))
        first = decide(config, prediction, target, archive)
        second = decide(config, prediction, target, archive)
        self.assertEqual(first, second)

    def test_random_skip_respects_empty_bin_force_eval(self) -> None:
        config = AcquisitionConfig(
            mode="filter",
            policy="random_skip",
            random_skip_rate=1.0,
            never_skip_empty_bin=True,
        )
        decision = decide(
            config,
            _prediction(0.0, 0.0),
            _target((2, 2)),
            GridArchive(resolution=5),
            rng=np.random.default_rng(0),
        )
        self.assertEqual(decision.action, "eval")
        self.assertEqual(decision.reason, REASON_EMPTY_BIN_EXPLORE)

    def test_random_skip_rate_one_always_skips_filled(self) -> None:
        bin_ij = (0, 0)
        config = AcquisitionConfig(
            mode="filter",
            policy="random_skip",
            random_skip_rate=1.0,
            never_skip_empty_bin=True,
        )
        decision = decide(
            config,
            _prediction(0.9, 0.0),
            _target(bin_ij),
            _archive_filled_at(bin_ij),
            rng=np.random.default_rng(0),
        )
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, REASON_RANDOM_SKIP)
        self.assertEqual(decision.policy_version, RANDOM_SKIP_POLICY_VERSION)

    def test_random_skip_empirical_rate_near_target(self) -> None:
        bin_ij = (0, 0)
        config = AcquisitionConfig(
            mode="filter",
            policy="random_skip",
            random_skip_rate=0.335,
            never_skip_empty_bin=True,
        )
        archive = _archive_filled_at(bin_ij)
        target = _target(bin_ij)
        pred = _prediction(0.9, 0.0)
        rng = np.random.default_rng(123)
        skips = sum(
            1
            for _ in range(4000)
            if decide(config, pred, target, archive, rng=rng).action == "skip"
        )
        rate = skips / 4000.0
        self.assertGreater(rate, 0.30)
        self.assertLess(rate, 0.37)


if __name__ == "__main__":
    unittest.main()

"""Tests for full hold-out compose-gate alignment analysis."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from scripts.analyze_holdout_compose_gate_alignment import (
    _bootstrap_ci,
    _metrics,
    _nmae,
    _target_distribution,
)


class TestHoldoutComposeGateAlignment(unittest.TestCase):
    def test_metrics_and_bootstrap_ci_shape(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.uniform(0.0, 1.0, size=200)
        pred = y + rng.normal(0.0, 0.05, size=200)
        m = _metrics(y, pred)
        self.assertIn("r2_fitness", m)
        self.assertGreater(m["r2_fitness"], 0.5)
        lo, hi = _bootstrap_ci(
            y,
            pred,
            lambda yy, pp: float(r2_score(yy, pp)),
            b=200,
            random_state=1,
        )
        self.assertLess(lo, hi)
        self.assertLessEqual(lo, m["r2_fitness"])
        self.assertGreaterEqual(hi, m["r2_fitness"])

    def test_bootstrap_mae_ci(self) -> None:
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        pred = np.array([0.12, 0.18, 0.31, 0.39, 0.52])
        lo, hi = _bootstrap_ci(
            y,
            pred,
            lambda yy, pp: float(mean_absolute_error(yy, pp)),
            b=100,
            random_state=2,
        )
        point = float(mean_absolute_error(y, pred))
        self.assertLess(lo, hi)
        self.assertAlmostEqual(point, 0.016, places=3)

    def test_nmae_and_target_std(self) -> None:
        y = np.array([0.0, 0.0, 0.0, 0.8, 0.9])
        pred = np.array([0.0, 0.0, 0.1, 0.7, 0.85])
        m = _metrics(y, pred)
        dist = _target_distribution(y)
        self.assertGreater(dist["std"], 0.0)
        self.assertAlmostEqual(m["nmae_fitness"], _nmae(y, pred), places=6)
        self.assertAlmostEqual(
            m["nmae_fitness"], m["mae_fitness"] / dist["std"], places=6
        )
        self.assertGreater(dist["frac_zero"], 0.5)


if __name__ == "__main__":
    unittest.main()

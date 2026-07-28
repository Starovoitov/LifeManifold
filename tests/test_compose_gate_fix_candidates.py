"""Unit tests for offline compose-gate fix-candidate replay."""

from __future__ import annotations

import unittest

from scripts.analyze_compose_gate_fix_candidates import (
    ProposalView,
    confidence_gated_policies,
    localize_gray_zone,
    retarget_soft_tau,
)


def _view(
    *,
    p_ext: float,
    fit_a: float,
    fit_b: float,
    fit_soft: float,
    uncertainty: float = 0.1,
) -> ProposalView:
    return ProposalView(
        p_ext=p_ext,
        uncertainty=uncertainty,
        force_empty=False,
        fit_hard_a=fit_a,
        fit_hard_b=fit_b,
        fit_soft=fit_soft,
        logged_skip=False,
    )


class TestComposeGateFixCandidates(unittest.TestCase):
    def test_gray_zone_localization_out_of_band_zero_d1(self) -> None:
        # Outside band: both gates agree (both keep fitness).
        out = _view(p_ext=0.2, fit_a=0.6, fit_b=0.6, fit_soft=0.5)
        # In band: gate 0.5 zeros, gate 0.95 keeps → skip flips at τ=0.45.
        inn = _view(p_ext=0.7, fit_a=0.0, fit_b=0.6, fit_soft=0.2)
        stats = localize_gray_zone([out, inn], tau=0.45)
        self.assertEqual(stats["outside_gray_zone"]["d1"], 0.0)
        self.assertEqual(stats["gray_zone_p_ext_in_[0.5,0.95)"]["d1"], 1.0)
        self.assertEqual(stats["all"]["d1"], 0.5)

    def test_force_eval_gray_zone_zeroes_d1(self) -> None:
        views = [
            _view(p_ext=0.7, fit_a=0.0, fit_b=0.6, fit_soft=0.2),
            _view(p_ext=0.96, fit_a=0.0, fit_b=0.0, fit_soft=0.02),
        ]
        stats = confidence_gated_policies(views, tau=0.45)
        self.assertEqual(stats["force_eval_gray_zone"]["d1"], 0.0)
        self.assertEqual(stats["skip_only_if_p_ext_ge_0.95"]["d1"], 0.0)

    def test_soft_retarget_picks_near_target(self) -> None:
        # Soft fitness grid: half below 0.3, half above 0.5 → τ≈0.4 hits ~50%.
        views = [
            _view(p_ext=0.5, fit_a=0.0, fit_b=0.4, fit_soft=0.2) for _ in range(5)
        ] + [_view(p_ext=0.2, fit_a=0.7, fit_b=0.7, fit_soft=0.7) for _ in range(5)]
        out = retarget_soft_tau(views, target_skip=0.5, grid=(0.1, 0.3, 0.5, 0.7))
        self.assertIn(out["retargeted"]["tau"], (0.3, 0.5))
        self.assertEqual(out["d1_soft_gate0.5_vs_0.95"], 0.0)


if __name__ == "__main__":
    unittest.main()

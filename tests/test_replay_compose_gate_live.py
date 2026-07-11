"""Unit tests for live compose-gate D1 replay helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.replay_compose_gate_live import (
    aggregate,
    discover_seed_archives,
    replay_seed,
    would_skip,
)


def _record(
    *,
    p_ext: float,
    fitness_logged: float,
    uncertainty: float = 0.1,
    decision: str = "eval",
    reason: str = "accepted_for_eval",
    stability: float = 0.8,
    diversity: float = 0.7,
    final_density: float = 0.4,
) -> dict:
    comps = {
        "diversity": diversity,
        "early_extinction_prob": p_ext,
        "final_density": final_density,
        "oscillation_score": 0.2,
        "stability": stability,
        "topology_interface_index": 0.3,
        "topology_window_heterogeneity": 0.4,
    }
    return {
        "decision": decision,
        "decision_reason": reason,
        "emitter_type": "random",
        "prediction": {
            "components": comps,
            "measures": {"stability": stability, "diversity": diversity},
            "fitness": fitness_logged,
            "uncertainty": uncertainty,
        },
    }


class TestReplayComposeGateLive(unittest.TestCase):
    def test_would_skip_threshold(self) -> None:
        self.assertTrue(would_skip(0.4, 0.5))
        self.assertFalse(would_skip(0.5, 0.5))
        self.assertFalse(would_skip(0.4, 0.5, force_eval_empty=True))

    def test_divergent_when_pext_in_band(self) -> None:
        # p_ext in [0.5, 0.95): gate 0.5 zeros fitness; gate 0.95 keeps base.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "surrogate_archive.jsonl"
            # Non-extinct components with decent base fitness; logged fitness at 0.95.
            rec = _record(
                p_ext=0.7,
                fitness_logged=0.6,
                decision="eval",
                reason="accepted_for_eval",
            )
            path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
            stats = replay_seed(path)
            self.assertEqual(stats["n"], 1)
            self.assertEqual(stats["divergent_skip_fraction"], 1.0)
            self.assertEqual(stats["agree_logged_skip_vs_gate_0.95"], 1.0)

    def test_aggregate_confirmatory_rule(self) -> None:
        seed_stats = [
            {
                "n": 100,
                "divergent_skip_fraction": 0.02,
                "frac_pred_ext_p_in_[0.5,0.95)": 0.1,
                "mean_abs_pred_diff": 0.05,
                "agree_logged_skip_vs_gate_0.95": 1.0,
                "logged_empty_bin_explore_frac": 0.0,
                "logged_fitness_matches_recompose_0.95": 1.0,
            },
            {
                "n": 100,
                "divergent_skip_fraction": 0.04,
                "frac_pred_ext_p_in_[0.5,0.95)": 0.1,
                "mean_abs_pred_diff": 0.05,
                "agree_logged_skip_vs_gate_0.95": 1.0,
                "logged_empty_bin_explore_frac": 0.0,
                "logged_fitness_matches_recompose_0.95": 1.0,
            },
        ]
        agg = aggregate(seed_stats, min_predicted_fitness=0.45)
        self.assertAlmostEqual(agg["divergent_skip_fraction"], 0.03)
        self.assertAlmostEqual(agg["divergent_skip_fraction_at_min_fit_0.45"], 0.03)
        self.assertTrue(agg["RQ3_confirmatory_rule_div_le_0.05"])

    def test_discover_seed_archives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "seed_0").mkdir()
            (root / "seed_1").mkdir()
            a0 = root / "seed_0" / "surrogate_archive.jsonl"
            a1 = root / "seed_1" / "surrogate_archive.jsonl"
            a0.write_text("{}\n", encoding="utf-8")
            a1.write_text("{}\n", encoding="utf-8")
            found = discover_seed_archives(root)
            self.assertEqual(found, [a0, a1])


if __name__ == "__main__":
    unittest.main()

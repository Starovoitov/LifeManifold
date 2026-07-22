"""Unit tests for paired Vargha–Delaney A₁₂ in Q1 statistics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_q1_statistics import (
    run_v4_dungeon_statistics,
    vargha_delaney_a12_paired,
)


class VarghaDelaneyA12Tests(unittest.TestCase):
    def test_all_wins_greater(self) -> None:
        delta = np.array([1.0, 2.0, 0.5])
        self.assertAlmostEqual(
            vargha_delaney_a12_paired(delta, direction="greater"), 1.0
        )

    def test_all_losses_greater(self) -> None:
        delta = np.array([-1.0, -2.0])
        self.assertAlmostEqual(
            vargha_delaney_a12_paired(delta, direction="greater"), 0.0
        )

    def test_half_wins(self) -> None:
        delta = np.array([1.0, -1.0])
        self.assertAlmostEqual(
            vargha_delaney_a12_paired(delta, direction="greater"), 0.5
        )

    def test_ties_count_half(self) -> None:
        delta = np.array([0.0, 1.0])
        self.assertAlmostEqual(
            vargha_delaney_a12_paired(delta, direction="greater"), 0.75
        )

    def test_less_direction(self) -> None:
        delta = np.array([-0.1, 0.2, -0.3])
        self.assertAlmostEqual(
            vargha_delaney_a12_paired(delta, direction="less"), 2 / 3
        )

    def test_empty_returns_nan(self) -> None:
        self.assertTrue(np.isnan(vargha_delaney_a12_paired(np.array([]))))

    def test_v4_dungeon_family_uses_matched_anytime_auc(self) -> None:
        levels = {
            "genetic": 0.10,
            "genetic_filter": 0.20,
            "llm_stub": 0.20,
            "llm_hints": 0.30,
            "llm_hints_filter": 0.50,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for condition, level in levels.items():
                for seed in range(10):
                    run = root / condition / f"seed_{seed}"
                    run.mkdir(parents=True)
                    (run / "nightly_run_summary.json").write_text(
                        json.dumps({"seed": seed, "evaluations": 100})
                    )
                    rows = (
                        {"evaluations": 0, "coverage": 0.0, "qd_score": 0.0},
                        {
                            "evaluations": 100,
                            "coverage": level,
                            "qd_score": level * 100.0,
                        },
                    )
                    (run / "archive_trace.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in rows)
                    )
            result = run_v4_dungeon_statistics(root)
            self.assertEqual(result["n_seeds"], 10)
            self.assertEqual(result["m"], 8)
            self.assertTrue(result["family_pass"])


if __name__ == "__main__":
    unittest.main()

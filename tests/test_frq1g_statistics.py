"""Unit tests for F-RQ1g (G1 multi-LLM) statistics."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_q1_statistics import run_frq1g_provider, run_frq1g_statistics


def _write_g1_summary(path: Path, *, delta_cov: float, delta_fit: float) -> None:
    rows: list[dict[str, str]] = []
    for seed in range(10):
        rows.append(
            {
                "condition": "stub",
                "seed": str(seed),
                "coverage_pct": "40.0",
                "mean_best_fitness": "0.40",
            }
        )
        rows.append(
            {
                "condition": "hints",
                "seed": str(seed),
                "coverage_pct": str(40.0 + delta_cov),
                "mean_best_fitness": str(0.40 + delta_fit),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["condition", "seed", "coverage_pct", "mean_best_fitness"],
        )
        writer.writeheader()
        writer.writerows(rows)


class Frq1gStatisticsTests(unittest.TestCase):
    def test_run_frq1g_provider_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            _write_g1_summary(path, delta_cov=10.0, delta_fit=0.05)
            stats = run_frq1g_provider(
                path,
                provider="test-llm",
                tier="q1-test",
            )
            self.assertEqual(stats["verdict"], "PASS")
            self.assertTrue(stats["family_pass"])
            self.assertTrue(stats["holm_reject_cov"])
            self.assertTrue(stats["holm_reject_fit"])
            self.assertEqual(stats["sign_cov"], "10/10")

    def test_live_g1_providers_pass(self) -> None:
        stats = run_frq1g_statistics()
        self.assertTrue(stats["all_providers_pass"])
        for name in ("gpt-4o-mini", "deepseek-v4-pro"):
            self.assertEqual(stats["providers"][name]["verdict"], "PASS")


if __name__ == "__main__":
    unittest.main()

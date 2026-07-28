"""Unit tests for offline filter τ replay by emitter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_filter_threshold_emitter_replay import (
    aggregate_stratum,
    replay_tau_stratum,
)


def _record(
    *,
    emitter: str,
    fitness: float,
    uncertainty: float = 0.1,
    decision: str = "eval",
) -> dict:
    return {
        "emitter_type": emitter,
        "decision": decision,
        "decision_reason": "accepted_for_eval",
        "prediction": {
            "fitness": fitness,
            "uncertainty": uncertainty,
            "components": {"early_extinction_prob": 0.2},
            "measures": {},
        },
    }


class TestFilterThresholdEmitterReplay(unittest.TestCase):
    def test_replay_tau_stratum_respects_emitter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "surrogate_archive.jsonl"
            rows = [
                _record(emitter="llm", fitness=0.2),
                _record(emitter="llm", fitness=0.6),
                _record(emitter="random", fitness=0.2),
            ]
            path.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
            )

            llm = replay_tau_stratum(path, tau=0.45, emitter="llm")
            all_ = replay_tau_stratum(path, tau=0.45, emitter="all")

            self.assertEqual(llm["n"], 2)
            self.assertEqual(llm["skip_rate"], 0.5)
            self.assertEqual(all_["n"], 3)
            self.assertAlmostEqual(all_["skip_rate"], 2 / 3)

    def test_aggregate_stratum_pooled(self) -> None:
        per_seed = [
            {
                "n": 100,
                "skip_rate": 0.2,
                "logged_skip_rate": 0.2,
                "agree_replay_vs_logged_skip": 1.0,
            },
            {
                "n": 100,
                "skip_rate": 0.4,
                "logged_skip_rate": 0.4,
                "agree_replay_vs_logged_skip": 1.0,
            },
        ]
        agg = aggregate_stratum(per_seed)
        self.assertEqual(agg["n_proposals"], 200)
        self.assertAlmostEqual(agg["skip_rate_pooled"], 0.3)


if __name__ == "__main__":
    unittest.main()

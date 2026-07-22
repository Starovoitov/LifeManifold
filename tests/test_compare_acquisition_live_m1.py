"""Unit tests for M1 Phase 3 live archive acquisition counterfactual."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_acquisition_live_m1 import (
    aggregate_seeds,
    build_policies,
    discover_seed_archives,
    replay_seed,
)


def _record(
    *,
    fitness: float,
    uncertainty: float = 0.1,
    decision: str = "eval",
    reason: str = "accepted_for_eval",
    cell_id: int = 0,
    eval_fitness: float | None = None,
) -> dict:
    out: dict = {
        "decision": decision,
        "decision_reason": reason,
        "target_cell_id": cell_id,
        "prediction": {
            "components": {},
            "measures": {"stability": 0.5, "diversity": 0.5},
            "fitness": fitness,
            "uncertainty": uncertainty,
        },
        "eval_outcome": None,
    }
    if eval_fitness is not None:
        out["eval_outcome"] = {
            "accepted": False,
            "fitness": eval_fitness,
            "improved": False,
            "measures": {"stability": 0.5, "diversity": 0.5},
        }
    return out


class TestCompareAcquisitionLiveM1(unittest.TestCase):
    def test_threshold_matches_logged_skip(self) -> None:
        policies = build_policies(
            min_predicted_fitness=0.45,
            max_uncertainty_to_skip=1.0,
            betas=[1.0],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "surrogate_archive.jsonl"
            rows = [
                _record(fitness=0.2, decision="skip", reason="below_fitness_threshold"),
                _record(
                    fitness=0.6,
                    decision="eval",
                    reason="accepted_for_eval",
                    eval_fitness=0.7,
                ),
            ]
            path.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
            )
            stats = replay_seed(path, policies)
            tg = stats["policies"][0]
            self.assertEqual(tg["policy"], "threshold_gate")
            self.assertEqual(tg["agree_logged_rate"], 1.0)
            self.assertEqual(tg["skip_count"], 1)

    def test_ucb_promotes_borderline_skip_to_eval(self) -> None:
        # μ=0.40, σ=0.10 → threshold skips; UCB β=1 → 0.50 ≥ 0.45 → eval
        policies = build_policies(
            min_predicted_fitness=0.45,
            max_uncertainty_to_skip=1.0,
            betas=[1.0],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "surrogate_archive.jsonl"
            rec = _record(
                fitness=0.40,
                uncertainty=0.10,
                decision="skip",
                reason="below_fitness_threshold",
            )
            path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
            stats = replay_seed(path, policies)
            ucb = stats["policies"][1]
            self.assertEqual(ucb["policy"], "ucb_promote")
            self.assertEqual(ucb["skip_count"], 0)
            self.assertEqual(ucb["flip_skip_to_eval"], 1)
            self.assertEqual(ucb["flip_eval_to_skip"], 0)

    def test_false_skip_proxy_on_eval_to_skip(self) -> None:
        # Force ucb skip on an evaluated high-fitness row by β=0 (degenerates to μ gate)
        # Using β=0 via replace isn't in build_policies; craft low μ high logged eval.
        # With β=1.0, UCB never adds skips when σ≥0 — use μ=0.2 logged as eval somehow
        # Impossible for ucb to flip eval→skip if logged used threshold with same μ.
        # So test the counter: when policy skips and actual ≥ min_fit.
        policies = build_policies(
            min_predicted_fitness=0.45,
            max_uncertainty_to_skip=1.0,
            betas=[0.0],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "surrogate_archive.jsonl"
            # Logged as eval (e.g. different historical policy), μ low → UCB β=0 skips
            rec = _record(
                fitness=0.2,
                uncertainty=0.5,
                decision="eval",
                reason="accepted_for_eval",
                eval_fitness=0.8,
            )
            path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
            stats = replay_seed(path, policies)
            ucb = stats["policies"][1]
            self.assertEqual(ucb["flip_eval_to_skip"], 1)
            self.assertEqual(ucb["false_skip_count_on_eval_to_skip"], 1)

    def test_aggregate_and_discover(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "seed_0").mkdir()
            (root / "seed_1").mkdir()
            a0 = root / "seed_0" / "surrogate_archive.jsonl"
            a1 = root / "seed_1" / "surrogate_archive.jsonl"
            rec = _record(
                fitness=0.2, decision="skip", reason="below_fitness_threshold"
            )
            a0.write_text(json.dumps(rec) + "\n", encoding="utf-8")
            a1.write_text(json.dumps(rec) + "\n", encoding="utf-8")
            found = discover_seed_archives(root)
            self.assertEqual(found, [a0, a1])
            policies = build_policies(
                min_predicted_fitness=0.45,
                max_uncertainty_to_skip=1.0,
                betas=[0.15],
            )
            seeds = [replay_seed(p, policies) for p in found]
            for i, s in enumerate(seeds):
                s["seed"] = f"seed_{i}"
            agg = aggregate_seeds(seeds)
            self.assertEqual(agg["n_seeds"], 2)
            self.assertEqual(agg["n_proposals"], 2)
            self.assertAlmostEqual(agg["logged_skip_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

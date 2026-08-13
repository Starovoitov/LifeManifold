"""Unit tests for per-run evaluated-proposal JSONL logging."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from worldspace.illuminators.archive import InsertResult
from worldspace.illuminators.evaluation import EvalResult
from worldspace.illuminators.proposal_log import (
    configure_proposal_log,
    insert_outcome_label,
    open_proposal_log,
    proposal_log_enabled_for_emitter,
    resolve_proposal_log_path,
    serialize_proposal_record,
)
from worldspace.illuminators.scheduler import TargetBin
from worldspace.metrics import WorldMetrics
from worldspace.specs.spec import WorldSpec
from worldspace.surrogate.types import SurrogatePrediction


def _spec() -> WorldSpec:
    return WorldSpec(
        birth=[3],
        survival=[2, 3],
        noise=0.01,
        resource_regen=0.1,
        predation=0.1,
        cell_types=["life", "food"],
        grid_size=50,
        steps=200,
        seed=0,
    )


def _eval(fitness: float = 0.4) -> EvalResult:
    metrics = WorldMetrics(
        entropy=0.0,
        stability=0.5,
        average_lifespan=0.0,
        density_mean=0.2,
        oscillation_score=0.1,
        diversity=0.5,
        mo_eoc_indicator=0.0,
        topology_interface_index=0.1,
        topology_window_heterogeneity=0.1,
        compressibility_score=0.0,
        ecology_state_entropy_norm=0.0,
        ecology_resource_adjacency=0.0,
    )
    return EvalResult(
        world_spec=_spec(),
        metrics=metrics,
        measures={"stability": 0.5, "diversity": 0.5},
        fitness=fitness,
        bin=(10, 10),
        early_extinct=False,
    )


class ProposalLogTests(unittest.TestCase):
    def tearDown(self) -> None:
        configure_proposal_log(None)

    def test_env_zero_disables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"LIFEMANIFOLD_PROPOSAL_LOG": "0"}):
                self.assertIsNone(resolve_proposal_log_path(output_dir=tmp))

    def test_default_path_under_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("LIFEMANIFOLD_PROPOSAL_LOG", None)
                path = resolve_proposal_log_path(output_dir=tmp)
                self.assertEqual(path, Path(tmp) / "proposal_log.jsonl")

    def test_llm_only_filter_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LIFEMANIFOLD_PROPOSAL_LOG_ALL_EMITTERS", None)
            self.assertTrue(proposal_log_enabled_for_emitter("llm"))
            self.assertTrue(proposal_log_enabled_for_emitter("llm_rewrite"))
            self.assertFalse(proposal_log_enabled_for_emitter("genetic"))

    def test_all_emitters_env(self) -> None:
        with patch.dict(
            os.environ, {"LIFEMANIFOLD_PROPOSAL_LOG_ALL_EMITTERS": "1"}, clear=False
        ):
            self.assertTrue(proposal_log_enabled_for_emitter("genetic"))

    def test_outcome_labels(self) -> None:
        self.assertEqual(
            insert_outcome_label(
                InsertResult(accepted=True, improved=False, rejected=False)
            ),
            "fill_empty",
        )
        self.assertEqual(
            insert_outcome_label(
                InsertResult(accepted=True, improved=True, rejected=False)
            ),
            "improve",
        )
        self.assertEqual(
            insert_outcome_label(
                InsertResult(accepted=False, improved=False, rejected=True)
            ),
            "occupied_not_better",
        )

    def test_writer_appends_llm_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proposal_log.jsonl"
            writer = open_proposal_log(path, run_id="run1", enabled=True, flush_every=1)
            insert = InsertResult(accepted=False, improved=False, rejected=True)
            writer.append_evaluated(
                iteration=1,
                candidate_id=42,
                emitter_type="llm",
                target=TargetBin(
                    bin=(3, 4), target_stability=0.5, target_diversity=0.5
                ),
                target_cell_id=154,
                eval_result=_eval(0.3),
                insert=insert,
                parent_id="parent-1",
                incumbent_fitness=0.55,
                prediction=None,
            )
            writer.close()
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["outcome"], "occupied_not_better")
            self.assertTrue(row["rejected"])
            self.assertEqual(row["incumbent_fitness"], 0.55)
            self.assertIn("world_spec", row)
            self.assertEqual(row["emitter_type"], "llm")

    def test_writer_skips_genetic_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proposal_log.jsonl"
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("LIFEMANIFOLD_PROPOSAL_LOG_ALL_EMITTERS", None)
                writer = open_proposal_log(
                    path, run_id="run1", enabled=True, flush_every=1
                )
                writer.append_evaluated(
                    iteration=1,
                    candidate_id=1,
                    emitter_type="genetic",
                    target=TargetBin(
                        bin=(0, 0), target_stability=0.1, target_diversity=0.1
                    ),
                    target_cell_id=0,
                    eval_result=_eval(),
                    insert=InsertResult(accepted=True, improved=False, rejected=False),
                )
                writer.close()
            self.assertFalse(path.is_file())

    def test_serialize_includes_schema(self) -> None:
        record = serialize_proposal_record(
            run_id="r",
            iteration=2,
            candidate_id=9,
            emitter_type="llm",
            target=TargetBin(bin=(1, 2), target_stability=0.2, target_diversity=0.3),
            target_cell_id=52,
            eval_result=_eval(),
            insert=InsertResult(accepted=True, improved=False, rejected=False),
            parent_id=None,
            incumbent_fitness=None,
            prediction=None,
        )
        self.assertEqual(record["schema_version"], "2.0")
        self.assertIn("ts_utc", record)
        self.assertEqual(record["outcome"], "fill_empty")

    def test_serialize_includes_llm_audit_link_and_parent(self) -> None:
        prediction = SurrogatePrediction(
            components={},
            measures={},
            fitness=0.42,
            uncertainty=0.08,
        )
        record = serialize_proposal_record(
            run_id="r",
            iteration=2,
            candidate_id=9,
            emitter_type="llm",
            target=TargetBin(bin=(1, 2), target_stability=0.2, target_diversity=0.3),
            target_cell_id=52,
            eval_result=_eval(),
            insert=InsertResult(accepted=True, improved=False, rejected=False),
            parent_id="p",
            incumbent_fitness=0.4,
            prediction=None,
            parent_world_spec=_spec(),
            llm_call_id="call-1",
            llm_parse_outcome="valid",
            scalar_treatment="shuffled",
            prompt_prediction=prediction,
            source_prediction=SurrogatePrediction(
                components={},
                measures={},
                fitness=0.7,
                uncertainty=0.1,
            ),
            target_selection="uniform_frontier",
        )
        self.assertEqual(record["llm_call_id"], "call-1")
        self.assertEqual(record["llm_parse_outcome"], "valid")
        self.assertEqual(record["scalar_treatment"], "shuffled")
        self.assertEqual(record["target_selection"], "uniform_frontier")
        self.assertIsNotNone(record["parent_world_spec_hash"])
        self.assertEqual(record["prompt_prediction"]["fitness"], 0.42)


if __name__ == "__main__":
    unittest.main()

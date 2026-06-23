"""Unit tests for SurrogateArchive writer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldspace.illuminators.archive import InsertResult
from worldspace.illuminators.evaluation import EvalResult
from worldspace.illuminators.scheduler import TargetBin
from worldspace.surrogate.acquisition import AcquisitionDecision
from worldspace.surrogate.canonical_hash import world_spec_canonical_hash
from worldspace.surrogate.surrogate import StubSurrogate
from worldspace.surrogate.surrogate_archive import (
    NoOpSurrogateArchiveWriter,
    SurrogateArchiveWriter,
    build_archive_record,
    open_surrogate_archive,
    serialize_eval_outcome,
    serialize_prediction,
)
from worldspace.surrogate.types import SurrogatePrediction
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec

from tests.test_dashboard_surrogate_archive import _sample_record

_LOW_COMPONENTS = {
    "stability": 0.1,
    "diversity": 0.1,
    "oscillation_score": 0.1,
    "topology_interface_index": 0.1,
    "topology_window_heterogeneity": 0.1,
    "final_density": 0.1,
    "early_extinction_prob": 0.1,
}

_SPEC = WorldSpec(
    birth=[1, 3],
    survival=[2, 3],
    noise=0.02,
    resource_regen=0.05,
    predation=0.1,
    cell_types=CANONICAL_CELL_TYPES.copy(),
    grid_size=8,
    steps=200,
    seed=0,
)


class TestSurrogateArchive(unittest.TestCase):
    def test_round_trip_record_matches_dashboard_loader(self) -> None:
        from dashboard.components.surrogate_archive_loader import (
            try_flatten_archive_record,
        )

        record = _sample_record()
        flat = try_flatten_archive_record(record)
        assert flat is not None
        self.assertEqual(flat["decision"], "skip")
        self.assertEqual(flat["target_bin_i"], 2)

    def test_writer_skip_and_eval_lines(self) -> None:
        target = TargetBin(bin=(1, 2), target_stability=0.5, target_diversity=0.5)
        prediction = StubSurrogate(mean=0.2, uncertainty=0.1).predict(_SPEC)
        decision = AcquisitionDecision(
            action="skip",
            reason="below_fitness_threshold",
            policy_version="threshold_gate_v1",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "surrogate_archive.jsonl"
            writer = SurrogateArchiveWriter(
                path=path, run_id="run-test", flush_every=32
            )
            writer.append_slot(
                iteration=1,
                candidate_id=0,
                emitter_type="random",
                target=target,
                target_cell_id=7,
                world_spec_hash=world_spec_canonical_hash(_SPEC),
                prediction=prediction,
                decision=decision,
                acquisition_mode="filter",
            )
            writer.flush()
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            parsed = json.loads(lines[0])
            self.assertIsNone(parsed["eval_outcome"])
            self.assertEqual(parsed["target_cell_id"], 7)

    def test_no_op_writer_creates_no_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "surrogate_archive.jsonl"
            writer = open_surrogate_archive(path, run_id="x", enabled=False)
            self.assertIsInstance(writer, NoOpSurrogateArchiveWriter)
            writer.append_slot(
                iteration=1,
                candidate_id=0,
                emitter_type="random",
                target=TargetBin(
                    bin=(0, 0), target_stability=0.5, target_diversity=0.5
                ),
                target_cell_id=0,
                world_spec_hash="abc",
                prediction=StubSurrogate(0.5, 0.5).predict(_SPEC),
                decision=AcquisitionDecision(
                    action="eval",
                    reason="accepted_for_eval",
                    policy_version="threshold_gate_v1",
                ),
                acquisition_mode="off",
            )
            writer.close()
            self.assertFalse(path.exists())

    def test_canonical_hash_matches_facade_cache_key(self) -> None:
        from worldspace.surrogate.surrogate import build_surrogate_facade
        from unittest import mock

        model = mock.MagicMock()
        model.predict_components.return_value = dict(_LOW_COMPONENTS)
        model.predict_uncertainty.return_value = 0.2
        facade = build_surrogate_facade(model, uncertainty_fallback=0.5)
        spec = _SPEC
        facade.predict(spec)
        self.assertEqual(world_spec_canonical_hash(spec), facade._cache_key(spec))

    def test_serialize_eval_outcome(self) -> None:
        from worldspace.metrics import WorldMetrics

        metrics = WorldMetrics(
            entropy=0.1,
            stability=0.3,
            average_lifespan=0.2,
            density_mean=0.5,
            oscillation_score=0.1,
            diversity=0.4,
            mo_eoc_indicator=0.1,
            topology_interface_index=0.2,
            topology_window_heterogeneity=0.2,
            compressibility_score=0.1,
            ecology_state_entropy_norm=0.1,
            ecology_resource_adjacency=0.1,
            langton_lambda_runtime=0.1,
        )
        eval_result = EvalResult(
            world_spec=_SPEC,
            metrics=metrics,
            measures={"stability": 0.3, "diversity": 0.4},
            fitness=0.5,
            bin=(1, 2),
            early_extinct=False,
        )
        payload = serialize_eval_outcome(
            eval_result,
            InsertResult(accepted=True, improved=True, rejected=False),
        )
        self.assertTrue(payload["accepted"])
        self.assertAlmostEqual(payload["fitness"], 0.5)

    def test_build_archive_record_json_serializable(self) -> None:
        record = build_archive_record(
            run_id="r",
            iteration=0,
            candidate_id=0,
            emitter_type="random",
            target=TargetBin(bin=(0, 0), target_stability=0.5, target_diversity=0.5),
            target_cell_id=0,
            world_spec_hash="h",
            prediction=SurrogatePrediction(
                components={},
                measures={},
                fitness=0.1,
                uncertainty=0.2,
            ),
            decision=AcquisitionDecision(
                action="eval",
                reason="accepted_for_eval",
                policy_version="threshold_gate_v1",
            ),
            acquisition_mode="shadow",
            eval_result=None,
            insert=None,
        )
        json.dumps(record)
        prediction = SurrogatePrediction(
            components=dict(_LOW_COMPONENTS),
            measures={"stability": 0.1, "diversity": 0.1},
            fitness=0.1,
            uncertainty=0.2,
        )
        self.assertEqual(serialize_prediction(prediction)["fitness"], 0.1)


if __name__ == "__main__":
    unittest.main()

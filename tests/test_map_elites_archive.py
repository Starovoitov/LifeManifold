"""Unit tests for MAP-Elites grid archive."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from worldspace.illuminators.archive import (
    ARCHIVE_SCHEMA_VERSION,
    DEFAULT_GRID_RESOLUTION,
    ArchiveElite,
    EliteMetadata,
    GridArchive,
    InsertResult,
    append_archive_line,
    elite_from_eval,
    elite_to_archive_record,
    insert_and_persist,
    insert_evaluated,
    new_elite_metadata,
)
from worldspace.illuminators.evaluation import evaluate_candidate
from worldspace.metrics import METRIC_KEYS, WorldMetrics
from worldspace.specs.spec import WorldSpec

_BASE_SPEC = WorldSpec(
    birth=[1],
    survival=[2],
    noise=0.0,
    resource_regen=0.0,
    predation=0.0,
    cell_types=["life", "food"],
    grid_size=4,
    steps=200,
    seed=0,
)


def _minimal_elite(
    bin_coord: tuple[int, int],
    fitness: float,
    *,
    elite_id: str = "test-id",
) -> ArchiveElite:
    return ArchiveElite(
        bin=bin_coord,
        fitness=fitness,
        world_spec=replace(_BASE_SPEC, seed=1),
        measures={"stability": 0.5, "diversity": 0.5},
        metadata=new_elite_metadata(
            generated_by="random",
            emitter_type="random",
            elite_id=elite_id,
            timestamp="2026-01-01T00:00:00+00:00",
        ),
    )


def _example_metrics(**overrides: float) -> WorldMetrics:
    defaults: dict[str, float] = {
        "entropy": 0.5,
        "stability": 0.55,
        "average_lifespan": 1.0,
        "density_mean": 0.3,
        "oscillation_score": 0.2,
        "diversity": 0.68,
        "mo_eoc_indicator": 0.5,
        "topology_interface_index": 0.3,
        "topology_window_heterogeneity": 0.4,
        "compressibility_score": 0.5,
        "ecology_state_entropy_norm": 0.6,
        "ecology_resource_adjacency": 0.2,
    }
    defaults.update(overrides)
    return WorldMetrics(
        entropy=defaults["entropy"],
        stability=defaults["stability"],
        average_lifespan=defaults["average_lifespan"],
        density_mean=defaults["density_mean"],
        oscillation_score=defaults["oscillation_score"],
        diversity=defaults["diversity"],
        mo_eoc_indicator=defaults["mo_eoc_indicator"],
        topology_interface_index=defaults["topology_interface_index"],
        topology_window_heterogeneity=defaults["topology_window_heterogeneity"],
        compressibility_score=defaults["compressibility_score"],
        ecology_state_entropy_norm=defaults["ecology_state_entropy_norm"],
        ecology_resource_adjacency=defaults["ecology_resource_adjacency"],
    )


class TestGridArchiveStructure(unittest.TestCase):
    def test_grid_archive_size_50(self) -> None:
        archive = GridArchive(DEFAULT_GRID_RESOLUTION)
        self.assertEqual(archive.resolution, 50)
        self.assertEqual(archive.filled_count(), 0)
        self.assertEqual(archive.empty_count(), 2500)

    def test_bc_range_fixed(self) -> None:
        archive = GridArchive(10)
        self.assertEqual(archive.bc_min, 0.0)
        self.assertEqual(archive.bc_max, 1.0)

    def test_invalid_bin_raises(self) -> None:
        archive = GridArchive(5)
        with self.assertRaises(IndexError):
            archive.get(-1, 0)
        with self.assertRaises(IndexError):
            archive.is_empty(5, 0)

    def test_resolution_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            GridArchive(0)


class TestGridArchiveInsert(unittest.TestCase):
    def test_insert_empty_accepts_zero_fitness(self) -> None:
        archive = GridArchive(5)
        elite = _minimal_elite((2, 3), 0.0)
        result = archive.try_insert(elite)
        self.assertEqual(result, InsertResult(accepted=True, improved=False, rejected=False))
        stored = archive.get(2, 3)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.fitness, 0.0)

    def test_insert_improves_strict(self) -> None:
        archive = GridArchive(5)
        archive.try_insert(_minimal_elite((1, 1), 0.5, elite_id="a"))
        improved = archive.try_insert(_minimal_elite((1, 1), 0.6, elite_id="b"))
        self.assertEqual(improved, InsertResult(accepted=True, improved=True, rejected=False))
        stored = archive.get(1, 1)
        assert stored is not None
        self.assertEqual(stored.fitness, 0.6)

    def test_insert_equal_fitness_rejected(self) -> None:
        archive = GridArchive(5)
        archive.try_insert(_minimal_elite((0, 0), 0.5))
        result = archive.try_insert(_minimal_elite((0, 0), 0.5, elite_id="other"))
        self.assertEqual(result, InsertResult(accepted=False, improved=False, rejected=True))

    def test_insert_lower_fitness_rejected(self) -> None:
        archive = GridArchive(5)
        archive.try_insert(_minimal_elite((4, 4), 0.8, elite_id="keep"))
        result = archive.try_insert(_minimal_elite((4, 4), 0.2, elite_id="lose"))
        self.assertTrue(result.rejected)
        stored = archive.get(4, 4)
        assert stored is not None
        self.assertEqual(stored.fitness, 0.8)
        assert stored.metadata is not None
        self.assertEqual(stored.metadata.id, "keep")

    def test_insert_reject_does_not_mutate(self) -> None:
        archive = GridArchive(5)
        original = _minimal_elite((2, 2), 0.7, elite_id="orig")
        archive.try_insert(original)
        archive.try_insert(_minimal_elite((2, 2), 0.1, elite_id="new"))
        stored = archive.get(2, 2)
        assert stored is not None
        self.assertEqual(stored.fitness, 0.7)
        assert stored.metadata is not None
        self.assertEqual(stored.metadata.id, "orig")

    def test_filled_and_empty_counts(self) -> None:
        archive = GridArchive(4)
        self.assertEqual(archive.empty_count(), 16)
        archive.try_insert(_minimal_elite((0, 0), 0.1))
        archive.try_insert(_minimal_elite((3, 3), 0.2))
        self.assertEqual(archive.filled_count(), 2)
        self.assertEqual(archive.empty_count(), 14)


class TestInsertEvaluated(unittest.TestCase):
    def test_elite_from_eval_copies_fields(self) -> None:
        spec = replace(_BASE_SPEC, steps=220, seed=3)
        eval_result = evaluate_candidate(spec, resolution=10)
        metadata = new_elite_metadata(
            generated_by="genetic",
            emitter_type="genetic",
            parent_id="parent-1",
            elite_id="elite-1",
            timestamp="2026-05-19T12:00:00+00:00",
        )
        elite = elite_from_eval(eval_result, metadata)
        self.assertEqual(elite.bin, eval_result.bin)
        self.assertEqual(elite.fitness, eval_result.fitness)
        self.assertEqual(elite.measures, eval_result.measures)
        self.assertIs(elite.world_spec, eval_result.world_spec)
        assert elite.metadata is not None
        self.assertEqual(elite.metadata.id, "elite-1")

    def test_insert_evaluated_empty_cell(self) -> None:
        archive = GridArchive(10)
        eval_result = evaluate_candidate(replace(_BASE_SPEC, seed=5), resolution=10)
        metadata = new_elite_metadata(generated_by="random", emitter_type="random")
        result = insert_evaluated(archive, eval_result, metadata)
        self.assertTrue(result.accepted)
        stored = archive.get(*eval_result.bin)
        assert stored is not None
        assert stored.world_spec is not None
        self.assertEqual(stored.world_spec.seed, eval_result.world_spec.seed)

    def test_insert_evaluated_bin_out_of_range(self) -> None:
        archive = GridArchive(5)
        eval_result = evaluate_candidate(_BASE_SPEC, resolution=5)
        eval_result.bin = (9, 9)
        metadata = new_elite_metadata(generated_by="random", emitter_type="random")
        with self.assertRaises(IndexError):
            insert_evaluated(archive, eval_result, metadata)


class TestArchiveJsonl(unittest.TestCase):
    def test_elite_to_archive_record_schema(self) -> None:
        elite = _minimal_elite((12, 34), 0.81, elite_id="uuid-1")
        elite.metrics = _example_metrics()
        record = elite_to_archive_record(elite)
        self.assertEqual(record["schema_version"], ARCHIVE_SCHEMA_VERSION)
        self.assertEqual(record["bin"], [12, 34])
        self.assertEqual(record["fitness"], 0.81)
        self.assertEqual(set(record["measures"].keys()), {"stability", "diversity"})
        self.assertIn("seed", record["world_spec"])
        self.assertEqual(set(record["metrics"].keys()), set(METRIC_KEYS))
        self.assertEqual(record["metadata"]["id"], "uuid-1")

    def test_append_archive_line_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "archive.jsonl"
            record = elite_to_archive_record(_minimal_elite((0, 0), 0.5))
            append_archive_line(path, record)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            parsed = json.loads(lines[0])
            self.assertEqual(parsed["schema_version"], ARCHIVE_SCHEMA_VERSION)

    def test_insert_and_persist_rejected_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "archive.jsonl"
            archive = GridArchive(10)
            spec = replace(_BASE_SPEC, seed=11)
            first = evaluate_candidate(spec, resolution=10)
            meta_a = new_elite_metadata(
                generated_by="random", emitter_type="random", elite_id="a"
            )
            insert_and_persist(archive, first, meta_a, path)

            worse = evaluate_candidate(spec, resolution=10)
            worse.bin = first.bin
            worse.fitness = first.fitness - 0.1
            meta_b = new_elite_metadata(
                generated_by="random", emitter_type="random", elite_id="b"
            )
            result = insert_and_persist(archive, worse, meta_b, path)
            self.assertTrue(result.rejected)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)

    def test_insert_and_persist_improve_appends_second_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "archive.jsonl"
            archive = GridArchive(10)
            spec = replace(_BASE_SPEC, seed=17)
            first = evaluate_candidate(spec, resolution=10)
            meta_a = new_elite_metadata(
                generated_by="random", emitter_type="random", elite_id="first"
            )
            insert_and_persist(archive, first, meta_a, path)

            better = evaluate_candidate(spec, resolution=10)
            better.bin = first.bin
            better.fitness = min(1.0, first.fitness + 0.2)
            meta_b = new_elite_metadata(
                generated_by="genetic", emitter_type="genetic", elite_id="second"
            )
            result = insert_and_persist(archive, better, meta_b, path)
            self.assertTrue(result.accepted)
            self.assertTrue(result.improved)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            second = json.loads(lines[1])
            self.assertEqual(second["metadata"]["id"], "second")
            stored = archive.get(*first.bin)
            assert stored is not None
            assert stored.metadata is not None
            self.assertEqual(stored.metadata.id, "second")


if __name__ == "__main__":
    unittest.main()

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
    GridArchive,
    InsertResult,
    append_archive_line,
    archive_record_to_elite,
    elite_from_eval,
    elite_to_archive_record,
    insert_and_persist,
    insert_evaluated,
    count_archive_jsonl_lines,
    load_and_collapse_jsonl,
    merge_archives,
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


def _record_for_bin(
    bin_coord: tuple[int, int],
    fitness: float,
    *,
    elite_id: str,
) -> dict:
    return elite_to_archive_record(
        _minimal_elite(bin_coord, fitness, elite_id=elite_id)
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


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
        self.assertEqual(
            result, InsertResult(accepted=True, improved=False, rejected=False)
        )
        stored = archive.get(2, 3)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.fitness, 0.0)

    def test_insert_improves_strict(self) -> None:
        archive = GridArchive(5)
        archive.try_insert(_minimal_elite((1, 1), 0.5, elite_id="a"))
        improved = archive.try_insert(_minimal_elite((1, 1), 0.6, elite_id="b"))
        self.assertEqual(
            improved, InsertResult(accepted=True, improved=True, rejected=False)
        )
        stored = archive.get(1, 1)
        assert stored is not None
        self.assertEqual(stored.fitness, 0.6)

    def test_insert_equal_fitness_rejected(self) -> None:
        archive = GridArchive(5)
        archive.try_insert(_minimal_elite((0, 0), 0.5))
        result = archive.try_insert(_minimal_elite((0, 0), 0.5, elite_id="other"))
        self.assertEqual(
            result, InsertResult(accepted=False, improved=False, rejected=True)
        )

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
        self.assertEqual(record["metadata"]["prompt_version"], "")

    def test_prompt_version_roundtrip(self) -> None:
        elite = _minimal_elite((1, 2), 0.5, elite_id="pv-test")
        assert elite.metadata is not None
        elite.metadata = new_elite_metadata(
            generated_by="llm",
            emitter_type="llm",
            elite_id="pv-test",
            timestamp="2026-01-01T00:00:00+00:00",
            prompt_version="abc123",
        )
        record = elite_to_archive_record(elite)
        self.assertEqual(record["metadata"]["prompt_version"], "abc123")
        restored = archive_record_to_elite(record)
        assert restored.metadata is not None
        self.assertEqual(restored.metadata.prompt_version, "abc123")

    def test_normalize_archive_record_metadata_converts_null_prompt_version(
        self,
    ) -> None:
        from worldspace.illuminators.archive import normalize_archive_record_metadata

        record = elite_to_archive_record(_minimal_elite((0, 0), 0.5))
        record["metadata"]["prompt_version"] = None
        normalized = normalize_archive_record_metadata(record)
        self.assertEqual(normalized["metadata"]["prompt_version"], "")

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


class TestArchiveRecordRoundtrip(unittest.TestCase):
    def test_archive_record_roundtrip(self) -> None:
        elite = _minimal_elite((5, 7), 0.72, elite_id="roundtrip-id")
        elite.metrics = _example_metrics()
        record = elite_to_archive_record(elite)
        restored = archive_record_to_elite(record)
        self.assertEqual(restored.bin, elite.bin)
        self.assertEqual(restored.fitness, elite.fitness)
        self.assertEqual(restored.measures, elite.measures)
        assert restored.metadata is not None
        assert elite.metadata is not None
        self.assertEqual(restored.metadata.id, elite.metadata.id)
        assert restored.world_spec is not None
        assert elite.world_spec is not None
        self.assertEqual(restored.world_spec.seed, elite.world_spec.seed)
        assert restored.metrics is not None
        self.assertEqual(
            restored.metrics.as_vector().tolist(),
            elite.metrics.as_vector().tolist(),
        )


class TestLoadAndCollapseJsonl(unittest.TestCase):
    def test_collapse_keeps_max_fitness_per_bin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "archive.jsonl"
            _write_jsonl(
                path,
                [
                    _record_for_bin((2, 3), 0.3, elite_id="low"),
                    _record_for_bin((2, 3), 0.8, elite_id="high"),
                    _record_for_bin((2, 3), 0.5, elite_id="mid"),
                ],
            )
            archive = load_and_collapse_jsonl(path, resolution=10)
            stored = archive.get(2, 3)
            assert stored is not None
            assert stored.metadata is not None
            self.assertEqual(stored.fitness, 0.8)
            self.assertEqual(stored.metadata.id, "high")

    def test_collapse_tie_keeps_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "archive.jsonl"
            _write_jsonl(
                path,
                [
                    _record_for_bin((1, 1), 0.6, elite_id="first"),
                    _record_for_bin((1, 1), 0.6, elite_id="second"),
                ],
            )
            archive = load_and_collapse_jsonl(path, resolution=5)
            stored = archive.get(1, 1)
            assert stored is not None
            assert stored.metadata is not None
            self.assertEqual(stored.metadata.id, "first")

    def test_collapse_skip_invalid_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "archive.jsonl"
            path.write_text(
                "not json\n"
                + json.dumps(_record_for_bin((0, 0), 0.4, elite_id="ok"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertLogs("worldspace.illuminators.archive", level="WARNING"):
                archive = load_and_collapse_jsonl(path, resolution=5)
                line_count = count_archive_jsonl_lines(path)
            stored = archive.get(0, 0)
            assert stored is not None
            self.assertEqual(stored.fitness, 0.4)
            self.assertEqual(line_count, 1)
            self.assertEqual(archive.filled_count(), line_count)

    def test_collapse_skip_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "archive.jsonl"
            record_line = json.dumps(_record_for_bin((0, 0), 0.4, elite_id="ok"))
            path.write_text(f"\n{record_line}\n\n\n", encoding="utf-8")
            archive = load_and_collapse_jsonl(path, resolution=5)
            line_count = count_archive_jsonl_lines(path)
            stored = archive.get(0, 0)
            assert stored is not None
            self.assertEqual(stored.fitness, 0.4)
            self.assertEqual(line_count, 1)
            self.assertEqual(archive.filled_count(), line_count)

    def test_collapse_raise_on_invalid_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text("{broken\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_and_collapse_jsonl(path, resolution=5, on_invalid_line="raise")

    def test_load_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_and_collapse_jsonl("/nonexistent/archive.jsonl", resolution=5)


class TestMergeArchives(unittest.TestCase):
    def test_merge_prefers_higher_fitness(self) -> None:
        base = GridArchive(5)
        incoming = GridArchive(5)
        base.try_insert(_minimal_elite((2, 2), 0.4, elite_id="base"))
        incoming.try_insert(_minimal_elite((2, 2), 0.9, elite_id="incoming"))
        merge_archives(base, incoming)
        stored = base.get(2, 2)
        assert stored is not None
        assert stored.metadata is not None
        self.assertEqual(stored.fitness, 0.9)
        self.assertEqual(stored.metadata.id, "incoming")

    def test_merge_keeps_base_when_incoming_worse(self) -> None:
        base = GridArchive(5)
        incoming = GridArchive(5)
        base.try_insert(_minimal_elite((3, 3), 0.85, elite_id="base"))
        incoming.try_insert(_minimal_elite((3, 3), 0.2, elite_id="lose"))
        merge_archives(base, incoming)
        stored = base.get(3, 3)
        assert stored is not None
        assert stored.metadata is not None
        self.assertEqual(stored.metadata.id, "base")

    def test_merge_fills_empty_cells_from_incoming(self) -> None:
        base = GridArchive(5)
        incoming = GridArchive(5)
        incoming.try_insert(_minimal_elite((4, 1), 0.55, elite_id="only-incoming"))
        merge_archives(base, incoming)
        stored = base.get(4, 1)
        assert stored is not None
        self.assertEqual(stored.fitness, 0.55)

    def test_merge_resolution_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            merge_archives(GridArchive(5), GridArchive(8))


if __name__ == "__main__":
    unittest.main()

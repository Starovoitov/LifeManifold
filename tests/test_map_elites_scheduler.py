"""Unit tests for MAP-Elites scheduler YAML and target-bin selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from worldspace.illuminators.archive import (
    ArchiveElite,
    GridArchive,
    new_elite_metadata,
)
from worldspace.illuminators.evaluation import bin_center
from worldspace.illuminators.scheduler import (
    DEFAULT_SCHEDULER_PATH,
    EmitterKind,
    RunCounters,
    SchedulerConfig,
    load_scheduler,
    resolve_emitter_for_slot,
    resolve_emitter_kind,
    select_target_bin,
    slot_emitter_for_candidate,
)
from worldspace.specs.spec import WorldSpec

_SPECS = Path(__file__).resolve().parents[1] / "worldspace" / "specs"
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
    from dataclasses import replace

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


class TestLoadScheduler(unittest.TestCase):
    def test_load_production_scheduler(self) -> None:
        config = load_scheduler(DEFAULT_SCHEDULER_PATH)
        self.assertEqual(config.schema_version, "1.2")
        self.assertEqual(config.batch_size, 50)
        self.assertEqual(len(config.batch_emitters), 50)
        self.assertEqual(config.batch_emitters.count("random"), 20)
        self.assertEqual(config.batch_emitters.count("genetic"), 20)
        self.assertEqual(config.batch_emitters.count("llm"), 10)
        self.assertEqual(config.iterations, 10_000)
        self.assertEqual(config.grid_resolution, 50)
        self.assertEqual(config.initial_random_candidates, 100)
        self.assertTrue(config.llm_enabled)
        self.assertFalse(config.surrogate_enabled)

    def test_iterations_override(self) -> None:
        config = load_scheduler(DEFAULT_SCHEDULER_PATH, iterations_override=42)
        self.assertEqual(config.iterations, 42)
        self.assertEqual(config.batch_size, 50)

    def test_batch_emitters_length_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            doc = yaml.safe_load(DEFAULT_SCHEDULER_PATH.read_text(encoding="utf-8"))
            doc["batch_size"] = 49
            path.write_text(yaml.safe_dump(doc), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_scheduler(path)
            self.assertIn("batch_emitters length", str(ctx.exception))

    def test_invalid_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            doc = yaml.safe_load(DEFAULT_SCHEDULER_PATH.read_text(encoding="utf-8"))
            doc["schema_version"] = "1.1"
            path.write_text(yaml.safe_dump(doc), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_scheduler(path)
            self.assertIn("schema_version", str(ctx.exception))

    def test_invalid_emitter_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            doc = yaml.safe_load(DEFAULT_SCHEDULER_PATH.read_text(encoding="utf-8"))
            emitters = list(doc["batch_emitters"])
            emitters[0] = "hybrid"
            doc["batch_emitters"] = emitters
            path.write_text(yaml.safe_dump(doc), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_scheduler(path)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_scheduler("/nonexistent/map_elites_scheduler.yaml")


class TestSelectTargetBin(unittest.TestCase):
    def test_empty_archive_uniform_among_empty(self) -> None:
        archive = GridArchive(3)
        rng = np.random.default_rng(7)
        bins = {select_target_bin(archive, rng).bin for _ in range(200)}
        self.assertTrue(all(0 <= i < 3 and 0 <= j < 3 for i, j in bins))
        self.assertGreater(len(bins), 1)

    def test_empty_archive_deterministic_with_seed(self) -> None:
        archive = GridArchive(4)
        a = select_target_bin(archive, np.random.default_rng(99))
        b = select_target_bin(archive, np.random.default_rng(99))
        self.assertEqual(a, b)

    def test_partial_fill_skips_interior_of_occupied_block(self) -> None:
        archive = GridArchive(4)
        for i in range(3):
            for j in range(3):
                fitness = 0.1 if (i, j) == (2, 2) else 0.9
                archive.try_insert(_minimal_elite((i, j), fitness, elite_id=f"{i}-{j}"))
        target = select_target_bin(archive, np.random.default_rng(0))
        self.assertEqual(target.bin, (2, 2))
        self.assertNotEqual(target.bin, (1, 1))

    def test_boundary_prefers_min_fitness(self) -> None:
        archive = GridArchive(3)
        archive.try_insert(_minimal_elite((1, 1), 0.9, elite_id="high"))
        archive.try_insert(_minimal_elite((1, 0), 0.2, elite_id="low"))
        target = select_target_bin(archive, np.random.default_rng(0))
        self.assertEqual(target.bin, (1, 0))

    def test_boundary_tie_lexicographic(self) -> None:
        archive = GridArchive(3)
        archive.try_insert(_minimal_elite((0, 1), 0.3, elite_id="a"))
        archive.try_insert(_minimal_elite((2, 1), 0.3, elite_id="b"))
        target = select_target_bin(archive, np.random.default_rng(0))
        self.assertEqual(target.bin, (0, 1))

    def test_full_archive_random_bin(self) -> None:
        archive = GridArchive(2)
        for i in range(2):
            for j in range(2):
                archive.try_insert(
                    _minimal_elite((i, j), float(i + j), elite_id=f"{i}-{j}")
                )
        rng = np.random.default_rng(3)
        seen = {select_target_bin(archive, rng).bin for _ in range(50)}
        self.assertGreater(len(seen), 1)

    def test_bin_centers_match_evaluation_helper(self) -> None:
        archive = GridArchive(5)
        target = select_target_bin(archive, np.random.default_rng(1))
        i, j = target.bin
        expected = bin_center(i, j, 5)
        self.assertAlmostEqual(target.target_stability, expected[0])
        self.assertAlmostEqual(target.target_diversity, expected[1])

    def test_isinstance_scheduler_config_from_load(self) -> None:
        config = load_scheduler(DEFAULT_SCHEDULER_PATH)
        self.assertIsInstance(config, SchedulerConfig)


def _mini_config(*, initial_random_candidates: int = 10) -> SchedulerConfig:
    return SchedulerConfig(
        schema_version="1.2",
        iterations=5,
        batch_size=4,
        grid_resolution=5,
        early_extinction_step=200,
        min_steps=200,
        batch_emitters=("random", "genetic", "genetic", "llm"),
        initial_random_candidates=initial_random_candidates,
        llm_enabled=True,
        surrogate_enabled=False,
        surrogate_stub_mean=0.5,
        surrogate_stub_uncertainty=1.0,
    )


class TestInitialRandomPhase(unittest.TestCase):
    def test_threshold_zero_no_override(self) -> None:
        config = _mini_config(initial_random_candidates=0)
        self.assertEqual(
            resolve_emitter_kind(
                config, slot_emitter="genetic", candidates_evaluated=0
            ),
            "genetic",
        )

    def test_before_threshold_forces_random(self) -> None:
        config = _mini_config(initial_random_candidates=100)
        for slot in ("genetic", "llm"):
            self.assertEqual(
                resolve_emitter_kind(
                    config, slot_emitter=slot, candidates_evaluated=0
                ),
                "random",
            )
            self.assertEqual(
                resolve_emitter_kind(
                    config, slot_emitter=slot, candidates_evaluated=99
                ),
                "random",
            )

    def test_at_threshold_uses_slot_emitter(self) -> None:
        config = _mini_config(initial_random_candidates=100)
        self.assertEqual(
            resolve_emitter_kind(
                config, slot_emitter="genetic", candidates_evaluated=100
            ),
            "genetic",
        )
        self.assertEqual(
            resolve_emitter_kind(
                config, slot_emitter="llm", candidates_evaluated=100
            ),
            "llm",
        )

    def test_resolve_for_slot_by_candidate_id(self) -> None:
        config = _mini_config(initial_random_candidates=10)
        self.assertEqual(
            resolve_emitter_for_slot(config, candidate_id=2, candidates_evaluated=5),
            "random",
        )
        self.assertEqual(
            slot_emitter_for_candidate(config, candidate_id=2),
            "genetic",
        )
        self.assertEqual(
            resolve_emitter_for_slot(config, candidate_id=2, candidates_evaluated=10),
            "genetic",
        )
        self.assertEqual(
            resolve_emitter_for_slot(config, candidate_id=3, candidates_evaluated=10),
            "llm",
        )

    def test_simulated_iterations_cross_threshold_mid_batch(self) -> None:
        config = _mini_config(initial_random_candidates=10)
        counters = RunCounters()
        resolved: list[EmitterKind] = []
        for _iteration in range(3):
            for candidate_id in range(config.batch_size):
                resolved.append(
                    resolve_emitter_for_slot(
                        config,
                        candidate_id=candidate_id,
                        candidates_evaluated=counters.candidates_evaluated,
                    )
                )
                counters.record_evaluation()
        self.assertEqual(len(resolved), 12)
        self.assertEqual(resolved[:10], ["random"] * 10)
        self.assertEqual(resolved[10], "genetic")
        self.assertEqual(resolved[11], "llm")

    def test_run_counters_persist_across_iterations(self) -> None:
        counters = RunCounters()
        for _ in range(8):
            counters.record_evaluation()
        self.assertEqual(counters.candidates_evaluated, 8)
        counters.record_evaluation()
        self.assertEqual(counters.candidates_evaluated, 9)

    def test_invalid_candidate_id_raises(self) -> None:
        config = _mini_config()
        with self.assertRaises(ValueError):
            slot_emitter_for_candidate(config, candidate_id=4)


if __name__ == "__main__":
    unittest.main()

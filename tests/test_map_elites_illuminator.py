"""Tests for MapElitesIlluminator orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldspace.illuminators.archive import (
    ARCHIVE_SCHEMA_VERSION_V1_3,
    append_archive_line,
    elite_from_eval,
    elite_to_archive_record,
    load_and_collapse_jsonl,
    new_elite_metadata,
)
from worldspace.illuminators.evaluation import (
    ILLUMINATOR_MIN_STEPS,
    evaluate_candidate,
)
from worldspace.illuminators.illuminator import (
    MapElitesIlluminator,
    archive_jsonl_path,
    normalize_illuminator_steps,
)
from worldspace.illuminators.archive_factory import (
    archive_factory_config_from_scheduler,
    create_archive,
)
from worldspace.illuminators.cvt import (
    CVT_CENTROIDS_FILENAME,
    centroids_path_for_output,
)
from worldspace.illuminators.scheduler import (
    DEFAULT_MINI_CVT_SCHEDULER_PATH,
    DEFAULT_MINI_SCHEDULER_PATH,
    load_scheduler,
)
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec

_BASE = WorldSpec(
    birth=[1],
    survival=[2, 3],
    noise=0.02,
    resource_regen=0.05,
    predation=0.1,
    cell_types=CANONICAL_CELL_TYPES.copy(),
    grid_size=8,
    steps=200,
    seed=0,
)


def _occupied_fitness(
    archive_path: Path, *, resolution: int
) -> dict[tuple[int, int], float]:
    archive = load_and_collapse_jsonl(archive_path, resolution=resolution)
    snapshot: dict[tuple[int, int], float] = {}
    for i in range(resolution):
        for j in range(resolution):
            elite = archive.get(i, j)
            if elite is not None:
                snapshot[(i, j)] = elite.fitness
    return snapshot


class TestNormalizeIlluminatorSteps(unittest.TestCase):
    def test_steps_below_minimum_are_raised(self) -> None:
        self.assertEqual(normalize_illuminator_steps(50), ILLUMINATOR_MIN_STEPS)


class TestMapElitesIlluminator(unittest.TestCase):
    def test_cold_start_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = MapElitesIlluminator().run(
                scheduler_path=DEFAULT_MINI_SCHEDULER_PATH,
                output_dir=out,
                seed=7,
                grid_size=8,
                steps=200,
                iterations=1,
            )
            path = archive_jsonl_path(out)
            self.assertEqual(result.archive_jsonl_path, path)
            self.assertEqual(result.evaluations, 4)
            self.assertTrue(path.is_file())
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreater(len(lines), 0)
            record = json.loads(lines[0])
            self.assertEqual(record["schema_version"], "1.2")

    def test_load_archive_skips_initial_random_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            archive_path = archive_jsonl_path(out)
            eval_result = evaluate_candidate(
                _BASE, resolution=10, enforce_min_steps=True
            )
            elite = elite_from_eval(
                eval_result,
                new_elite_metadata(
                    generated_by="random",
                    emitter_type="random",
                ),
            )
            append_archive_line(archive_path, elite_to_archive_record(elite))
            result = MapElitesIlluminator().run(
                scheduler_path=DEFAULT_MINI_SCHEDULER_PATH,
                output_dir=out,
                seed=3,
                grid_size=8,
                steps=200,
                iterations=1,
                load_archive_path=archive_path,
            )
            self.assertEqual(result.counters.candidates_evaluated, 104)

    def test_same_seed_reproducible_via_illuminator(self) -> None:
        def snap(seed: int) -> dict[tuple[int, int], float]:
            with tempfile.TemporaryDirectory() as tmp:
                result = MapElitesIlluminator().run(
                    scheduler_path=DEFAULT_MINI_SCHEDULER_PATH,
                    output_dir=Path(tmp),
                    seed=seed,
                    grid_size=8,
                    steps=200,
                    iterations=2,
                )
                return _occupied_fitness(result.archive_jsonl_path, resolution=10)

        self.assertEqual(snap(42), snap(42))
        self.assertNotEqual(snap(1), snap(2))

    def test_mini_cvt_cold_start_writes_jsonl_1_3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = MapElitesIlluminator().run(
                scheduler_path=DEFAULT_MINI_CVT_SCHEDULER_PATH,
                output_dir=out,
                seed=7,
                grid_size=8,
                steps=200,
                iterations=1,
            )
            path = archive_jsonl_path(out)
            self.assertEqual(result.evaluations, 4)
            self.assertGreater(result.filled_cells, 0)
            self.assertLessEqual(result.filled_cells, 25)
            centroids_path = centroids_path_for_output(out)
            self.assertTrue(centroids_path.is_file())
            self.assertEqual(centroids_path.name, CVT_CENTROIDS_FILENAME)
            record = json.loads(
                path.read_text(encoding="utf-8").strip().splitlines()[0]
            )
            self.assertEqual(record["schema_version"], ARCHIVE_SCHEMA_VERSION_V1_3)
            self.assertEqual(record["archive_type"], "cvt")

    def test_mini_cvt_resume_reuses_centroids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            MapElitesIlluminator().run(
                scheduler_path=DEFAULT_MINI_CVT_SCHEDULER_PATH,
                output_dir=out,
                seed=3,
                grid_size=8,
                steps=200,
                iterations=1,
            )
            centroids_path = centroids_path_for_output(out)
            centroids_before = centroids_path.read_bytes()
            archive_path = archive_jsonl_path(out)
            config = load_scheduler(DEFAULT_MINI_CVT_SCHEDULER_PATH)
            collapsed = load_and_collapse_jsonl(
                archive_path,
                archive_type="cvt",
                centroids_path=centroids_path,
            )
            self.assertGreater(collapsed.filled_count(), 0)
            reloaded = create_archive(
                archive_factory_config_from_scheduler(config),
                output_dir=out,
            )
            self.assertEqual(centroids_before, centroids_path.read_bytes())
            self.assertEqual(reloaded.n_cells, config.n_centroids)


if __name__ == "__main__":
    unittest.main()

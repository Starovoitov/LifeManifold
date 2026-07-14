"""Tests for pyribs baseline runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldspace.illuminators.pyribs_baseline import (
    PyribsBaselineConfig,
    build_scheduler,
    export_archive_jsonl,
    genome_bounds,
    load_baseline_into_archives,
    run_pyribs_baseline,
)


class TestPyribsBaselineUnit(unittest.TestCase):
    def test_genome_bounds_length(self) -> None:
        bounds = genome_bounds()
        self.assertEqual(len(bounds), 21)
        self.assertEqual(bounds[0], (0.0, 1.0))
        self.assertEqual(bounds[18], (0.0, 0.2))

    def test_build_scheduler_cma_me_and_mae(self) -> None:
        for algo in ("cma_me", "cma_mae"):
            with self.subTest(algo=algo):
                cfg = PyribsBaselineConfig(algo=algo, seed=0, evaluations=50)
                scheduler, archive, result_archive = build_scheduler(cfg)
                sols = scheduler.ask()
                self.assertEqual(sols.shape[0], 5 * 50)
                if algo == "cma_me":
                    self.assertIsNone(result_archive)
                else:
                    self.assertIsNotNone(result_archive)

    def test_evaluations_must_divide_ask_size(self) -> None:
        cfg = PyribsBaselineConfig(
            algo="cma_me",
            seed=0,
            evaluations=100,
            num_emitters=5,
            emitter_batch_size=50,
            load_archive=None,
            parallel_eval=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_pyribs_baseline(cfg, output_dir=Path(tmp))


class TestPyribsBaselineSmoke(unittest.TestCase):
    def test_cma_me_short_run_writes_summary(self) -> None:
        cfg = PyribsBaselineConfig(
            algo="cma_me",
            seed=0,
            evaluations=20,
            num_emitters=1,
            emitter_batch_size=10,
            grid_size=16,
            steps=200,
            load_archive=None,
            parallel_eval=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = run_pyribs_baseline(cfg, output_dir=out)
            self.assertEqual(result.evaluations, 20)
            self.assertEqual(result.warm_start_elites, 0)
            summary = json.loads((out / "nightly_run_summary.json").read_text())
            self.assertEqual(summary["evaluations"], 20)
            self.assertIn("qd_score", summary)
            self.assertEqual(summary["llm_enabled"], False)
            self.assertTrue((out / "map_elites_archive.jsonl").is_file())
            self.assertGreater(result.filled_cells, 0)
            trace = json.loads(
                (out / "archive_trace.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[1]
            )
            self.assertEqual(trace["evaluations"], 10)
            self.assertIn("coverage", trace)
            self.assertIn("qd_score", trace)

    def test_cma_mae_short_run_seed_reproducible_coverage(self) -> None:
        cfg = PyribsBaselineConfig(
            algo="cma_mae",
            seed=1,
            evaluations=20,
            num_emitters=1,
            emitter_batch_size=10,
            grid_size=16,
            steps=200,
            load_archive=None,
            parallel_eval=False,
        )
        with (
            tempfile.TemporaryDirectory() as tmp1,
            tempfile.TemporaryDirectory() as tmp2,
        ):
            r1 = run_pyribs_baseline(cfg, output_dir=Path(tmp1))
            r2 = run_pyribs_baseline(cfg, output_dir=Path(tmp2))
            self.assertEqual(r1.evaluations, 20)
            self.assertEqual(r2.evaluations, 20)
            self.assertEqual(r1.filled_cells, r2.filled_cells)
            self.assertAlmostEqual(r1.coverage, r2.coverage)
            self.assertEqual(r1.mean_best_fitness, r2.mean_best_fitness)

    def test_warm_start_load_counts(self) -> None:
        baseline = Path(
            "artifacts/map_elites_nightly/baseline/map_elites_archive.jsonl"
        )
        cfg = PyribsBaselineConfig(algo="cma_me", seed=0)
        _, archive, result_archive = build_scheduler(cfg)
        n = load_baseline_into_archives(
            baseline,
            archive=archive,
            result_archive=result_archive,
            grid_size=50,
            steps=200,
        )
        self.assertGreater(n, 100)
        # JSONL may contain multiple lines per niche; archive keeps one elite/cell.
        self.assertGreater(int(archive.stats.num_elites), 100)
        self.assertLessEqual(int(archive.stats.num_elites), n)
        path = Path(tempfile.mkdtemp()) / "out.jsonl"
        exported = export_archive_jsonl(archive, path)
        self.assertEqual(exported, int(archive.stats.num_elites))


if __name__ == "__main__":
    unittest.main()

"""Tests for the CA-independent pyribs standard benchmark runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_pyribs_standard import parse_args
from worldspace.benchmarks.pyribs_standard_runner import (
    PyribsStandardConfig,
    build_scheduler,
    run_pyribs_standard,
    standard_hyperparams,
)


class TestPyribsStandardUnit(unittest.TestCase):
    def test_locked_hyperparameters(self) -> None:
        sphere = PyribsStandardConfig(
            benchmark="sphere",
            algo="cma_me",
            seed=0,
        )
        hp = standard_hyperparams(sphere)
        self.assertEqual(hp["solution_dim"], 20)
        self.assertEqual(hp["archive_dims"], [100, 100])
        self.assertEqual(hp["archive_ranges"], [[-51.2, 51.2], [-51.2, 51.2]])
        self.assertEqual(hp["ask_size"], 250)
        self.assertEqual(hp["asks"], 130)
        self.assertEqual(hp["x0"], [2.048] * 20)
        self.assertEqual(hp["warm_start_archive"], None)

        random_hp = standard_hyperparams(
            PyribsStandardConfig(
                benchmark="sphere",
                algo="me_random",
                seed=0,
            )
        )
        self.assertEqual(random_hp["num_emitters"], 1)
        self.assertEqual(random_hp["emitter_batch_size"], 250)
        self.assertEqual(random_hp["sigma"], 0.5)

    def test_builds_each_valid_scheduler(self) -> None:
        for benchmark, algo in (
            ("sphere", "cma_me"),
            ("sphere", "cma_mae"),
            ("sphere", "me_random"),
            ("rastrigin", "cma_me"),
            ("rastrigin", "cma_mae"),
        ):
            with self.subTest(benchmark=benchmark, algo=algo):
                config = PyribsStandardConfig(
                    benchmark=benchmark,  # type: ignore[arg-type]
                    algo=algo,  # type: ignore[arg-type]
                    seed=0,
                    evaluations=250,
                )
                scheduler, _, result_archive = build_scheduler(config)
                self.assertEqual(scheduler.ask().shape, (250, 20))
                self.assertEqual(result_archive is not None, algo == "cma_mae")

    def test_invalid_pair_and_budget_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PyribsStandardConfig(
                benchmark="rastrigin",
                algo="me_random",
                seed=0,
            ).validate()
        with self.assertRaises(ValueError):
            PyribsStandardConfig(
                benchmark="sphere",
                algo="cma_me",
                seed=0,
                evaluations=251,
            ).validate()
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--benchmark",
                    "rastrigin",
                    "--algo",
                    "me_random",
                    "--seed",
                    "0",
                    "--output-dir",
                    "unused",
                ]
            )


class TestPyribsStandardSmoke(unittest.TestCase):
    @staticmethod
    def _short_config(algo: str = "cma_me") -> PyribsStandardConfig:
        return PyribsStandardConfig(
            benchmark="sphere",
            algo=algo,  # type: ignore[arg-type]
            seed=7,
            evaluations=20,
            num_emitters=1,
            emitter_batch_size=10,
            archive_dims=(10, 10),
        )

    def test_run_writes_summary_trace_and_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = run_pyribs_standard(
                self._short_config(),
                output_dir=out,
            )
            self.assertEqual(result.evaluations, 20)
            self.assertGreater(result.filled_cells, 0)
            summary = json.loads(
                (out / "nightly_run_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["benchmark"], "sphere")
            self.assertEqual(summary["condition"], "cma_me")
            self.assertTrue(summary["standard_benchmark"])
            self.assertEqual(summary["pyribs_warm_start_elites"], 0)
            self.assertIn("mean_best_fitness", summary)
            self.assertIn("qd_score", summary)
            traces = [
                json.loads(line)
                for line in (out / "archive_trace.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([row["evaluations"] for row in traces], [0, 10, 20])
            self.assertTrue((out / "pyribs_archive.npz").is_file())
            with np.load(out / "pyribs_archive.npz") as archive:
                self.assertIn("objective", archive.files)
                self.assertIn("measures", archive.files)

    def test_same_seed_is_reproducible(self) -> None:
        config = self._short_config("cma_mae")
        with (
            tempfile.TemporaryDirectory() as tmp1,
            tempfile.TemporaryDirectory() as tmp2,
        ):
            first = run_pyribs_standard(config, output_dir=Path(tmp1))
            second = run_pyribs_standard(config, output_dir=Path(tmp2))
            self.assertEqual(first.filled_cells, second.filled_cells)
            self.assertEqual(first.coverage, second.coverage)
            self.assertEqual(first.mean_best_fitness, second.mean_best_fitness)
            self.assertEqual(first.qd_score, second.qd_score)


if __name__ == "__main__":
    unittest.main()

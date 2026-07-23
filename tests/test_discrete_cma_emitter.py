"""Tests for native discrete CMA emitter and pyribs integration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from worldspace.illuminators.discrete_cma_emitter import (
    discrete_x0,
)
from worldspace.illuminators.emitters.genetics import GENOME_SIZE
from worldspace.illuminators.pyribs_baseline import (
    PyribsBaselineConfig,
    build_scheduler,
    pyribs_hyperparams,
    run_pyribs_baseline,
)


class TestDiscreteCMAEmitterUnit(unittest.TestCase):
    def test_discrete_x0_is_binary_rule_bits(self) -> None:
        x0 = discrete_x0()
        self.assertEqual(x0.shape[0], GENOME_SIZE)
        self.assertTrue(np.all(np.isin(x0[:18], (0.0, 1.0))))

    def test_ask_returns_binary_rule_bits(self) -> None:
        cfg = PyribsBaselineConfig(
            algo="cma_me",
            seed=0,
            emitter_kind="discrete_cma",
        )
        scheduler, archive, _ = build_scheduler(cfg)
        solutions = scheduler.ask()
        self.assertEqual(solutions.shape[1], GENOME_SIZE)
        self.assertTrue(np.all(np.isin(solutions[:, :18], (0.0, 1.0))))

    def test_hyperparams_record_discrete_emitter(self) -> None:
        hp = pyribs_hyperparams(
            PyribsBaselineConfig(
                algo="cma_me",
                seed=0,
                emitter_kind="discrete_cma",
                decode_mode="threshold",
            )
        )
        self.assertEqual(hp["emitter_kind"], "discrete_cma")
        self.assertEqual(hp["decode_mode"], "threshold")

    def test_discrete_cma_mae_rejected(self) -> None:
        cfg = PyribsBaselineConfig(
            algo="cma_mae",
            seed=0,
            emitter_kind="discrete_cma",
        )
        with self.assertRaises(ValueError):
            build_scheduler(cfg)


class TestDiscreteCMAEmitterSmoke(unittest.TestCase):
    def test_discrete_cma_me_short_run_writes_summary(self) -> None:
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
            emitter_kind="discrete_cma",
            decode_mode="threshold",
            condition_label="cma_me_discrete",
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = run_pyribs_baseline(cfg, output_dir=out)
            self.assertEqual(result.evaluations, 20)
            summary = json.loads((out / "nightly_run_summary.json").read_text())
            self.assertEqual(summary["condition"], "cma_me_discrete")
            self.assertEqual(
                summary["pyribs_hyperparams"]["emitter_kind"], "discrete_cma"
            )
            self.assertGreater(result.filled_cells, 0)


if __name__ == "__main__":
    unittest.main()

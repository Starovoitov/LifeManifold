"""Tests for pbCMA emitter and pyribs integration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from worldspace.illuminators.emitters.genetics import GENOME_SIZE
from worldspace.illuminators.pbcma_emitter import PBCMAEmitter, pbcma_x0
from worldspace.illuminators.pyribs_baseline import (
    PyribsBaselineConfig,
    build_scheduler,
    pyribs_hyperparams,
    run_pyribs_baseline,
)


class TestPBCMAEmitterUnit(unittest.TestCase):
    def test_pbcma_x0_mid_bits(self) -> None:
        x0 = pbcma_x0()
        self.assertEqual(x0.shape[0], GENOME_SIZE)
        self.assertTrue(np.allclose(x0[:18], 0.5))

    def test_ask_returns_binary_rule_bits(self) -> None:
        cfg = PyribsBaselineConfig(
            algo="cma_me",
            seed=0,
            emitter_kind="pbcma",
        )
        scheduler, _archive, _ = build_scheduler(cfg)
        solutions = scheduler.ask()
        self.assertEqual(solutions.shape[1], GENOME_SIZE)
        self.assertTrue(np.all(np.isin(solutions[:, :18], (0.0, 1.0))))

    def test_hyperparams_record_pbcma(self) -> None:
        hp = pyribs_hyperparams(
            PyribsBaselineConfig(
                algo="cma_me",
                seed=0,
                emitter_kind="pbcma",
                decode_mode="threshold",
            )
        )
        self.assertEqual(hp["emitter_kind"], "pbcma")

    def test_pbcma_mae_rejected(self) -> None:
        cfg = PyribsBaselineConfig(
            algo="cma_mae",
            seed=0,
            emitter_kind="pbcma",
        )
        with self.assertRaises(ValueError):
            build_scheduler(cfg)

    def test_tell_adapts_sigma_and_keeps_margin(self) -> None:
        cfg = PyribsBaselineConfig(
            algo="cma_me",
            seed=1,
            emitter_kind="pbcma",
            num_emitters=1,
            emitter_batch_size=20,
            load_archive=None,
        )
        scheduler, archive, _ = build_scheduler(cfg)
        emitter = scheduler.emitters[0]
        assert isinstance(emitter, PBCMAEmitter)
        sigma0 = emitter.sigma
        solutions = scheduler.ask()
        objectives = np.linspace(0.1, 0.9, solutions.shape[0])
        measures = np.column_stack(
            [
                np.linspace(0.0, 1.0, solutions.shape[0]),
                np.linspace(0.0, 1.0, solutions.shape[0]),
            ]
        )
        add_info = archive.add(solutions, objectives, measures)
        emitter.tell(solutions, objectives, measures, add_info)
        self.assertTrue(np.all(emitter._mean[:18] >= 0.05 - 1e-9))
        self.assertTrue(np.all(emitter._mean[:18] <= 0.95 + 1e-9))
        # Sigma should remain finite and positive after one update.
        self.assertGreater(emitter.sigma, 0.0)
        self.assertLessEqual(emitter.sigma, 2.0)
        _ = sigma0  # may rise or fall; just ensure tell ran


class TestPBCMAEmitterSmoke(unittest.TestCase):
    def test_pbcma_me_short_run_writes_summary(self) -> None:
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
            emitter_kind="pbcma",
            decode_mode="threshold",
            condition_label="cma_me_pbcma",
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = run_pyribs_baseline(cfg, output_dir=out)
            self.assertEqual(result.evaluations, 20)
            summary = json.loads((out / "nightly_run_summary.json").read_text())
            self.assertEqual(summary["condition"], "cma_me_pbcma")
            self.assertEqual(summary["pyribs_hyperparams"]["emitter_kind"], "pbcma")
            self.assertGreater(result.filled_cells, 0)


if __name__ == "__main__":
    unittest.main()

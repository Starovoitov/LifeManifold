"""Tests for surrogate buffer analysis helpers and CLI script."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import unittest.mock
from importlib.util import find_spec
from pathlib import Path

from scripts.analyze_surrogate_buffer import main as analyze_main
from worldspace.surrogate.buffer_analysis import (
    analyze_buffer_path,
    format_analysis_report,
    scan_buffer_metadata,
)
from worldspace.surrogate.model import TARGET_KEYS
from worldspace.surrogate.synthetic_buffer import write_synthetic_buffer
from worldspace.surrogate.training_runtime import train_from_buffer


class TestAnalyzeSurrogateBuffer(unittest.TestCase):
    def test_scan_buffer_metadata_counts_emitter_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            write_synthetic_buffer(buffer_path, n_samples=12, seed=3)
            metadata = scan_buffer_metadata(buffer_path)
            self.assertEqual(metadata.emitter_types, {"synthetic": 12})
            self.assertEqual(metadata.metadata_sources, {"unknown": 12})

    def test_analyze_buffer_path_reports_distribution_and_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            write_synthetic_buffer(buffer_path, n_samples=200, seed=5)
            report = analyze_buffer_path(buffer_path, fit_model=False)
            self.assertEqual(report["sample_count"], 200)
            self.assertIn("stability", report["targets"])
            self.assertEqual(len(report["stability_histogram"]), 5)
            self.assertIn("train_mean_mae", report["stability_baselines"])
            self.assertGreater(report["stability_baselines"]["train_mean_mae"], 0.0)
            self.assertIn("feature_correlations_stability", report)
            split = report["holdout_split"]
            for key in TARGET_KEYS:
                self.assertIn(key, split["train_mean"])
                self.assertIn(key, split["holdout_mean"])
                self.assertIn(key, split["train_std"])
                self.assertIn(key, split["holdout_std"])
            text = format_analysis_report(report)
            self.assertIn("Target stability:", text)
            self.assertIn("Quality gate reference:", text)

    def test_format_analysis_report_ignores_boolean_fitness_compose_ab(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            write_synthetic_buffer(buffer_path, n_samples=200, seed=5)
            report = analyze_buffer_path(buffer_path, fit_model=False)
            report["fitness_compose_ab"] = True
            text = format_analysis_report(report)
            self.assertNotIn("Fitness compose A/B", text)

    def test_compare_models_primary_fail_fitness_compose_ab(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / "buffer.jsonl"
            write_synthetic_buffer(buffer_path, n_samples=200, seed=5)

            def fail_lightgbm(model_type, *_args, **_kwargs):
                if model_type == "lightgbm":
                    raise RuntimeError("simulated primary failure")
                return {
                    "model_type": model_type,
                    "model_holdout": {
                        "r2_fitness": 0.1,
                        "mae_fitness": 0.1,
                        "mae_stability": 0.1,
                    },
                    "per_target_holdout": [],
                    "stability_mae_bands": [],
                    "quality_gate": {
                        "model_mae_stability": 0.1,
                        "model_mae_stability_gap_to_gate": 0.04,
                    },
                    "fitness_compose_ab": {
                        "hard": {"r2_fitness": 0.1, "mae_fitness": 0.1},
                        "soft": {"r2_fitness": 0.2, "mae_fitness": 0.09},
                    },
                }

            with unittest.mock.patch(
                "worldspace.surrogate.buffer_analysis._fit_holdout_model",
                side_effect=fail_lightgbm,
            ):
                report = analyze_buffer_path(
                    buffer_path,
                    fit_model=True,
                    compare_models=True,
                    fitness_compose_ab=True,
                    model_type="lightgbm",
                )
            self.assertTrue(report["fitness_compose_ab_requested"])
            self.assertNotIn("fitness_compose_ab", report)
            text = format_analysis_report(report)
            self.assertNotIn("Fitness compose A/B", text)

    def test_analyze_main_writes_json_for_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            buffer_path = root / "buffer.jsonl"
            output_path = root / "analysis.json"
            write_synthetic_buffer(buffer_path, n_samples=80, seed=9)
            argv = [
                "analyze_surrogate_buffer.py",
                "--buffer",
                str(buffer_path),
                "--output-json",
                str(output_path),
                "--quiet",
            ]
            with unittest.mock.patch("sys.argv", argv):
                with unittest.mock.patch("sys.stdout", io.StringIO()) as stdout:
                    exit_code = analyze_main()
            self.assertEqual(exit_code, 0)
            self.assertEqual(stdout.getvalue(), "")
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["mode"], "buffer")
            self.assertEqual(payload["report"]["sample_count"], 80)

    def test_analyze_cli_fit_model_matches_train_holdout(self) -> None:
        if find_spec("lightgbm") is None:
            self.skipTest("lightgbm not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            buffer_path = root / "buffer.jsonl"
            output_path = root / "analysis.json"
            checkpoint_path = root / "model.pkl"
            summary_path = root / "model.summary.json"
            write_synthetic_buffer(buffer_path, n_samples=240, seed=42)
            train_result = train_from_buffer(
                buffer_path=buffer_path,
                checkpoint_path=checkpoint_path,
                summary_path=summary_path,
                model_type="lightgbm",
                micro=True,
                require_quality_gate=False,
                consistency_weight=0.0,
            )
            self.assertTrue(train_result.success, msg=train_result.error_message)
            argv = [
                "analyze_surrogate_buffer.py",
                "--buffer",
                str(buffer_path),
                "--output-json",
                str(output_path),
                "--fit-model",
                "--ensemble-size",
                "8",
                "--random-state",
                "42",
                "--test-fraction",
                "0.2",
                "--consistency-weight",
                "0",
                "--quiet",
            ]
            with unittest.mock.patch("sys.argv", argv):
                with unittest.mock.patch("sys.stdout", io.StringIO()):
                    exit_code = analyze_main()
            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            analyze_metrics = payload["report"]["model_holdout"]
            for key in ("r2_fitness", "mae_fitness", "mae_stability"):
                self.assertAlmostEqual(
                    analyze_metrics[key],
                    train_result.holdout_metrics[key],
                    places=3,
                    msg=f"cli mismatch on {key}",
                )


if __name__ == "__main__":
    unittest.main()

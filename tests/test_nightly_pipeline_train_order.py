"""Unit tests for nightly pipeline step order (no full illuminator run)."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from worldspace.illuminators.nightly_report import NightlyRunReport
from worldspace.scripts import run_map_elites_nightly as nightly_mod


def _fake_report(*, surrogate_enabled: bool, archive_path: str) -> NightlyRunReport:
    return NightlyRunReport(
        schema_version="1.0",
        scheduler_path="scheduler.yaml",
        seed=0,
        iterations=2,
        evaluations=10,
        filled_cells=5,
        grid_resolution=50,
        coverage=0.01,
        jsonl_raw_lines=10,
        jsonl_collapsed_cells=5,
        elapsed_seconds=1.0,
        llm_enabled=False,
        surrogate_enabled=surrogate_enabled,
        archive_jsonl_path=archive_path,
    )


class TestNightlyPipelineTrainOrder(unittest.TestCase):
    def test_run_nightly_pipeline_calls_backfill_before_surrogate_then_train(
        self,
    ) -> None:
        calls: list[str] = []
        baseline_archive = "/tmp/baseline/map_elites_archive.jsonl"
        surrogate_archive = "/tmp/surrogate/map_elites_archive.jsonl"

        def _run_map_elites_nightly(**kwargs):
            if kwargs.get("load_archive_path"):
                calls.append("surrogate_run")
                return _fake_report(
                    surrogate_enabled=True,
                    archive_path=surrogate_archive,
                )
            calls.append("baseline")
            return _fake_report(
                surrogate_enabled=False,
                archive_path=baseline_archive,
            )

        def _ensure_backfill(archive: Path, *, buffer_path=None, overwrite=False):
            calls.append("backfill")
            self.assertEqual(str(archive), baseline_archive)
            return {"buffer_rows_written": 3}

        def _train_nightly_surrogate(**kwargs):
            calls.append("train")
            return Path("/tmp/nightly_v2.summary.json")

        with (
            patch.object(
                nightly_mod,
                "run_map_elites_nightly",
                side_effect=_run_map_elites_nightly,
            ),
            patch.object(
                nightly_mod,
                "ensure_nightly_buffer_backfill",
                side_effect=_ensure_backfill,
            ),
            patch.object(
                nightly_mod,
                "train_nightly_surrogate",
                side_effect=_train_nightly_surrogate,
            ),
            patch.object(nightly_mod, "_write_pipeline_summary"),
        ):
            nightly_mod.run_nightly_pipeline(output_dir="/tmp/nightly_out")

        self.assertEqual(
            calls,
            ["baseline", "backfill", "surrogate_run", "train"],
        )

    def test_run_nightly_pipeline_skip_training_omits_train_step(self) -> None:
        calls: list[str] = []

        def _run_map_elites_nightly(**kwargs):
            if kwargs.get("load_archive_path"):
                calls.append("surrogate_run")
                return _fake_report(
                    surrogate_enabled=True,
                    archive_path="/tmp/surrogate/map_elites_archive.jsonl",
                )
            calls.append("baseline")
            return _fake_report(
                surrogate_enabled=False,
                archive_path="/tmp/baseline/map_elites_archive.jsonl",
            )

        def _ensure_backfill(*args, **kwargs):
            calls.append("backfill")
            return {"buffer_rows_written": 1}

        with (
            patch.object(
                nightly_mod,
                "run_map_elites_nightly",
                side_effect=_run_map_elites_nightly,
            ),
            patch.object(
                nightly_mod,
                "ensure_nightly_buffer_backfill",
                side_effect=_ensure_backfill,
            ),
            patch.object(nightly_mod, "train_nightly_surrogate") as train_mock,
            patch.object(nightly_mod, "_write_pipeline_summary"),
        ):
            nightly_mod.run_nightly_pipeline(
                output_dir="/tmp/nightly_out", skip_training=True
            )

        self.assertEqual(calls, ["baseline", "backfill", "surrogate_run"])
        train_mock.assert_not_called()

    def test_ensure_nightly_buffer_backfill_skips_when_live_and_backfill_present(
        self,
    ) -> None:
        with patch.object(
            nightly_mod,
            "_NIGHTLY_BUFFER_PATH",
            Path("/tmp/buffer_nightly.jsonl"),
        ):
            with (
                patch(
                    "worldspace.surrogate.backfill.buffer_has_live_eval_rows",
                    return_value=True,
                ),
                patch(
                    "worldspace.surrogate.backfill.buffer_has_archive_backfill_rows",
                    return_value=True,
                ),
                patch(
                    "worldspace.surrogate.backfill.backfill_buffer_from_archive",
                ) as backfill_mock,
            ):
                result = nightly_mod.ensure_nightly_buffer_backfill(
                    Path("/tmp/baseline/map_elites_archive.jsonl")
                )
        self.assertIsNone(result)
        backfill_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()

"""Tests for MAP-Elites CLI flags."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch

from worldspace.cli import main as cli_main
from worldspace.cli_mapelites import add_mapelites_arguments, run_mapelites_cli
from worldspace.illuminators.evaluation import ILLUMINATOR_MIN_STEPS
from worldspace.illuminators.illuminator import MapElitesRunResult
from worldspace.illuminators.scheduler import DEFAULT_MINI_SCHEDULER_PATH, RunCounters


class TestIlluminatorsModuleCli(unittest.TestCase):
    def test_module_help_lists_flags(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "worldspace.illuminators", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--scheduler", proc.stdout)
        self.assertIn("--output-dir", proc.stdout)


class TestCliMapelitesHelp(unittest.TestCase):
    def test_help_lists_mapelites_flags(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "worldspace",
                "--illuminator",
                "mapelites",
                "--help",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--grid-resolution", proc.stdout)
        self.assertIn("--load-archive", proc.stdout)
        self.assertIn("--scheduler", proc.stdout)


class TestCliMapelitesValidation(unittest.TestCase):
    def test_steps_below_minimum_exit(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()
        add_mapelites_arguments(parser)
        parser.add_argument("--steps", type=int, default=100)
        parser.add_argument("--grid", type=int, default=50)
        args = parser.parse_args(["--illuminator", "mapelites", "--steps", "100"])
        with self.assertRaises(SystemExit):
            run_mapelites_cli(args)

    @patch("worldspace.illuminators.cli.MapElitesIlluminator")
    def test_invokes_illuminator(self, mock_cls: mock.MagicMock) -> None:
        import argparse

        mock_cls.return_value.run.return_value = MapElitesRunResult(
            iterations=1,
            evaluations=4,
            filled_cells=2,
            archive_jsonl_path=Path("out/map_elites_archive.jsonl"),
            counters=RunCounters(candidates_evaluated=4),
        )
        parser = argparse.ArgumentParser()
        add_mapelites_arguments(parser)
        parser.add_argument("--steps", type=int, default=ILLUMINATOR_MIN_STEPS)
        parser.add_argument("--grid", type=int, default=50)
        args = parser.parse_args(
            [
                "--illuminator",
                "mapelites",
                "--scheduler",
                str(DEFAULT_MINI_SCHEDULER_PATH),
                "--iterations",
                "1",
            ]
        )
        run_mapelites_cli(args)
        mock_cls.return_value.run.assert_called_once()


class TestLegacyCliUnchanged(unittest.TestCase):
    @patch("worldspace.cli.stream_world_space_to_jsonl")
    def test_generator_random_still_runs(
        self, mock_stream: unittest.mock.MagicMock
    ) -> None:
        with patch.object(
            sys, "argv", ["worldspace", "--generator", "random", "--worlds", "1"]
        ):
            cli_main()
        mock_stream.assert_called_once()


if __name__ == "__main__":
    unittest.main()

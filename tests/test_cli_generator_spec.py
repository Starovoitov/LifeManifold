"""Tests for ``--generator-spec`` YAML shape validation (Pydantic)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SPEC_MOD = _REPO / "worldspace" / "cli_generator_spec.py"
_loader = importlib.util.spec_from_file_location(
    "_ws_cli_generator_spec_isolated", _SPEC_MOD
)
assert _loader and _loader.loader
_v = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(_v)
parse_generator_spec_path = _v.parse_generator_spec_path
validate_generator_spec_yaml = _v.validate_generator_spec_yaml

_SPECS = _REPO / "worldspace" / "specs"


class TestCliGeneratorSpec(unittest.TestCase):
    def test_parse_empty_returns_none(self) -> None:
        self.assertIsNone(parse_generator_spec_path(""))
        self.assertIsNone(parse_generator_spec_path("   "))

    def test_parse_non_empty_path(self) -> None:
        p = parse_generator_spec_path(" /tmp/a.yaml ")
        self.assertEqual(p, Path("/tmp/a.yaml"))

    def test_genetic_accepts_genetic_yaml(self) -> None:
        validate_generator_spec_yaml("genetic", _SPECS / "genetic_world_generator.yaml")

    def test_genetic_rejects_llm_yaml(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_generator_spec_yaml("genetic", _SPECS / "llm_world_generator.yaml")
        self.assertIn("genetic", str(ctx.exception).lower())

    def test_llm_accepts_llm_yaml(self) -> None:
        validate_generator_spec_yaml("llm", _SPECS / "llm_world_generator.yaml")

    def test_llm_rejects_hybrid_yaml(self) -> None:
        with self.assertRaises(ValueError):
            validate_generator_spec_yaml("llm", _SPECS / "hybrid_world_generator.yaml")

    def test_hybrid_accepts_hybrid_yaml(self) -> None:
        validate_generator_spec_yaml("hybrid", _SPECS / "hybrid_world_generator.yaml")

    def test_hybrid_rejects_llm_only_yaml(self) -> None:
        with self.assertRaises(ValueError):
            validate_generator_spec_yaml("hybrid", _SPECS / "llm_world_generator.yaml")

    def test_neural_accepts_neural_yaml(self) -> None:
        validate_generator_spec_yaml("neural", _SPECS / "neural_world_generator.yaml")

    def test_neural_rejects_genetic_yaml(self) -> None:
        with self.assertRaises(ValueError):
            validate_generator_spec_yaml(
                "neural", _SPECS / "genetic_world_generator.yaml"
            )

    def test_cli_rejects_spec_with_non_yaml_generator(self) -> None:
        try:
            import worldspace.cli  # noqa: F401
        except ImportError:
            raise unittest.SkipTest(
                "worldspace CLI requires full optional dependencies"
            )
        spec = str(_SPECS / "genetic_world_generator.yaml")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "worldspace",
                "--generator",
                "random",
                "--generator-spec",
                spec,
                "--worlds",
                "0",
            ],
            capture_output=True,
            text=True,
            cwd=str(_REPO),
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("generator-spec", proc.stderr)


if __name__ == "__main__":
    unittest.main()

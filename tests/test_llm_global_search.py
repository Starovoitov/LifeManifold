"""Tests for LLM global_search mode and factory."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from worldspace.generators import (
    HybridGALlmWorldGenerator,
    LLMGlobalSearchWorldGenerator,
    LLMWorldGenerator,
    make_llm_world_generator,
)
from worldspace.generators.llm_config import LLMGeneratorConfig, load_llm_config

_SPECS = Path(__file__).resolve().parent.parent / "worldspace" / "specs"
_PATCH_JSON = json.dumps(
    {
        "birth": [3, 4],
        "survival": [2, 3, 4],
        "noise": 0.07,
        "resource_regen": 0.12,
        "predation": 0.31,
        "reasoning": "ok",
    }
)


def _write_temp_llm_spec(**llm_overrides: object) -> Path:
    base = yaml.safe_load(
        (_SPECS / "llm_world_generator.yaml").read_text(encoding="utf-8")
    )
    llm = dict(base["llm"])
    llm.update(llm_overrides)
    fd, path = tempfile.mkstemp(suffix=".yaml")
    import os

    os.close(fd)
    p = Path(path)
    p.write_text(yaml.safe_dump({"version": 1, "llm": llm}), encoding="utf-8")
    return p


class TestLLMGlobalSearch(unittest.TestCase):
    def test_factory_local_when_global_search_false(self) -> None:
        spec = _write_temp_llm_spec(global_search=False)
        try:
            gen = make_llm_world_generator(grid_size=8, steps=8, seed=0, spec_path=spec)
            self.assertIsInstance(gen, LLMWorldGenerator)
            self.assertNotIsInstance(gen, LLMGlobalSearchWorldGenerator)
        finally:
            spec.unlink()

    def test_factory_global_when_global_search_true(self) -> None:
        spec = _write_temp_llm_spec(global_search=True)
        try:
            gen = make_llm_world_generator(grid_size=8, steps=8, seed=0, spec_path=spec)
            self.assertIsInstance(gen, LLMGlobalSearchWorldGenerator)
        finally:
            spec.unlink()

    def test_local_generator_rejects_global_spec(self) -> None:
        spec = _write_temp_llm_spec(global_search=True)
        try:
            cfg = load_llm_config(spec)
            with self.assertRaises(ValueError):
                LLMWorldGenerator(grid_size=8, steps=8, seed=0, config=cfg)
        finally:
            spec.unlink()

    def test_global_iter_calls_vision_then_patch(self) -> None:
        spec = _write_temp_llm_spec(global_search=True)
        try:
            cfg = load_llm_config(spec)
            with (
                patch(
                    "worldspace.generators.call_llm_vision",
                    return_value="stationary blobs with thin boundaries",
                ) as m_vis,
                patch(
                    "worldspace.generators.call_llm", return_value=_PATCH_JSON
                ) as m_txt,
            ):
                gen = LLMGlobalSearchWorldGenerator(
                    grid_size=8, steps=8, seed=1, config=cfg
                )
                worlds = gen.generate(3)
            self.assertEqual(len(worlds), 3)
            self.assertEqual(m_vis.call_count, 2)
            self.assertEqual(m_txt.call_count, 2)
            self.assertEqual(worlds[1].birth, [3, 4])
        finally:
            spec.unlink()

    def test_hybrid_global_search_uses_vision_on_llm_mutations(self) -> None:
        hybrid_base = yaml.safe_load(
            (_SPECS / "hybrid_world_generator.yaml").read_text(encoding="utf-8")
        )
        hybrid_base["llm"]["global_search"] = True
        hybrid_base["evolution"]["population_size"] = 6
        hybrid_base["evolution"]["llm_mutations"] = 2
        hybrid_base["evolution"]["random_mutations"] = 0
        fd, path = tempfile.mkstemp(suffix=".yaml")
        import os

        os.close(fd)
        spec = Path(path)
        spec.write_text(yaml.safe_dump(hybrid_base), encoding="utf-8")
        try:
            with (
                patch(
                    "worldspace.generators.call_llm_vision",
                    return_value="sparse active filaments",
                ) as m_vis,
                patch(
                    "worldspace.generators.call_llm", return_value=_PATCH_JSON
                ) as m_txt,
            ):
                gen = HybridGALlmWorldGenerator(
                    grid_size=8, steps=8, seed=2, spec_path=spec
                )
                worlds = gen.generate(2)
            self.assertEqual(len(worlds), 2)
            self.assertGreaterEqual(m_vis.call_count, 1)
            self.assertGreaterEqual(m_txt.call_count, 1)
        finally:
            spec.unlink()

    def test_config_from_dict_defaults_global_search_false(self) -> None:
        cfg = LLMGeneratorConfig.from_llm_dict(
            {
                "mode": "local",
                "active_provider": "ollama",
                "providers": {"ollama": {"api_base": "http://x", "model": "m"}},
            }
        )
        self.assertFalse(cfg.global_search)
        self.assertEqual(cfg.vision_provider, "ollama")

"""Load tests for Q1 v2 CVT nightly LLM scheduler specs."""

from __future__ import annotations

import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SPECS = _REPO_ROOT / "worldspace" / "specs"


class TestQ1CvtSchedulers(unittest.TestCase):
    def test_nightly_cvt_llm_schedulers_load(self) -> None:
        from worldspace.illuminators.scheduler import load_scheduler

        cases = {
            "stub": ("map_elites_scheduler_nightly_llm_stub_cvt.yaml", False, False),
            "hints": ("map_elites_scheduler_nightly_llm_cvt.yaml", True, False),
            "filter": (
                "map_elites_scheduler_nightly_llm_filter_cvt.yaml",
                True,
                True,
            ),
            "shadow_hints": (
                "map_elites_scheduler_nightly_llm_shadow_hints_cvt.yaml",
                True,
                False,
            ),
            "shadow": (
                "map_elites_scheduler_nightly_llm_shadow_cvt.yaml",
                True,
                True,
            ),
        }
        for name, (filename, surrogate, acquisition) in cases.items():
            with self.subTest(scheduler=name):
                config = load_scheduler(_SPECS / filename)
                self.assertEqual(config.schema_version, "1.3")
                self.assertEqual(config.archive_type, "cvt")
                self.assertEqual(config.n_centroids, 2500)
                self.assertEqual(config.n_cells, 2500)
                self.assertEqual(config.iterations, 650)
                self.assertEqual(config.batch_emitters.count("llm"), 10)
                self.assertTrue(config.llm_enabled)
                self.assertEqual(config.surrogate_enabled, surrogate)
                self.assertTrue(config.performance.parallel_eval)
                if acquisition:
                    self.assertIn(
                        config.acquisition.mode,
                        ("filter", "shadow"),
                    )

    def test_resolve_cvt_baseline_for_hints_scheduler(self) -> None:
        import tempfile

        import scripts.run_github_llm_map_elites as mod

        scheduler = _SPECS / "map_elites_scheduler_nightly_llm_cvt.yaml"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "map_elites_nightly"
            baseline = root / "cvt" / "baseline" / "map_elites_archive.jsonl"
            baseline.parent.mkdir(parents=True)
            baseline.write_text("{}\n", encoding="utf-8")
            orig_root = mod._NIGHTLY_ROOT
            try:
                mod._NIGHTLY_ROOT = root
                resolved = mod.resolve_baseline_archive_for_scheduler(scheduler)
                self.assertEqual(resolved, baseline)
            finally:
                mod._NIGHTLY_ROOT = orig_root


if __name__ == "__main__":
    unittest.main()

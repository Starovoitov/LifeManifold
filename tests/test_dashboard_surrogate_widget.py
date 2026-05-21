"""Unit tests for dashboard surrogate widget helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


class TestDashboardSurrogateWidget(unittest.TestCase):
    def test_resolve_checkpoint_prefers_existing_file(self) -> None:
        from dashboard.components.surrogate_widget import resolve_checkpoint_path

        with tempfile.TemporaryDirectory() as tmp:
            primary = Path(tmp) / "latest.pkl"
            fallback = Path(tmp) / "micro.pkl"
            primary.write_bytes(b"not-a-real-pickle")
            cfg = {
                "paths": {"surrogate_checkpoint": str(primary)},
                "surrogate": {"micro_checkpoint_fallback": str(fallback)},
            }
            resolved = resolve_checkpoint_path(cfg)
            self.assertEqual(resolved, primary.resolve())

    def test_feature_importance_none_for_untrained_model(self) -> None:
        from dashboard.components.surrogate_widget import feature_importance_from_model
        from worldspace.surrogate.model import SurrogateModel

        model = SurrogateModel(model_type="lightgbm")
        self.assertIsNone(feature_importance_from_model(model))

    def test_predict_world_spec_dict_returns_keys(self) -> None:
        from dashboard.components.surrogate_widget import predict_world_spec_dict
        from dashboard.utils.data_processing import flatten_archive_record
        import json

        smoke = (
            _REPO_ROOT / "artifacts" / "map_elites_smoke" / "map_elites_archive.jsonl"
        )
        if not smoke.is_file():
            self.skipTest("smoke archive missing")
        record = json.loads(smoke.read_text(encoding="utf-8").splitlines()[0])
        row = flatten_archive_record(record)
        result = predict_world_spec_dict(row["world_spec"])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("fitness", result)
        self.assertIn("uncertainty", result)

    def test_predict_world_spec_dict_invalid_spec_returns_none(self) -> None:
        from dashboard.components.surrogate_widget import predict_world_spec_dict

        self.assertIsNone(predict_world_spec_dict({}))

    def test_surrogate_status_stub_without_checkpoint(self) -> None:
        from dashboard.components.surrogate_widget import surrogate_status

        cfg = {
            "paths": {"surrogate_checkpoint": "nonexistent/checkpoints/missing.pkl"},
            "surrogate": {
                "enabled": True,
                "micro_checkpoint_fallback": "also/missing.pkl",
            },
        }
        status = surrogate_status(cfg)
        self.assertTrue(status.is_stub)


if __name__ == "__main__":
    unittest.main()

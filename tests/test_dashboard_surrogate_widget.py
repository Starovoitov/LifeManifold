"""Unit tests for dashboard surrogate widget helpers."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from worldspace.surrogate.calibration import (
    CALIBRATION_METHOD_ISOTONIC,
    CALIBRATION_SCHEMA_VERSION,
    UncertaintyCalibrator,
)
from worldspace.surrogate.checkpoint_io import save_surrogate_checkpoint
from worldspace.surrogate.model import SurrogateModel

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_model_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_surrogate_checkpoint(SurrogateModel(), path)


def _write_calibration_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    calibrator = UncertaintyCalibrator(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        method=CALIBRATION_METHOD_ISOTONIC,
        x_thresholds=(0.0, 1.0),
        y_thresholds=(0.0, 1.0),
    )
    with path.open("wb") as handle:
        pickle.dump(calibrator, handle)


class TestDashboardSurrogateWidget(unittest.TestCase):
    def test_resolve_checkpoint_explicit_none_skips_archive_local(self) -> None:
        from dashboard.components.surrogate_widget import resolve_checkpoint_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "run" / "map_elites_archive.jsonl"
            archive.parent.mkdir(parents=True)
            archive.write_text("{}\n", encoding="utf-8")
            local = archive.parent / "models" / "local.pkl"
            _write_model_checkpoint(local)
            cfg: dict[str, object] = {"paths": {}, "surrogate": {}}
            resolved = resolve_checkpoint_path(cfg, archive_path=archive)
            self.assertEqual(resolved, local.resolve())
            self.assertIsNone(
                resolve_checkpoint_path(
                    cfg,
                    archive_path=archive,
                    checkpoint_path=None,
                )
            )

    def test_resolve_checkpoint_path_rejects_non_path_explicit(self) -> None:
        from dashboard.components.surrogate_widget import resolve_checkpoint_path

        with self.assertRaises(TypeError):
            resolve_checkpoint_path(checkpoint_path="not-a-path")  # type: ignore[arg-type]

    def test_format_repo_relative_path_with_symlink(self) -> None:
        from dashboard.components.artifact_selectors import (
            format_repo_relative_path,
            format_repo_relative_path_with_symlink,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "real.pkl"
            target.write_bytes(b"x")
            link = root / "linked.pkl"
            link.symlink_to(target)
            self.assertEqual(
                format_repo_relative_path_with_symlink(link),
                f"{format_repo_relative_path(link)} -> {format_repo_relative_path(target)}",
            )

    def test_checkpoint_paths_found_recursively_under_run_dir(self) -> None:
        from dashboard.utils.config import checkpoint_paths_near_archive

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "experiments" / "cond" / "0" / "map_elites_archive.jsonl"
            archive.parent.mkdir(parents=True)
            archive.write_text("{}\n", encoding="utf-8")
            nested = archive.parent / "artifacts" / "model.pkl"
            _write_model_checkpoint(nested)
            found = checkpoint_paths_near_archive(archive)
            self.assertEqual(found, [nested.resolve()])

    def test_checkpoint_paths_ignore_pkl_beyond_search_depth(self) -> None:
        from dashboard.utils.config import (
            CHECKPOINT_SEARCH_MAX_DEPTH,
            checkpoint_paths_near_archive,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "run" / "map_elites_archive.jsonl"
            archive.parent.mkdir(parents=True)
            archive.write_text("{}\n", encoding="utf-8")
            shallow = archive.parent / "checkpoints" / "model.pkl"
            _write_model_checkpoint(shallow)
            deep_parts = ["d"] * (CHECKPOINT_SEARCH_MAX_DEPTH + 1)
            deep = archive.parent.joinpath(*deep_parts, "deep.pkl")
            _write_model_checkpoint(deep)
            found = checkpoint_paths_near_archive(archive)
            self.assertEqual(found, [shallow.resolve()])

    def test_checkpoint_paths_exclude_calibration_pickle(self) -> None:
        from dashboard.utils.config import checkpoint_paths_near_archive

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "run" / "map_elites_archive.jsonl"
            archive.parent.mkdir(parents=True)
            archive.write_text("{}\n", encoding="utf-8")
            _write_calibration_checkpoint(archive.parent / "calibration.pkl")
            model = archive.parent / "model.pkl"
            _write_model_checkpoint(model)
            found = checkpoint_paths_near_archive(archive)
            self.assertEqual(found, [model.resolve()])

    def test_checkpoint_paths_include_valid_symlink(self) -> None:
        from dashboard.utils.config import checkpoint_paths_near_archive

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "run" / "map_elites_archive.jsonl"
            archive.parent.mkdir(parents=True)
            archive.write_text("{}\n", encoding="utf-8")
            target = root / "bundle" / "model.pkl"
            _write_model_checkpoint(target)
            link = archive.parent / "checkpoints" / "linked.pkl"
            link.parent.mkdir(parents=True)
            link.symlink_to(target)
            found = checkpoint_paths_near_archive(archive)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].resolve(), target.resolve())
            self.assertTrue(found[0].is_symlink())

    def test_resolve_checkpoint_uses_config_only_when_session_override_set(
        self,
    ) -> None:
        from dashboard.components.surrogate_widget import resolve_checkpoint_path

        with tempfile.TemporaryDirectory() as tmp:
            primary = Path(tmp) / "configured" / "latest.pkl"
            fallback = Path(tmp) / "configured" / "micro.pkl"
            archive = Path(tmp) / "run" / "map_elites_archive.jsonl"
            archive.parent.mkdir(parents=True)
            primary.parent.mkdir(parents=True)
            archive.write_text("{}\n", encoding="utf-8")
            _write_model_checkpoint(primary)
            _write_model_checkpoint(fallback)
            cfg = {
                "paths": {"surrogate_checkpoint": str(primary)},
                "surrogate": {"micro_checkpoint_fallback": str(fallback)},
            }
            resolved = resolve_checkpoint_path(cfg, archive_path=archive)
            self.assertIsNone(resolved)
            resolved_explicit = resolve_checkpoint_path(
                cfg,
                archive_path=archive,
                checkpoint_path=primary,
            )
            self.assertEqual(resolved_explicit, primary.resolve())

    def test_resolve_checkpoint_prefers_archive_adjacent_checkpoints(self) -> None:
        from dashboard.utils.config import resolve_surrogate_checkpoint_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "baseline" / "map_elites_archive.jsonl"
            archive.parent.mkdir(parents=True)
            archive.write_text("{}\n", encoding="utf-8")
            checkpoint = root / "checkpoints" / "model.pkl"
            _write_model_checkpoint(checkpoint)
            config_path = root / "missing" / "model.pkl"
            cfg = {
                "paths": {"surrogate_checkpoint": str(config_path)},
                "surrogate": {"checkpoint_fallbacks": []},
            }
            resolved = resolve_surrogate_checkpoint_path(
                cfg,
                archive_path=archive,
            )
            self.assertEqual(resolved, checkpoint.resolve())

    def test_experiment_archive_does_not_auto_pick_global_checkpoint(self) -> None:
        from dashboard.utils.config import (
            checkpoint_paths_near_archive,
            list_surrogate_checkpoint_candidates,
        )

        repo = _REPO_ROOT
        global_ckpt = (
            repo
            / "artifacts"
            / "map-elites-nightly-surrogate"
            / "checkpoints"
            / "nightly_v2.pkl"
        )
        if not global_ckpt.is_file():
            self.skipTest("surrogate checkpoint missing")
        experiment_archive = (
            repo
            / "artifacts"
            / "experiments"
            / "q1-min"
            / "stub"
            / "seed_0"
            / "map_elites_archive.jsonl"
        )
        if not experiment_archive.is_file():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                experiment_archive = (
                    root
                    / "artifacts"
                    / "experiments"
                    / "q1-min"
                    / "stub"
                    / "seed_0"
                    / "map_elites_archive.jsonl"
                )
                experiment_archive.parent.mkdir(parents=True)
                experiment_archive.write_text("{}\n", encoding="utf-8")
                global_dir = (
                    root / "artifacts" / "map-elites-nightly-surrogate" / "checkpoints"
                )
                _write_model_checkpoint(global_dir / "model.pkl")
                cfg = {
                    "paths": {"surrogate_checkpoint": str(global_dir / "model.pkl")},
                    "surrogate": {"checkpoint_fallbacks": []},
                }
                near = {
                    str(path.resolve())
                    for path in checkpoint_paths_near_archive(experiment_archive)
                }
                self.assertNotIn(str((global_dir / "model.pkl").resolve()), near)
                manual = list_surrogate_checkpoint_candidates(
                    cfg,
                    archive_path=experiment_archive,
                )
                self.assertEqual(manual, [])
        else:
            near = checkpoint_paths_near_archive(experiment_archive)
            resolved_near = {path.resolve() for path in near}
            self.assertNotIn(global_ckpt.resolve(), resolved_near)
            self.assertTrue(
                all("calibration.pkl" not in path.name for path in near),
                msg=f"calibration.pkl must not appear: {near}",
            )
            manual = list_surrogate_checkpoint_candidates(
                {},
                archive_path=experiment_archive,
            )
            self.assertEqual({path.resolve() for path in manual}, resolved_near)

    def test_list_surrogate_checkpoint_candidates_only_archive_local(self) -> None:
        from dashboard.utils.config import list_surrogate_checkpoint_candidates

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "run" / "map_elites_archive.jsonl"
            archive.parent.mkdir(parents=True)
            archive.write_text("{}\n", encoding="utf-8")
            local = archive.parent / "checkpoints" / "model.pkl"
            _write_model_checkpoint(local)
            configured = root / "configured.pkl"
            _write_model_checkpoint(configured)
            cfg = {"paths": {"surrogate_checkpoint": str(configured)}}
            candidates = list_surrogate_checkpoint_candidates(cfg, archive_path=archive)
            self.assertEqual(candidates, [local.resolve()])

    def test_feature_importance_none_for_untrained_model(self) -> None:
        from dashboard.components.surrogate_widget import feature_importance_from_model

        model = SurrogateModel(model_type="lightgbm")
        self.assertIsNone(feature_importance_from_model(model))

    def test_predict_world_spec_dict_default_preserves_checkpoint_auto_discovery(
        self,
    ) -> None:
        from unittest import mock

        from dashboard.utils.config import UNSET
        from dashboard.components.surrogate_widget import predict_world_spec_dict

        world_spec = {
            "birth": [3, 7, 8],
            "survival": [3, 4, 5, 8],
            "noise": 0.0,
            "resource_regen": 0.15,
            "predation": 0.1,
            "cell_types": ["life", "food"],
            "neighborhood": "moore",
            "grid_size": 50,
            "steps": 200,
            "seed": 1,
        }
        archive = Path("/tmp/run/map_elites_archive.jsonl")
        with mock.patch(
            "dashboard.components.surrogate_widget.load_surrogate"
        ) as load_mock:
            load_mock.return_value.predict.return_value.fitness = 0.42
            load_mock.return_value.predict.return_value.uncertainty = 0.1
            result = predict_world_spec_dict(world_spec, archive_path=archive)
            self.assertIsNotNone(result)
            self.assertIs(
                load_mock.call_args.kwargs["checkpoint_path"],
                UNSET,
            )

            load_mock.reset_mock()
            load_mock.return_value.predict.return_value.fitness = 0.5
            load_mock.return_value.predict.return_value.uncertainty = 0.85
            predict_world_spec_dict(
                world_spec,
                archive_path=archive,
                checkpoint_path=None,
            )
            self.assertIsNone(load_mock.call_args.kwargs["checkpoint_path"])

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

        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "map_elites_archive.jsonl"
            archive.write_text("{}\n", encoding="utf-8")
            cfg = {
                "paths": {
                    "surrogate_checkpoint": "nonexistent/checkpoints/missing.pkl"
                },
                "surrogate": {
                    "enabled": True,
                    "micro_checkpoint_fallback": "also/missing.pkl",
                },
            }
            status = surrogate_status(cfg, archive_path=archive, checkpoint_path=None)
        self.assertTrue(status.is_stub)

    def test_feature_importance_returns_21_v2_labels(self) -> None:
        from dashboard.components.surrogate_widget import feature_importance_from_model
        from worldspace.surrogate.feature_extractor import FEATURE_NAMES
        from worldspace.surrogate.model import TARGET_KEYS, SurrogateModel

        model = SurrogateModel(model_type="lightgbm")
        estimator = _MockLightGbmEstimator(
            np.arange(len(FEATURE_NAMES), dtype=np.float64)
        )
        model._uses_lightgbm = True
        model._ensemble = {key: [estimator] for key in TARGET_KEYS}
        importances = feature_importance_from_model(model)
        self.assertIsNotNone(importances)
        assert importances is not None
        self.assertEqual(set(importances.keys()), set(FEATURE_NAMES))

    def test_feature_importance_none_for_wrong_feature_dim(self) -> None:
        from dashboard.components.surrogate_widget import feature_importance_from_model
        from worldspace.surrogate.model import TARGET_KEYS, SurrogateModel

        model = SurrogateModel(model_type="lightgbm")
        estimator = _MockLightGbmEstimator(np.ones(8, dtype=np.float64))
        model._uses_lightgbm = True
        model._ensemble = {key: [estimator] for key in TARGET_KEYS}
        self.assertIsNone(feature_importance_from_model(model))


class _MockLightGbmEstimator:
    def __init__(self, importances: np.ndarray) -> None:
        self.feature_importances_ = importances
        self.n_features_in_ = int(importances.size)


if __name__ == "__main__":
    unittest.main()

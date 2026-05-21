"""Ensure Streamlit entry points can import the dashboard package."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DASHBOARD_DIR = _REPO_ROOT / "dashboard"


class TestDashboardPathSetup(unittest.TestCase):
    def test_install_paths_from_home_entry(self) -> None:
        import importlib

        if "path_setup" in sys.modules:
            del sys.modules["path_setup"]
        sys.path[:] = [
            p for p in sys.path if p not in {str(_REPO_ROOT), str(_DASHBOARD_DIR)}
        ]

        spec = importlib.util.spec_from_file_location(
            "path_setup",
            _DASHBOARD_DIR / "path_setup.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        repo = module.install_paths(_DASHBOARD_DIR / "Home.py")
        self.assertEqual(repo, _REPO_ROOT)
        self.assertIn(str(_REPO_ROOT), sys.path)

        import dashboard.utils.config as config_mod

        self.assertTrue(hasattr(config_mod, "load_config"))

    def test_install_paths_from_pages_entry(self) -> None:
        import importlib

        if "path_setup" in sys.modules:
            del sys.modules["path_setup"]
        sys.path[:] = [
            p for p in sys.path if p not in {str(_REPO_ROOT), str(_DASHBOARD_DIR)}
        ]

        spec = importlib.util.spec_from_file_location(
            "path_setup_pages",
            _DASHBOARD_DIR / "path_setup.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        page = _DASHBOARD_DIR / "pages" / "1_Archive_Explorer.py"
        repo = module.install_paths(page)
        self.assertEqual(repo, _REPO_ROOT)

        import dashboard.utils.bootstrap as bootstrap_mod

        bootstrap_mod.ensure_repo_on_path()
        self.assertIn(str(_REPO_ROOT), sys.path)


if __name__ == "__main__":
    unittest.main()

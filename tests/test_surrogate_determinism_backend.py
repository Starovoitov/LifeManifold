"""Unit tests for surrogate backend determinism helpers (E6.1)."""

from __future__ import annotations

import unittest

from worldspace.surrogate.determinism import (
    DEFAULT_ENSEMBLE_SIZE,
    DEFAULT_RANDOM_STATE,
    lightgbm_deterministic_params,
    member_random_state,
)


class TestDeterminismHelpers(unittest.TestCase):
    def test_defaults_match_tz(self) -> None:
        self.assertEqual(DEFAULT_RANDOM_STATE, 42)
        self.assertEqual(DEFAULT_ENSEMBLE_SIZE, 8)

    def test_lightgbm_params_include_deterministic_flag(self) -> None:
        params = lightgbm_deterministic_params()
        self.assertTrue(params["deterministic"])
        self.assertEqual(params["random_state"], 42)

    def test_member_random_state_is_stable(self) -> None:
        self.assertEqual(member_random_state(0), 42)
        self.assertEqual(member_random_state(3), 45)


if __name__ == "__main__":
    unittest.main()

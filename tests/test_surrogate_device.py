"""Tests for surrogate training device resolution."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from worldspace.surrogate.device import (
    lightgbm_gpu_available,
    resolve_lightgbm_device,
    resolve_training_device,
)


class TestSurrogateDevice(unittest.TestCase):
    def test_resolve_training_device_auto_without_cuda(self) -> None:
        with patch("worldspace.surrogate.device.cuda_available", return_value=False):
            self.assertEqual(resolve_training_device("auto"), "cpu")

    def test_resolve_training_device_auto_with_cuda(self) -> None:
        with patch("worldspace.surrogate.device.cuda_available", return_value=True):
            self.assertEqual(resolve_training_device("auto"), "cuda")

    def test_resolve_training_device_cpu(self) -> None:
        with patch("worldspace.surrogate.device.cuda_available", return_value=True):
            self.assertEqual(resolve_training_device("cpu"), "cpu")

    def test_resolve_training_device_cuda_falls_back_to_cpu(self) -> None:
        with patch("worldspace.surrogate.device.cuda_available", return_value=False):
            self.assertEqual(resolve_training_device("cuda"), "cpu")

    def test_resolve_lightgbm_device_uses_cpu_when_cuda_unavailable(self) -> None:
        with patch("worldspace.surrogate.device.cuda_available", return_value=False):
            self.assertEqual(resolve_lightgbm_device("auto"), "cpu")

    def test_resolve_lightgbm_device_falls_back_when_probe_fails(self) -> None:
        with (
            patch("worldspace.surrogate.device.cuda_available", return_value=True),
            patch(
                "worldspace.surrogate.device.lightgbm_gpu_available",
                return_value=False,
            ),
        ):
            self.assertEqual(resolve_lightgbm_device("auto"), "cpu")

    def test_lightgbm_gpu_available_is_cached(self) -> None:
        import worldspace.surrogate.device as device_module

        device_module._lightgbm_gpu_cache = None
        with patch("worldspace.surrogate.device.cuda_available", return_value=False):
            self.assertFalse(lightgbm_gpu_available())
            self.assertFalse(lightgbm_gpu_available())


if __name__ == "__main__":
    unittest.main()

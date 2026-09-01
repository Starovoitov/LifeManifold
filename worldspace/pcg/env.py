"""Pinned PCG Benchmark environment wrapper (in-process, no hidden network)."""

from __future__ import annotations

from typing import Any

PINNED_COMMIT = "cd0f55b26c412a26e8797193e5417f5e651cf6cd"
PINNED_VERSION = "0.1.0"
PINNED_LICENSE = "MIT"
PINNED_REPO = "https://github.com/amidos2006/pcg_benchmark"


class BenchmarkPcgEnv:
    """Thin wrapper so smoke can seed sampling without using diversity as fitness."""

    def __init__(self, problem_name: str, *, seed: int) -> None:
        import pcg_benchmark

        self.problem_name = problem_name
        self._env = pcg_benchmark.make(problem_name)
        self._env.seed(seed)

    def seed(self, seed: int) -> None:
        self._env.seed(seed)

    def sample_content(self) -> object:
        return self._env.content_space.sample()

    def info(self, contents: object) -> dict[str, Any]:
        return _one_info_dict(self._env.info(contents), origin="info")

    def quality(self, contents: object) -> tuple[float, float, dict[str, Any]]:
        passed, quality, info = self._env.quality(contents)
        return float(passed), float(quality), _one_info_dict(info, origin="quality")


def _one_info_dict(payload: object, *, origin: str) -> dict[str, Any]:
    if isinstance(payload, list):
        if len(payload) != 1:
            raise TypeError(
                f"pcg env.{origin} must return one info dict for one content, "
                f"got a list of {len(payload)}"
            )
        payload = payload[0]
    if not isinstance(payload, dict):
        raise TypeError(f"pcg env.{origin} must return an info dict for one content")
    return {str(key): value for key, value in payload.items()}

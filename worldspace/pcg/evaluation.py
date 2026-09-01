"""Search-only PCG evaluation. Structurally invalid grids never call A*."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from worldspace.pcg.descriptors import PcgBinEdges, bin_for_measures
from worldspace.pcg.spec import PcgSpec, PcgTask, try_parse_grid


class PcgEnvLike(Protocol):
    def info(self, contents: object) -> dict[str, Any]: ...

    def quality(self, contents: object) -> tuple[float, float, dict[str, Any]]: ...


@dataclass(frozen=True)
class PcgEvaluation:
    spec: PcgSpec | None
    structurally_valid: bool
    fitness: float | None
    measures: tuple[float, float] | None
    bin: tuple[int, int] | None
    playable: bool
    info_keys: tuple[str, ...]
    miss_reason: str | None


def measures_from_info(task: PcgTask, info: dict[str, Any]) -> tuple[float, float]:
    name0, name1 = task.measure_names
    return (_measure_value(task, info, name0), _measure_value(task, info, name1))


def _measure_value(task: PcgTask, info: dict[str, Any], name: str) -> float:
    if name == "solution_length":
        solution = info.get("solution") or []
        return float(len(solution))
    value = info[name]
    return float(value)


def evaluate_fitness_measures(
    spec: PcgSpec,
    env: PcgEnvLike,
    task: PcgTask,
) -> tuple[float, tuple[float, float], dict[str, Any]]:
    content = spec.to_nested_list()
    passed, quality, info = env.quality(content)
    del passed
    if not isinstance(info, dict):
        raise TypeError("pcg env.quality must return an info dict for one content")
    return float(quality), measures_from_info(task, info), info


def evaluate_spec(
    spec: PcgSpec,
    env: PcgEnvLike,
    edges: PcgBinEdges,
    task: PcgTask,
) -> PcgEvaluation:
    quality, measures, info = evaluate_fitness_measures(spec, env, task)
    return PcgEvaluation(
        spec=spec,
        structurally_valid=True,
        fitness=quality,
        measures=measures,
        bin=bin_for_measures(measures, edges),
        playable=quality >= 1.0,
        info_keys=tuple(sorted(info)),
        miss_reason=None,
    )


def evaluate_payload(
    payload: object,
    env: PcgEnvLike,
    edges: PcgBinEdges,
    task: PcgTask,
) -> PcgEvaluation:
    spec = try_parse_grid(payload, task)
    if spec is None:
        return PcgEvaluation(
            spec=None,
            structurally_valid=False,
            fitness=None,
            measures=None,
            bin=None,
            playable=False,
            info_keys=(),
            miss_reason="structurally_invalid",
        )
    return evaluate_spec(spec, env, edges, task)

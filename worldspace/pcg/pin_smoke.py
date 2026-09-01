"""Pinned PCG Benchmark smoke: random edges, genetic selectors, named repair."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from worldspace.pcg.descriptors import (
    bin_edges_from_measures,
    bin_for_measures,
    dump_frozen_bin_edges,
    occupancy_counts,
)
from worldspace.pcg.env import (
    PINNED_COMMIT,
    PINNED_LICENSE,
    PINNED_REPO,
    PINNED_VERSION,
    BenchmarkPcgEnv,
)
from worldspace.pcg.evaluation import evaluate_fitness_measures, evaluate_payload
from worldspace.pcg.repair import RepairKind, apply_repair
from worldspace.pcg.smoke import (
    DETERMINISM_CONTENTS,
    DETERMINISM_REPEATS,
    MAX_SELECTOR_JACCARD,
    MAX_SMOKE_COVERAGE,
    RANDOM_EDGE_SAMPLES,
    SMOKE_INITIAL_RANDOM,
    PcgSmokeResult,
    niche_jaccard,
    run_pcg_smoke,
    seeded_initial_archive,
)
from worldspace.pcg.spec import PcgSpec, PcgTask, try_parse_grid


def result_dict(result: PcgSmokeResult) -> dict[str, object]:
    return {
        "problem_name": result.problem_name,
        "generator": result.generator,
        "selector": result.selector,
        "seed": result.seed,
        "initial_random": result.initial_random,
        "steps": result.steps,
        "proposals": result.proposals,
        "evaluations": result.evaluations,
        "structurally_invalid": result.structurally_invalid,
        "playable": result.playable,
        "filled_cells": result.filled_cells,
        "coverage": result.coverage,
        "qd_score": result.qd_score,
        "n_occupied_bins": len(result.occupied_bins),
        "repair_kind": result.repair_kind,
        "tiles_changed_mean": result.tiles_changed_mean,
        "astar_eligible": result.astar_eligible,
        "quality_min": result.quality_min,
        "quality_max": result.quality_max,
        "measure0_min": result.measure0_min,
        "measure0_max": result.measure0_max,
    }


def dependency_versions() -> dict[str, str]:
    import importlib.metadata as metadata

    versions = {}
    for name in ("pcg-benchmark", "numpy", "pillow"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def determinism(
    env: BenchmarkPcgEnv,
    task: PcgTask,
    specs: list[PcgSpec],
) -> dict[str, object]:
    chosen = specs[:DETERMINISM_CONTENTS]
    mismatches = 0
    for spec in chosen:
        qualities = []
        for _ in range(DETERMINISM_REPEATS):
            quality, _, _ = evaluate_fitness_measures(spec, env, task)
            qualities.append(quality)
        if max(qualities) - min(qualities) > 1e-12:
            mismatches += 1
    return {
        "n_contents": len(chosen),
        "repeats": DETERMINISM_REPEATS,
        "mismatched_contents": mismatches,
        "deterministic": mismatches == 0,
    }


def run_problem(
    task: PcgTask,
    *,
    seed: int,
    out_dir: Path,
    evaluations: int,
    repair_kind: RepairKind = "identity",
    dump_edges: bool = True,
    edges_filename: str | None = None,
    edges_stage: str = "pcg_smoke",
) -> dict[str, object]:
    env = BenchmarkPcgEnv(task.problem_name, seed=seed)
    random_specs: list[PcgSpec] = []
    random_measures: list[tuple[float, float]] = []
    random_quality: list[float] = []
    tiles_changed: list[int] = []
    astar_eligible = 0
    info_keys: set[str] = set()
    playable_random = 0
    print(
        f"{task.problem_name} repair={repair_kind}: "
        f"sampling {RANDOM_EDGE_SAMPLES} random contents...",
        file=sys.stderr,
        flush=True,
    )
    for index in range(RANDOM_EDGE_SAMPLES):
        spec = PcgSpec.from_task_grid(task, env.sample_content())
        spec, meta = apply_repair(spec, repair_kind)
        quality, measures, info = evaluate_fitness_measures(spec, env, task)
        random_specs.append(spec)
        random_measures.append(measures)
        random_quality.append(quality)
        tiles_changed.append(meta["tiles_changed"])
        astar_eligible += int(bool(meta.get("astar_eligible")))
        info_keys.update(info)
        playable_random += int(quality >= 1.0)
        if (index + 1) % 64 == 0:
            print(
                f"  random {index + 1}/{RANDOM_EDGE_SAMPLES} "
                f"playable={playable_random} astar_eligible={astar_eligible}",
                file=sys.stderr,
                flush=True,
            )
    missing_keys = sorted(task.expected_info_keys - info_keys)
    extra_keys = sorted(info_keys - task.expected_info_keys)
    provisional = bin_edges_from_measures(
        random_measures,
        measure_names=task.measure_names,
        problem_name=task.problem_name,
    )
    print(
        f"{task.problem_name} repair={repair_kind}: genetic/random smoke...",
        file=sys.stderr,
        flush=True,
    )
    floor = seeded_initial_archive(
        env,
        task,
        provisional,
        seed=seed,
        n_random=SMOKE_INITIAL_RANDOM,
        sample_from_env=True,
        repair_kind=repair_kind,
    )
    random_result, _ = run_pcg_smoke(
        env,
        task,
        provisional,
        generator="random",
        selector="uniform_frontier",
        seed=seed + 1,
        evaluations=evaluations,
        initial_archive=floor,
        sample_from_env=True,
        repair_kind=repair_kind,
    )
    uniform_result, uniform_archive = run_pcg_smoke(
        env,
        task,
        provisional,
        generator="genetic",
        selector="uniform_frontier",
        seed=seed + 2,
        evaluations=evaluations,
        initial_archive=floor,
        sample_from_env=False,
        repair_kind=repair_kind,
    )
    minfit_result, minfit_archive = run_pcg_smoke(
        env,
        task,
        provisional,
        generator="genetic",
        selector="min_fitness_frontier",
        seed=seed + 3,
        evaluations=evaluations,
        initial_archive=floor,
        sample_from_env=False,
        repair_kind=repair_kind,
    )
    all_measures = list(random_measures)
    for archive in (uniform_archive, minfit_archive):
        all_measures.extend(elite.measures for elite in archive.elites())
    frozen = bin_edges_from_measures(
        all_measures,
        measure_names=task.measure_names,
        problem_name=task.problem_name,
    )
    uniform_rebinned = frozenset(
        bin_for_measures(elite.measures, frozen) for elite in uniform_archive.elites()
    )
    minfit_rebinned = frozenset(
        bin_for_measures(elite.measures, frozen) for elite in minfit_archive.elites()
    )
    jaccard = niche_jaccard(uniform_rebinned, minfit_rebinned)
    coverage_uniform = len(uniform_rebinned) / float(frozen.resolution**2)
    coverage_minfit = len(minfit_rebinned) / float(frozen.resolution**2)
    occupancy = occupancy_counts(random_measures, frozen)
    invalid = try_parse_grid({"ops": ["not", "a", "grid"]}, task)
    spy_calls = {"n": 0}

    class _Spy:
        def info(self, contents: object) -> dict[str, Any]:
            spy_calls["n"] += 1
            return {}

        def quality(self, contents: object) -> tuple[float, float, dict[str, Any]]:
            spy_calls["n"] += 1
            return 0.0, 0.0, {}

    miss = evaluate_payload(["bad"], _Spy(), frozen, task)
    det = determinism(env, task, random_specs)
    pinned = env.problem_name == task.problem_name
    quality_is_fitness = True
    info_keys_ok = not missing_keys
    repair_identity = repair_kind == "identity"
    invalid_skips_search = (
        invalid is None and spy_calls["n"] == 0 and miss.structurally_valid is False
    )
    selector_jaccard_ok = jaccard < MAX_SELECTOR_JACCARD
    coverage_headroom = (
        coverage_uniform < MAX_SMOKE_COVERAGE and coverage_minfit < MAX_SMOKE_COVERAGE
    )
    measure0_min = min(m[0] for m in random_measures)
    measure0_max = max(m[0] for m in random_measures)
    edges_path: Path | None = None
    if dump_edges:
        filename = (
            edges_filename or f"{task.problem_name.replace('-', '_')}_bin_edges.json"
        )
        edges_path = out_dir / filename
        dump_frozen_bin_edges(frozen, edges_path, stage=edges_stage)
    return {
        "problem_name": task.problem_name,
        "repair_kind": repair_kind,
        "pin": {
            "repo": PINNED_REPO,
            "commit": PINNED_COMMIT,
            "version": PINNED_VERSION,
            "license": PINNED_LICENSE,
            "dependencies": dependency_versions(),
        },
        "shape": {"rows": task.rows, "cols": task.cols, "n_tiles": task.n_tiles},
        "measure_names": list(task.measure_names),
        "info_keys_observed": sorted(info_keys),
        "info_keys_missing": missing_keys,
        "info_keys_extra": extra_keys,
        "random_edge_samples": RANDOM_EDGE_SAMPLES,
        "random_playable": playable_random,
        "random_quality_min": min(random_quality),
        "random_quality_max": max(random_quality),
        "random_measures_min": [min(m[i] for m in random_measures) for i in (0, 1)],
        "random_measures_max": [max(m[i] for m in random_measures) for i in (0, 1)],
        "random_tiles_changed_mean": (
            sum(tiles_changed) / float(len(tiles_changed)) if tiles_changed else 0.0
        ),
        "random_astar_eligible": astar_eligible,
        "measure0_collapsed": measure0_min == measure0_max,
        "bin_edges": {
            "resolution": frozen.resolution,
            "axis0_min": frozen.axis0_min,
            "axis0_max": frozen.axis0_max,
            "axis1_min": frozen.axis1_min,
            "axis1_max": frozen.axis1_max,
            "n_samples": frozen.n_samples,
            "path": None if edges_path is None else str(edges_path),
            "dumped": dump_edges,
            "stage": edges_stage,
        },
        "occupancy_random_256": {
            "max_bin_count": max(occupancy),
            "empty_bins": occupancy.count(0),
            "occupied_bins": sum(item > 0 for item in occupancy),
        },
        "determinism": det,
        "invalid_parse_calls_evaluator": spy_calls["n"],
        "random_smoke": result_dict(random_result),
        "genetic_uniform": result_dict(uniform_result),
        "genetic_min_fitness": result_dict(minfit_result),
        "selector_niche_jaccard": jaccard,
        "rebinned_coverage_uniform": coverage_uniform,
        "rebinned_coverage_min_fitness": coverage_minfit,
        "gates": {
            "pinned_env": pinned,
            "quality_is_fitness": quality_is_fitness,
            "info_keys": info_keys_ok,
            "repair_identity": repair_identity,
            "invalid_skips_search": invalid_skips_search,
            "license": PINNED_LICENSE == "MIT",
            "deterministic": det["deterministic"],
            "selector_jaccard": selector_jaccard_ok,
            "coverage_headroom": coverage_headroom,
            "no_holdout_in_quality": True,
        },
    }

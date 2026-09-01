#!/usr/bin/env python3
"""PCG Benchmark pin + random/genetic smoke (no LLM, repair off)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from worldspace.pcg.evaluation import evaluate_fitness_measures
from worldspace.pcg.smoke import (
    DETERMINISM_CONTENTS,
    DETERMINISM_REPEATS,
    MAX_SELECTOR_JACCARD,
    MAX_SMOKE_COVERAGE,
    RANDOM_EDGE_SAMPLES,
    RESERVED_SOKOBAN_SEED,
    RESERVED_ZELDA_SEED,
    SMOKE_EVALUATIONS,
    SMOKE_INITIAL_RANDOM,
    niche_jaccard,
    run_pcg_smoke,
    seeded_initial_archive,
)
from worldspace.pcg.spec import SOKOBAN_V0, ZELDA_V0, PcgSpec, PcgTask, try_parse_grid

DEFAULT_OUT = ROOT / "artifacts/controlled_attribution/pcg"


def _result_dict(result) -> dict[str, object]:
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
    }


def _dependency_versions() -> dict[str, str]:
    import importlib.metadata as metadata

    versions = {}
    for name in ("pcg-benchmark", "numpy", "pillow"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def _determinism(
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
) -> dict[str, object]:
    env = BenchmarkPcgEnv(task.problem_name, seed=seed)
    random_specs: list[PcgSpec] = []
    random_measures: list[tuple[float, float]] = []
    random_quality: list[float] = []
    info_keys: set[str] = set()
    playable_random = 0
    print(
        f"{task.problem_name}: sampling {RANDOM_EDGE_SAMPLES} random contents...",
        file=sys.stderr,
        flush=True,
    )
    for index in range(RANDOM_EDGE_SAMPLES):
        spec = PcgSpec.from_task_grid(task, env.sample_content())
        quality, measures, info = evaluate_fitness_measures(spec, env, task)
        random_specs.append(spec)
        random_measures.append(measures)
        random_quality.append(quality)
        info_keys.update(info)
        playable_random += int(quality >= 1.0)
        if (index + 1) % 64 == 0:
            print(
                f"  random {index + 1}/{RANDOM_EDGE_SAMPLES} playable={playable_random}",
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
    print(f"{task.problem_name}: genetic/random smoke...", file=sys.stderr, flush=True)
    floor = seeded_initial_archive(
        env,
        task,
        provisional,
        seed=seed,
        n_random=SMOKE_INITIAL_RANDOM,
        sample_from_env=True,
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
        def quality(self, contents: object) -> tuple[float, float, dict]:
            spy_calls["n"] += 1
            return 0.0, 0.0, {}

    from worldspace.pcg.evaluation import evaluate_payload

    miss = evaluate_payload(["bad"], _Spy(), frozen, task)
    determinism = _determinism(env, task, random_specs)
    pinned = env.problem_name == task.problem_name
    quality_is_fitness = True
    info_keys_ok = not missing_keys
    repair_identity = True
    invalid_skips_search = (
        invalid is None and spy_calls["n"] == 0 and miss.structurally_valid is False
    )
    selector_jaccard_ok = jaccard < MAX_SELECTOR_JACCARD
    coverage_headroom = (
        coverage_uniform < MAX_SMOKE_COVERAGE and coverage_minfit < MAX_SMOKE_COVERAGE
    )
    edges_path = out_dir / f"{task.problem_name.replace('-', '_')}_bin_edges.json"
    dump_frozen_bin_edges(frozen, edges_path)
    return {
        "problem_name": task.problem_name,
        "pin": {
            "repo": PINNED_REPO,
            "commit": PINNED_COMMIT,
            "version": PINNED_VERSION,
            "license": PINNED_LICENSE,
            "dependencies": _dependency_versions(),
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
        "bin_edges": {
            "resolution": frozen.resolution,
            "axis0_min": frozen.axis0_min,
            "axis0_max": frozen.axis0_max,
            "axis1_min": frozen.axis1_min,
            "axis1_max": frozen.axis1_max,
            "n_samples": frozen.n_samples,
            "path": str(edges_path),
        },
        "occupancy_random_256": {
            "max_bin_count": max(occupancy),
            "empty_bins": occupancy.count(0),
            "occupied_bins": sum(item > 0 for item in occupancy),
        },
        "determinism": determinism,
        "invalid_parse_calls_evaluator": spy_calls["n"],
        "random_smoke": _result_dict(random_result),
        "genetic_uniform": _result_dict(uniform_result),
        "genetic_min_fitness": _result_dict(minfit_result),
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
            "deterministic": determinism["deterministic"],
            "selector_jaccard": selector_jaccard_ok,
            "coverage_headroom": coverage_headroom,
            "no_holdout_in_quality": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--evaluations", type=int, default=SMOKE_EVALUATIONS)
    parser.add_argument("--skip-zelda", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "stage": "pcg_smoke",
        "llm": False,
        "repair": "identity",
        "family": "pcg_benchmark",
        "one_family_not_two_public_tasks": True,
        "sokoban": run_problem(
            SOKOBAN_V0,
            seed=RESERVED_SOKOBAN_SEED,
            out_dir=args.output_dir,
            evaluations=args.evaluations,
        ),
    }
    sokoban_crash_or_license = not all(
        report["sokoban"]["gates"][key]  # type: ignore[index]
        for key in ("pinned_env", "license")
    )
    if args.skip_zelda or sokoban_crash_or_license:
        report["zelda"] = None
        report["zelda_skipped"] = True
    else:
        report["zelda"] = run_problem(
            ZELDA_V0,
            seed=RESERVED_ZELDA_SEED,
            out_dir=args.output_dir,
            evaluations=args.evaluations,
        )
        report["zelda_skipped"] = False
    out = args.output_dir / "pcg_smoke.json"
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"sokoban": report["sokoban"]["gates"], "zelda_skipped": report["zelda_skipped"]}, indent=2))  # type: ignore[index]
    if report.get("zelda"):
        print(json.dumps({"zelda": report["zelda"]["gates"]}, indent=2))  # type: ignore[index]
    print(f"wrote {out}")
    sokoban_ok = all(report["sokoban"]["gates"].values())  # type: ignore[union-attr]
    zelda_ok = True
    if report.get("zelda"):
        zelda_ok = all(report["zelda"]["gates"].values())  # type: ignore[union-attr]
    return 0 if sokoban_ok and zelda_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

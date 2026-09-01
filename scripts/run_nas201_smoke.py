"""NAS-Bench-201 pin + lookup smoke (no LLM, no test-set fitness)."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from worldspace.nas201.descriptors import (
    bin_edges_from_records,
    max_bin_fraction,
    occupancy_counts,
)
from worldspace.nas201.emitters import random_spec
from worldspace.nas201.evaluation import evaluate_payload, evaluate_spec
from worldspace.nas201.smoke import (
    RESERVED_SMOKE_SEED,
    SMOKE_EVALUATIONS,
    SMOKE_INITIAL_RANDOM,
    niche_jaccard,
    run_nas201_smoke,
    seeded_initial_archive,
)
from worldspace.nas201.spec import OPERATIONS, N_EDGES, Nas201Spec
from worldspace.nas201.table import CompactNas201Table, SpyLookup

EXPECTED_N = 15625
MAX_BIN_OCCUPANCY_FRACTION = 0.50
MAX_SELECTOR_JACCARD = 0.80
MAX_SMOKE_COVERAGE = 0.95
LOOKUP_SAMPLE = 200


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _result_dict(result) -> dict[str, object]:
    return {
        "generator": result.generator,
        "selector": result.selector,
        "seed": result.seed,
        "initial_random": result.initial_random,
        "steps": result.genetic_or_random_steps,
        "proposals": result.proposals,
        "evaluations": result.evaluations,
        "lookup_hits": result.lookup_hits,
        "lookup_misses": result.lookup_misses,
        "structurally_invalid": result.structurally_invalid,
        "filled_cells": result.filled_cells,
        "coverage": result.coverage,
        "qd_score": result.qd_score,
        "n_occupied_bins": len(result.occupied_bins),
    }


def run_nas201_lookup_smoke(
    table: CompactNas201Table,
    *,
    pth_sha256: str | None,
    evaluations: int = SMOKE_EVALUATIONS,
    seed: int = RESERVED_SMOKE_SEED,
) -> dict[str, object]:
    edges = bin_edges_from_records(
        table.records(),
        source_sha256=table.meta.source_sha256,
    )
    counts = occupancy_counts(table.records(), edges)
    max_frac = max_bin_fraction(counts, len(table))

    unique_hashes = {
        Nas201Spec.from_arch_str(row.arch).genotype_sha256() for row in table.records()
    }
    unique_indices = {row.index for row in table.records()}

    rng = np.random.default_rng(seed)
    sample_hits = 0
    for _ in range(LOOKUP_SAMPLE):
        spec = random_spec(rng)
        if table.lookup_search(spec.arch_str) is not None:
            sample_hits += 1

    full_hits = 0
    for ops in itertools.product(OPERATIONS, repeat=N_EDGES):
        spec = Nas201Spec(ops=ops)  # type: ignore[arg-type]
        if table.lookup_search(spec.arch_str) is not None:
            full_hits += 1

    spy = SpyLookup(table)
    invalid = evaluate_payload({"ops": ["not_an_op"] * N_EDGES}, spy, edges)
    garbage = evaluate_payload("|totally~broken|", spy, edges)
    miss = evaluate_spec(
        Nas201Spec.from_arch_str(next(iter(table.records())).arch),
        SpyLookup(_EmptyLookup()),
        edges,
    )

    floor = seeded_initial_archive(
        table, edges, seed=seed, n_random=SMOKE_INITIAL_RANDOM
    )
    random_result, _ = run_nas201_smoke(
        table,
        edges,
        generator="random",
        selector="uniform_frontier",
        seed=seed + 1,
        evaluations=evaluations,
        initial_archive=floor,
    )
    uniform_result, uniform_archive = run_nas201_smoke(
        table,
        edges,
        generator="genetic",
        selector="uniform_frontier",
        seed=seed + 2,
        evaluations=evaluations,
        initial_archive=floor,
    )
    minfit_result, minfit_archive = run_nas201_smoke(
        table,
        edges,
        generator="genetic",
        selector="min_fitness_frontier",
        seed=seed + 3,
        evaluations=evaluations,
        initial_archive=floor,
    )
    jaccard = niche_jaccard(
        uniform_archive.occupied_bins(),
        minfit_archive.occupied_bins(),
    )

    full_lookup = full_hits == EXPECTED_N and len(table) == EXPECTED_N
    unique_canonical_hash = (
        len(unique_hashes) == EXPECTED_N and len(unique_indices) == EXPECTED_N
    )
    search_split_only = (
        table.meta.search_dataset == "cifar10-valid"
        and table.meta.search_hp == "200"
        and table.meta.search_split == "x-valid"
        and table.meta.contains_test_metrics is False
    )
    no_bin_over_half = max_frac <= MAX_BIN_OCCUPANCY_FRACTION
    selector_jaccard_ok = jaccard < MAX_SELECTOR_JACCARD
    coverage_headroom = (
        uniform_result.coverage < MAX_SMOKE_COVERAGE
        and minfit_result.coverage < MAX_SMOKE_COVERAGE
    )

    return {
        "stage": "nas201_lookup_smoke",
        "llm": False,
        "reserved_seed": seed,
        "pth_sha256": pth_sha256,
        "compact_source_sha256": table.meta.source_sha256,
        "n_architectures": len(table),
        "fitness_scale": table.meta.fitness_scale,
        "search_dataset": table.meta.search_dataset,
        "search_hp": table.meta.search_hp,
        "search_split": table.meta.search_split,
        "contains_test_metrics": table.meta.contains_test_metrics,
        "bin_edges": {
            "resolution": edges.resolution,
            "log_params_min": edges.log_params_min,
            "log_params_max": edges.log_params_max,
            "log_flops_min": edges.log_flops_min,
            "log_flops_max": edges.log_flops_max,
        },
        "occupancy": {
            "max_bin_count": max(counts),
            "max_bin_fraction": max_frac,
            "empty_bins": counts.count(0),
            "occupied_bins": sum(1 for item in counts if item > 0),
        },
        "full_space_lookup_hits": full_hits,
        "random_200_lookup_hits": sample_hits,
        "unique_hashes": len(unique_hashes),
        "miss_policy": {
            "invalid_parse_calls_lookup": len(spy.calls),
            "invalid_structurally_valid": invalid.structurally_valid,
            "garbage_structurally_valid": garbage.structurally_valid,
            "valid_lookup_miss_reason": miss.miss_reason,
        },
        "random_smoke": _result_dict(random_result),
        "genetic_uniform": _result_dict(uniform_result),
        "genetic_min_fitness": _result_dict(minfit_result),
        "selector_niche_jaccard": jaccard,
        "gates": {
            "full_lookup": full_lookup,
            "unique_canonical_hash": unique_canonical_hash,
            "search_split_only": search_split_only,
            "no_bin_over_half": no_bin_over_half,
            "selector_jaccard": selector_jaccard_ok,
            "coverage_headroom": coverage_headroom,
        },
    }


class _EmptyLookup:
    def lookup_search(self, arch_str: str):
        return None

    def __len__(self) -> int:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="NAS-Bench-201 feasibility lookup smoke"
    )
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pth", type=Path, default=None)
    parser.add_argument("--evaluations", type=int, default=SMOKE_EVALUATIONS)
    args = parser.parse_args(argv)
    table = CompactNas201Table.from_jsonl(args.jsonl, args.meta)
    meta_extra = json.loads(args.meta.read_text(encoding="utf-8"))
    official_pth = meta_extra.get("official_nas201_pth_sha256")
    pth_sha256 = sha256_file(args.pth) if args.pth is not None else official_pth
    if (
        args.pth is not None
        and isinstance(official_pth, str)
        and pth_sha256 != official_pth
    ):
        print(
            "pth sha256 does not match compact meta official_nas201_pth_sha256",
            file=sys.stderr,
        )
        return 2
    report = run_nas201_lookup_smoke(
        table, pth_sha256=pth_sha256, evaluations=args.evaluations
    )
    report["lookup_backend"] = meta_extra.get("lookup_backend")
    report["nats_meta_sha256"] = table.meta.source_sha256
    report["official_nas201_pth_sha256"] = official_pth
    report["compact_jsonl_sha256"] = sha256_file(args.jsonl)
    report["valid_accuracy_min"] = meta_extra.get("valid_accuracy_min")
    report["valid_accuracy_max"] = meta_extra.get("valid_accuracy_max")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gates = report["gates"]
    if not isinstance(gates, dict):
        raise TypeError("NAS lookup-smoke report gates must be a dict")
    print(json.dumps(gates, indent=2))
    print(f"wrote {args.output}")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""NAS-Bench-201 random/genetic smoke on a synthetic complete lookup."""

from __future__ import annotations

import unittest

from worldspace.nas201.descriptors import Nas201BinEdges, bin_for_record, log10_positive
from worldspace.nas201.emitters import mutate_one_edge
from worldspace.nas201.smoke import (
    niche_jaccard,
    run_nas201_smoke,
    seeded_initial_archive,
)
from worldspace.nas201.spec import Nas201Spec, try_parse_arch_str
from worldspace.nas201.table import Nas201SearchRecord


class _SyntheticLookup:
    """Any structurally valid cell is a hit; costs come from the op tuple."""

    def lookup_search(self, arch_str: str) -> Nas201SearchRecord | None:
        spec = try_parse_arch_str(arch_str)
        if spec is None:
            return None
        weights = [1 + i for i in range(6)]
        op_index = {
            "none": 1,
            "skip_connect": 2,
            "nor_conv_1x1": 3,
            "nor_conv_3x3": 5,
            "avg_pool_3x3": 4,
        }
        score = sum(op_index[op] * weight for op, weight in zip(spec.ops, weights))
        params = 0.05 * score
        flops = 0.2 * score * score
        accuracy = 40.0 + 0.5 * score
        return Nas201SearchRecord(
            index=score,
            arch=spec.arch_str,
            flops=flops,
            params=params,
            latency=None,
            valid_accuracy=accuracy,
            n_trials=3,
        )

    def __len__(self) -> int:
        return 15625


def _edges() -> Nas201BinEdges:
    samples = []
    lookup = _SyntheticLookup()
    # Corners of the op space for min/max costs.
    for ops in (
        ("none",) * 6,
        ("nor_conv_3x3",) * 6,
        ("skip_connect",) * 6,
        ("avg_pool_3x3",) * 6,
        ("nor_conv_1x1",) * 6,
    ):
        record = lookup.lookup_search(Nas201Spec(ops=ops).arch_str)  # type: ignore[arg-type]
        assert record is not None
        samples.append(record)
    return Nas201BinEdges(
        resolution=20,
        log_params_min=min(log10_positive(row.params) for row in samples),
        log_params_max=max(log10_positive(row.params) for row in samples),
        log_flops_min=min(log10_positive(row.flops) for row in samples),
        log_flops_max=max(log10_positive(row.flops) for row in samples),
        n_architectures=15625,
        source_sha256="synthetic",
    )


class TestNas201Smoke(unittest.TestCase):
    def test_random_and_genetic_are_full_lookup_hits(self) -> None:
        table = _SyntheticLookup()
        edges = _edges()
        random_result, _ = run_nas201_smoke(
            table,
            edges,
            generator="random",
            selector="uniform_frontier",
            seed=3,
            evaluations=30,
            initial_random=5,
        )
        genetic_result, _ = run_nas201_smoke(
            table,
            edges,
            generator="genetic",
            selector="uniform_frontier",
            seed=4,
            evaluations=30,
            initial_random=5,
        )
        self.assertEqual(random_result.lookup_misses, 0)
        self.assertEqual(genetic_result.lookup_misses, 0)
        self.assertEqual(random_result.structurally_invalid, 0)
        self.assertEqual(genetic_result.evaluations, 35)
        self.assertLess(genetic_result.coverage, 0.95)

    def test_two_selectors_occupy_different_niches(self) -> None:
        table = _SyntheticLookup()
        edges = _edges()
        floor = seeded_initial_archive(table, edges, seed=9, n_random=20)
        _, uniform = run_nas201_smoke(
            table,
            edges,
            generator="genetic",
            selector="uniform_frontier",
            seed=10,
            evaluations=80,
            initial_archive=floor,
        )
        _, minfit = run_nas201_smoke(
            table,
            edges,
            generator="genetic",
            selector="min_fitness_frontier",
            seed=11,
            evaluations=80,
            initial_archive=floor,
        )
        jaccard = niche_jaccard(uniform.occupied_bins(), minfit.occupied_bins())
        self.assertLess(jaccard, 0.80)

    def test_bin_assignment_is_stable(self) -> None:
        table = _SyntheticLookup()
        edges = _edges()
        spec = Nas201Spec(ops=("nor_conv_3x3",) * 6)
        record = table.lookup_search(spec.arch_str)
        assert record is not None
        first = bin_for_record(record, edges)
        second = bin_for_record(record, edges)
        self.assertEqual(first, second)

    def test_genetic_child_stays_in_search_space(self) -> None:
        import numpy as np

        rng = np.random.default_rng(2)
        parent = Nas201Spec(ops=("none", "skip_connect", "nor_conv_1x1") * 2)
        child = mutate_one_edge(parent, rng)
        table = _SyntheticLookup()
        self.assertIsNotNone(table.lookup_search(child.arch_str))


if __name__ == "__main__":
    unittest.main()

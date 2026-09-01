"""NAS-Bench-201 lookup-miss policy and search-split isolation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldspace.nas201.descriptors import (
    bin_edges_from_records,
    max_bin_fraction,
    occupancy_counts,
)
from worldspace.nas201.evaluation import evaluate_payload, evaluate_spec
from worldspace.nas201.spec import Nas201Spec
from worldspace.nas201.table import (
    CompactNas201Table,
    CompactTableMeta,
    ForbiddenTestMetricError,
    Nas201SearchRecord,
    SpyLookup,
)


def _unique_ops(index: int) -> tuple[str, ...]:
    ops = ["none"] * 6
    ops[index] = "skip_connect"
    return tuple(ops)


def _record(
    index: int,
    ops: tuple[str, ...],
    *,
    params: float,
    flops: float,
    accuracy: float,
) -> Nas201SearchRecord:
    spec = Nas201Spec(ops=ops)  # type: ignore[arg-type]
    return Nas201SearchRecord(
        index=index,
        arch=spec.arch_str,
        flops=flops,
        params=params,
        latency=None,
        valid_accuracy=accuracy,
        n_trials=3,
    )


def _meta(n: int) -> CompactTableMeta:
    return CompactTableMeta(
        schema="nas201-search-v1",
        source_file="fixture.pth",
        source_sha256="a" * 64,
        n_architectures=n,
        search_dataset="cifar10-valid",
        search_hp="200",
        search_split="x-valid",
        fitness_scale="percent",
        contains_test_metrics=False,
    )


class TestNas201Lookup(unittest.TestCase):
    def test_invalid_payload_never_queries_table(self) -> None:
        record = _record(
            0,
            ("none",) * 6,
            params=1.0,
            flops=2.0,
            accuracy=50.0,
        )
        table = CompactNas201Table([record], _meta(1))
        edges = bin_edges_from_records(table.records(), source_sha256="a" * 64)
        spy = SpyLookup(table)
        result = evaluate_payload({"ops": ["nope"] * 6}, spy, edges)
        self.assertFalse(result.structurally_valid)
        self.assertFalse(result.lookup_hit)
        self.assertIsNone(result.fitness)
        self.assertEqual(spy.calls, [])

    def test_valid_unknown_arch_is_lookup_miss_without_fitness(self) -> None:
        present = _record(
            0,
            ("none",) * 6,
            params=1.0,
            flops=2.0,
            accuracy=50.0,
        )
        table = CompactNas201Table([present], _meta(1))
        edges = bin_edges_from_records(table.records(), source_sha256="a" * 64)
        missing = Nas201Spec(
            ops=(
                "skip_connect",
                "none",
                "none",
                "none",
                "none",
                "none",
            )
        )
        spy = SpyLookup(table)
        result = evaluate_spec(missing, spy, edges)
        self.assertTrue(result.structurally_valid)
        self.assertFalse(result.lookup_hit)
        self.assertIsNone(result.fitness)
        self.assertEqual(result.miss_reason, "lookup_miss")
        self.assertEqual(spy.calls, [missing.arch_str])

    def test_compact_loader_rejects_test_metric_keys(self) -> None:
        record = _record(
            0,
            ("none",) * 6,
            params=1.0,
            flops=2.0,
            accuracy=50.0,
        )
        table = CompactNas201Table([record], _meta(1))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl = root / "t.jsonl"
            meta = root / "t.json"
            table.write_jsonl(jsonl, meta)
            poisoned = json.loads(jsonl.read_text())
            # write_jsonl writes one object per line; poison the row.
            row = json.loads(jsonl.read_text().splitlines()[0])
            row["test_accuracy"] = 99.0
            jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ForbiddenTestMetricError):
                CompactNas201Table.from_jsonl(jsonl, meta)
            del poisoned

    def test_search_hit_uses_valid_accuracy_only(self) -> None:
        record = _record(
            0,
            ("nor_conv_3x3",) * 6,
            params=3.0,
            flops=12.0,
            accuracy=91.25,
        )
        table = CompactNas201Table([record], _meta(1))
        edges = bin_edges_from_records(table.records(), source_sha256="a" * 64)
        result = evaluate_spec(Nas201Spec.from_arch_str(record.arch), table, edges)
        self.assertTrue(result.lookup_hit)
        self.assertEqual(result.fitness, 91.25)
        self.assertIsNotNone(result.record)
        self.assertFalse(hasattr(result.record, "test_accuracy"))

    def test_occupancy_flags_collapsed_bins(self) -> None:
        collapsed = [
            _record(
                index,
                _unique_ops(index),
                params=2.0,
                flops=4.0,
                accuracy=40.0 + index,
            )
            for index in range(6)
        ]
        table = CompactNas201Table(collapsed, _meta(6))
        edges = bin_edges_from_records(table.records(), source_sha256="a" * 64)
        counts = occupancy_counts(table.records(), edges)
        self.assertGreater(max_bin_fraction(counts, 6), 0.50)

        spread = [
            _record(
                index,
                _unique_ops(index),
                params=10.0 ** (index / 5.0),
                flops=10.0 ** ((5 - index) / 5.0),
                accuracy=40.0 + index,
            )
            for index in range(6)
        ]
        spread_table = CompactNas201Table(spread, _meta(6))
        spread_edges = bin_edges_from_records(
            spread_table.records(), source_sha256="b" * 64
        )
        spread_counts = occupancy_counts(spread_table.records(), spread_edges)
        self.assertLessEqual(max_bin_fraction(spread_counts, 6), 0.50)


if __name__ == "__main__":
    unittest.main()

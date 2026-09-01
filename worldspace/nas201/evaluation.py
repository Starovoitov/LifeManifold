"""Search-only evaluation. Invalid genotypes never touch the lookup table."""

from __future__ import annotations

from dataclasses import dataclass

from worldspace.nas201.descriptors import (
    Nas201BinEdges,
    bin_for_record,
    measures_for_record,
)
from worldspace.nas201.spec import Nas201Spec, try_parse_arch_str, try_parse_ops_payload
from worldspace.nas201.table import Nas201Lookup, Nas201SearchRecord


@dataclass(frozen=True)
class Nas201Evaluation:
    spec: Nas201Spec | None
    structurally_valid: bool
    lookup_hit: bool
    fitness: float | None
    measures: tuple[float, float] | None
    bin: tuple[int, int] | None
    record: Nas201SearchRecord | None
    miss_reason: str | None


def evaluate_spec(
    spec: Nas201Spec,
    table: Nas201Lookup,
    edges: Nas201BinEdges,
) -> Nas201Evaluation:
    record = table.lookup_search(spec.arch_str)
    if record is None:
        return Nas201Evaluation(
            spec=spec,
            structurally_valid=True,
            lookup_hit=False,
            fitness=None,
            measures=None,
            bin=None,
            record=None,
            miss_reason="lookup_miss",
        )
    return Nas201Evaluation(
        spec=spec,
        structurally_valid=True,
        lookup_hit=True,
        fitness=record.valid_accuracy,
        measures=measures_for_record(record),
        bin=bin_for_record(record, edges),
        record=record,
        miss_reason=None,
    )


def evaluate_payload(
    payload: object,
    table: Nas201Lookup,
    edges: Nas201BinEdges,
) -> Nas201Evaluation:
    """Parse ops-JSON or arch string; do not query on parse failure."""
    spec = try_parse_ops_payload(payload)
    if spec is None and isinstance(payload, str):
        spec = try_parse_arch_str(payload)
    if spec is None:
        return Nas201Evaluation(
            spec=None,
            structurally_valid=False,
            lookup_hit=False,
            fitness=None,
            measures=None,
            bin=None,
            record=None,
            miss_reason="structurally_invalid",
        )
    return evaluate_spec(spec, table, edges)

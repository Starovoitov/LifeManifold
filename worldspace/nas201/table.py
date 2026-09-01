"""Search-only NAS-Bench-201 compact table (cifar10-valid / hp 200).

The compact rows MUST NOT contain test-set metrics. Hold-out CIFAR-10 test
readout is a separate function that lookup smoke does not call.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

SEARCH_DATASET = "cifar10-valid"
SEARCH_HP = "200"
SEARCH_SPLIT = "x-valid"
COMPACT_SCHEMA = "nas201-search-v1"


class ForbiddenTestMetricError(ValueError):
    """Raised when a compact row or query attempts to carry test metrics."""


@dataclass(frozen=True)
class Nas201SearchRecord:
    """One architecture's search-time metrics. No test accuracy."""

    index: int
    arch: str
    flops: float
    params: float
    latency: float | None
    valid_accuracy: float
    n_trials: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("index must be >= 0")
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if self.flops <= 0.0 or self.params <= 0.0:
            raise ValueError("flops and params must be positive")


@dataclass(frozen=True)
class CompactTableMeta:
    schema: str
    source_file: str
    source_sha256: str
    n_architectures: int
    search_dataset: str
    search_hp: str
    search_split: str
    fitness_scale: str
    contains_test_metrics: bool
    nas_bench_201_api_version: str | None = None

    def validate_search_contract(self) -> None:
        if self.schema != COMPACT_SCHEMA:
            raise ValueError(f"unsupported compact schema {self.schema!r}")
        if self.search_dataset != SEARCH_DATASET:
            raise ValueError("compact table search_dataset must be cifar10-valid")
        if self.search_hp != SEARCH_HP:
            raise ValueError("compact table search_hp must be 200")
        if self.search_split != SEARCH_SPLIT:
            raise ValueError("compact table search_split must be x-valid")
        if self.contains_test_metrics:
            raise ForbiddenTestMetricError(
                "compact search table must not contain test metrics"
            )


class Nas201Lookup(Protocol):
    """Lookup that is only allowed to return search-split records."""

    def lookup_search(self, arch_str: str) -> Nas201SearchRecord | None: ...

    def __len__(self) -> int: ...


_TEST_KEYS = frozenset(
    {
        "test_accuracy",
        "test-accuracy",
        "ori-test",
        "x-test",
        "test_acc",
        "cifar10_test",
    }
)


def _reject_test_keys(payload: Mapping[str, object], *, context: str) -> None:
    present = _TEST_KEYS.intersection(payload)
    if present:
        raise ForbiddenTestMetricError(
            f"{context} contains forbidden test-set keys {sorted(present)}"
        )


class CompactNas201Table:
    """In-memory search table keyed by official architecture string."""

    def __init__(
        self,
        records: Iterable[Nas201SearchRecord],
        meta: CompactTableMeta,
    ) -> None:
        meta.validate_search_contract()
        by_arch: dict[str, Nas201SearchRecord] = {}
        by_index: dict[int, Nas201SearchRecord] = {}
        for record in records:
            if record.arch in by_arch:
                raise ValueError(f"duplicate architecture string {record.arch!r}")
            if record.index in by_index:
                raise ValueError(f"duplicate architecture index {record.index}")
            by_arch[record.arch] = record
            by_index[record.index] = record
        if len(by_arch) != meta.n_architectures:
            raise ValueError(
                f"meta n_architectures={meta.n_architectures} "
                f"!= table size {len(by_arch)}"
            )
        self._meta = meta
        self._by_arch = by_arch
        self._by_index = by_index

    @property
    def meta(self) -> CompactTableMeta:
        return self._meta

    def __len__(self) -> int:
        return len(self._by_arch)

    def lookup_search(self, arch_str: str) -> Nas201SearchRecord | None:
        return self._by_arch.get(arch_str)

    def record_by_index(self, index: int) -> Nas201SearchRecord | None:
        return self._by_index.get(index)

    def records(self) -> Iterator[Nas201SearchRecord]:
        return iter(self._by_index[index] for index in sorted(self._by_index))

    @classmethod
    def from_jsonl(cls, jsonl_path: Path, meta_path: Path) -> CompactNas201Table:
        meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta_payload, dict):
            raise ValueError("compact meta must be a JSON object")
        _reject_test_keys(meta_payload, context="compact meta")
        meta = CompactTableMeta(
            schema=str(meta_payload["schema"]),
            source_file=str(meta_payload["source_file"]),
            source_sha256=str(meta_payload["source_sha256"]),
            n_architectures=int(meta_payload["n_architectures"]),
            search_dataset=str(meta_payload["search_dataset"]),
            search_hp=str(meta_payload["search_hp"]),
            search_split=str(meta_payload["search_split"]),
            fitness_scale=str(meta_payload["fitness_scale"]),
            contains_test_metrics=bool(meta_payload["contains_test_metrics"]),
            nas_bench_201_api_version=meta_payload.get("nas_bench_201_api_version"),
        )
        records: list[Nas201SearchRecord] = []
        with jsonl_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if not isinstance(payload, dict):
                    raise ValueError(f"compact row {line_number} must be an object")
                _reject_test_keys(payload, context=f"compact row {line_number}")
                records.append(
                    Nas201SearchRecord(
                        index=int(payload["index"]),
                        arch=str(payload["arch"]),
                        flops=float(payload["flops"]),
                        params=float(payload["params"]),
                        latency=(
                            None
                            if payload.get("latency") is None
                            else float(payload["latency"])
                        ),
                        valid_accuracy=float(payload["valid_accuracy"]),
                        n_trials=int(payload["n_trials"]),
                    )
                )
        return cls(records, meta)

    def write_jsonl(self, jsonl_path: Path, meta_path: Path) -> None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(
                {
                    "schema": self._meta.schema,
                    "source_file": self._meta.source_file,
                    "source_sha256": self._meta.source_sha256,
                    "n_architectures": self._meta.n_architectures,
                    "search_dataset": self._meta.search_dataset,
                    "search_hp": self._meta.search_hp,
                    "search_split": self._meta.search_split,
                    "fitness_scale": self._meta.fitness_scale,
                    "contains_test_metrics": False,
                    "nas_bench_201_api_version": self._meta.nas_bench_201_api_version,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in self.records():
                handle.write(
                    json.dumps(
                        {
                            "index": record.index,
                            "arch": record.arch,
                            "flops": record.flops,
                            "params": record.params,
                            "latency": record.latency,
                            "valid_accuracy": record.valid_accuracy,
                            "n_trials": record.n_trials,
                        },
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )


class SpyLookup:
    """Test double that records lookup_search calls."""

    def __init__(self, inner: Nas201Lookup) -> None:
        self.inner = inner
        self.calls: list[str] = []

    def lookup_search(self, arch_str: str) -> Nas201SearchRecord | None:
        self.calls.append(arch_str)
        return self.inner.lookup_search(arch_str)

    def __len__(self) -> int:
        return len(self.inner)

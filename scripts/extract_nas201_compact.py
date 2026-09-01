"""Extract a search-only compact table (cifar10-valid / hp 200, no test metrics).

Default backend is NATS-Bench topology search space (same 15,625 cells as
NAS-Bench-201). Do NOT torch.load NAS-Bench-201-v1_1-096897.pth in the same
session as Cursor: unpickling that 4.7G file peaked at ~15 GiB RSS and the
kernel OOM-killer took down the Python job (and starved the desktop).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import TypedDict

EXPECTED_N = 15625
SEARCH_DATASET = "cifar10-valid"
SEARCH_HP = "200"
SEARCH_SPLIT = "x-valid"
NAS201_PTH_ALLOW_ENV = "NAS201_ALLOW_FULL_PTH_LOAD"


class CompactSearchRow(TypedDict):
    index: int
    arch: str
    flops: float
    params: float
    latency: float | None
    valid_accuracy: float
    n_trials: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _write_compact(
    rows: list[CompactSearchRow],
    jsonl_path: Path,
    meta_path: Path,
    meta: dict[str, object],
) -> dict[str, object]:
    accuracies = [row["valid_accuracy"] for row in rows]
    meta["valid_accuracy_min"] = min(accuracies)
    meta["valid_accuracy_max"] = max(accuracies)
    meta["fitness_scale"] = "percent" if max(accuracies) > 1.5 else "unit_interval"
    meta["n_architectures"] = len(rows)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row, ensure_ascii=True, separators=(",", ":"), sort_keys=True
                )
                + "\n"
            )
    meta_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return meta


def extract_from_nats_tss(
    nats_path: Path, jsonl_path: Path, meta_path: Path
) -> dict[str, object]:
    """Load one architecture file at a time (fast_mode). Never pickle-load the 1.1G bundle."""
    from nats_bench.api_utils import PICKLE_EXT
    from nats_bench.api_utils import ArchResults
    from nats_bench.api_utils import pickle_load
    from nats_bench import create

    archive_dir = nats_path
    if nats_path.is_file() and nats_path.suffix == ".tar":
        dest = nats_path.with_suffix("")
        marker = dest / f"meta.{PICKLE_EXT}"
        if not marker.is_file():
            _log(f"extracting {nats_path} -> {dest.parent} ...")
            import tarfile

            with tarfile.open(nats_path) as archive:
                archive.extractall(dest.parent)
        archive_dir = dest
    if archive_dir.is_file() and str(archive_dir).endswith(".pickle.pbz2"):
        raise RuntimeError(
            "Refusing to pickle-load the bundled NATS .pickle.pbz2 (RSS climbed "
            "past 7 GiB). Pass the unpacked *-simple directory or .tar instead."
        )
    if not archive_dir.is_dir():
        raise RuntimeError(f"expected NATS simple directory, got {archive_dir}")

    meta_file = archive_dir / f"meta.{PICKLE_EXT}"
    if not meta_file.is_file():
        meta_file = archive_dir / "meta.pickle.pbz2"
    _log(f"sha256 {meta_file} + archive dir {archive_dir} ...")
    source_sha256 = sha256_file(meta_file)

    _log("create NATS TSS API fast_mode (meta only, no full table) ...")
    api = create(str(archive_dir), "tss", fast_mode=True, verbose=False)
    if len(api) != EXPECTED_N:
        raise RuntimeError(f"len(api)={len(api)}, expected {EXPECTED_N}")

    rows: list[CompactSearchRow] = []
    for index in range(len(api)):
        arch = api.meta_archs[index]
        xfile = archive_dir / f"{index:06d}.{PICKLE_EXT}"
        if not xfile.is_file():
            xfile = archive_dir / f"{index}.{PICKLE_EXT}"
        xdata = pickle_load(str(xfile))
        if SEARCH_HP not in xdata:
            raise RuntimeError(
                f"architecture {index} missing hp={SEARCH_HP}: {list(xdata)}"
            )
        info = ArchResults.create_from_state_dict(xdata[SEARCH_HP])
        del xdata
        metrics = info.get_metrics(
            SEARCH_DATASET, SEARCH_SPLIT, iepoch=None, is_random=False
        )
        costs = info.get_compute_costs(SEARCH_DATASET)
        n_trials = len(info.get_dataset_seeds(SEARCH_DATASET))
        rows.append(
            {
                "index": int(index),
                "arch": str(arch),
                "flops": float(costs["flops"]),
                "params": float(costs["params"]),
                "latency": (
                    None if costs.get("latency") is None else float(costs["latency"])
                ),
                "valid_accuracy": float(metrics["accuracy"]),
                "n_trials": int(n_trials),
            }
        )
        del info
        if (index + 1) % 2000 == 0:
            _log(f"  wrote {index + 1}/{EXPECTED_N}")
    try:
        from nats_bench import version as nats_version

        api_version = nats_version()
    except Exception:
        api_version = None
    meta = {
        "schema": "nas201-search-v1",
        "source_file": archive_dir.name,
        "source_sha256": source_sha256,
        "lookup_backend": "nats-tss-simple",
        "search_dataset": SEARCH_DATASET,
        "search_hp": SEARCH_HP,
        "search_split": SEARCH_SPLIT,
        "contains_test_metrics": False,
        "nas_bench_201_api_version": None,
        "nats_bench_api_version": api_version,
        "topology_isomorphic_to": "NAS-Bench-201 15,625-cell DAG",
        "official_nas201_pth_sha256": "2317a7dc911654322f4f932730a821cf38ecc5b5f1adde0aff26ff6b3f736666",
    }
    return _write_compact(rows, jsonl_path, meta_path, meta)


def extract_from_nas201_pth(
    pth: Path, jsonl_path: Path, meta_path: Path
) -> dict[str, object]:
    if os.environ.get(NAS201_PTH_ALLOW_ENV) != "1":
        raise RuntimeError(
            "Refusing to torch.load NAS-Bench-201-v1_1-096897.pth. "
            f"Set {NAS201_PTH_ALLOW_ENV}=1 only on a machine with spare RAM "
            "(unpickle peaked at ~15 GiB and OOM-killed this desktop). "
            "Use --backend nats-tss instead."
        )
    raise RuntimeError("nas201-pth backend is disabled in-process; use nats-tss")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract cifar10-valid / hp 200 search metrics. "
            "Default backend is NATS-Bench TSS (NAS-201 topology, small pickle)."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("nats-tss", "nas201-pth"),
        default="nats-tss",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not args.source.is_file() and not args.source.is_dir():
        print(f"missing source: {args.source}", file=sys.stderr)
        return 2
    if args.jsonl.exists() and not args.force:
        print(f"refusing to overwrite {args.jsonl} without --force", file=sys.stderr)
        return 2
    if args.backend == "nas201-pth":
        meta = extract_from_nas201_pth(args.source, args.jsonl, args.meta)
    else:
        meta = extract_from_nats_tss(args.source, args.jsonl, args.meta)
    print(json.dumps({k: meta[k] for k in sorted(meta)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

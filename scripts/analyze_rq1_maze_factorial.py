#!/usr/bin/env python3
"""Readout for maze RQ1 Phase B: stub/hints × minfit/uniform at 5k.

Descriptive only. Not Holm, not TOST, does not identify CA Holm Δ.
Protocol: artifacts/Q1_RQ1_SECOND_DOMAIN.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

ARMS = (
    "llm_stub_minfit",
    "llm_stub_uniform",
    "llm_hints_minfit",
    "llm_hints_uniform",
)
PRIMARY_PROPOSALS = 5000
EXPECTED_MODEL = "gpt-4o-mini-2024-07-18"


def _load_summaries(root: Path) -> dict[str, dict[int, dict]]:
    by_arm: dict[str, dict[int, dict]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for summary_path in sorted((root / arm).glob("seed_*/nightly_run_summary.json")):
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            seed = int(payload["seed"])
            by_arm[arm][seed] = payload
    return by_arm


def _cov_pct(payload: dict) -> float:
    return float(payload["coverage"]) * 100.0


def _fmt(values: list[float]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{mean(values):.2f} ± {pstdev(values):.2f}"


def _paired(left: dict[int, dict], right: dict[int, dict]) -> tuple[list[int], list[float]]:
    seeds = sorted(set(left) & set(right))
    return seeds, [_cov_pct(right[s]) - _cov_pct(left[s]) for s in seeds]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/experiments/q1-rq1-maze-factorial"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    by_arm = _load_summaries(root)
    print(f"root={root}")
    print(f"primary cut = terminal coverage @ {PRIMARY_PROPOSALS} proposals (empty archive)")
    print()
    for arm in ARMS:
        rows = by_arm[arm]
        covs = [_cov_pct(p) for p in rows.values()]
        models = {p.get("llm_model") for p in rows.values()}
        proposals = {p.get("proposals") for p in rows.values()}
        policies = {p.get("target_selection") for p in rows.values()}
        print(
            f"{arm:20s}  n={len(rows):2d}/10  "
            f"cov%={_fmt(covs):>14s}  "
            f"proposals={sorted(proposals) or '—'}  "
            f"policy={sorted(x for x in policies if x) or '—'}  "
            f"model={sorted(x for x in models if x) or '—'}"
        )
        unexpected = {m for m in models if m not in (None, EXPECTED_MODEL)}
        if unexpected:
            print(f"  WARNING unexpected llm_model={sorted(unexpected)}")
        bad_budget = {p.get("proposals") for p in rows.values()} - {PRIMARY_PROPOSALS}
        if rows and bad_budget:
            print(f"  WARNING unexpected proposals={sorted(bad_budget)}")

    print()
    print("Paired contrasts (complete seeds only)")
    contrasts = (
        ("policy @ stub", "llm_stub_minfit", "llm_stub_uniform"),
        ("policy @ live", "llm_hints_minfit", "llm_hints_uniform"),
        ("leftover @ minfit", "llm_stub_minfit", "llm_hints_minfit"),
        ("leftover @ uniform", "llm_stub_uniform", "llm_hints_uniform"),
    )
    for label, left, right in contrasts:
        seeds, deltas = _paired(by_arm[left], by_arm[right])
        if not seeds:
            print(f"{label:22s}  n=0")
            continue
        n_pos = sum(d > 0 for d in deltas)
        n_neg = sum(d < 0 for d in deltas)
        print(
            f"{label:22s}  n={len(seeds):2d}  "
            f"Δ={_fmt(deltas):>14s} pp  "
            f"+/−/0={n_pos}/{n_neg}/{len(seeds) - n_pos - n_neg}  "
            f"seeds={seeds}"
        )
    done = min(len(by_arm[arm]) for arm in ARMS)
    print()
    print(f"complete 2×2 seeds: {done}/10")
    if done < 10:
        raise SystemExit(0)


if __name__ == "__main__":
    main()

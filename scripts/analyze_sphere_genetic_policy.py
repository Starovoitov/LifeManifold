#!/usr/bin/env python3
"""Readout for Sphere RQ1 Phase A: genetic minfit vs uniform at 5k.

Descriptive only. Not Holm. Protocol: artifacts/Q1_RQ1_SPHERE_DOMAIN.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

ARMS = ("genetic", "genetic_minfit")
PRIMARY_PROPOSALS = 5000


def _load(root: Path) -> dict[str, dict[int, dict]]:
    by_arm: dict[str, dict[int, dict]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for path in sorted((root / arm).glob("seed_*/nightly_run_summary.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            by_arm[arm][int(payload["seed"])] = payload
    return by_arm


def _cov(payload: dict) -> float:
    return float(payload["coverage"]) * 100.0


def _fmt(values: list[float]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{mean(values):.2f} ± {pstdev(values):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/experiments/q1-sphere-genetic-policy"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    by_arm = _load(root)
    print(f"root={root}")
    print(f"primary cut = terminal coverage @ {PRIMARY_PROPOSALS} proposals")
    print()
    for arm in ARMS:
        rows = by_arm[arm]
        covs = [_cov(p) for p in rows.values()]
        print(
            f"{arm:18s}  n={len(rows):2d}/10  "
            f"cov%={_fmt(covs):>14s}  "
            f"policy={sorted({p.get('target_selection') for p in rows.values()})}"
        )
        bad = {p.get("proposals") for p in rows.values()} - {PRIMARY_PROPOSALS}
        if rows and bad:
            print(f"  WARNING unexpected proposals={sorted(bad)}")

    seeds = sorted(set(by_arm["genetic"]) & set(by_arm["genetic_minfit"]))
    deltas = [_cov(by_arm["genetic_minfit"][s]) - _cov(by_arm["genetic"][s]) for s in seeds]
    print()
    if not seeds:
        print("paired n=0")
        raise SystemExit(0)
    n_pos = sum(d > 0 for d in deltas)
    n_neg = sum(d < 0 for d in deltas)
    mean_delta = mean(deltas)
    print(
        f"minfit − uniform    n={len(seeds):2d}  "
        f"Δ={_fmt(deltas):>14s} pp  "
        f"+/−/0={n_pos}/{n_neg}/{len(seeds) - n_pos - n_neg}  "
        f"seeds={seeds}"
    )
    abs_mean = abs(mean_delta)
    same_sign = n_pos >= 8 or n_neg >= 8
    if abs_mean >= 5.0 or (same_sign and abs_mean >= 3.0):
        decision = "GO Phase B"
    elif abs_mean < 2.0:
        decision = "NO-GO (do not spend LLM)"
    else:
        decision = "Borderline (report; do not spend LLM yet)"
    print(f"decision: {decision}")
    if len(seeds) < 10:
        raise SystemExit(0)


if __name__ == "__main__":
    main()

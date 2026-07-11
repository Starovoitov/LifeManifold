#!/usr/bin/env python3
"""Live-proposal D1 replay: compose gate 0.5 vs 0.95 skip divergence.

Replays ``threshold_gate`` skip decisions on logged filter ``surrogate_archive.jsonl``
rows by recomposing fitness from predicted components at both extinction gates.

Primary metric: ``divergent_skip_fraction_at_min_fit_0.45``. Confirmatory iff ≤ 0.05.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.surrogate.types import SurrogatePrediction
from worldspace.surrogate.utils import compute_fitness_from_prediction

DEFAULT_FILTER_ROOT = ROOT / "artifacts/experiments/q1-full/filter"
OUT_DEFAULT = ROOT / "artifacts/surrogate/compose_gate_live_0p5_vs_0p95.json"
MIN_FIT = 0.45
MAX_UNC = 1.0
GATE_A = 0.5
GATE_B = 0.95
REASON_EMPTY = "empty_bin_explore"


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _prediction(record: dict[str, Any]) -> SurrogatePrediction:
    pred = record["prediction"]
    return SurrogatePrediction(
        components={k: float(v) for k, v in pred["components"].items()},
        measures={k: float(v) for k, v in pred["measures"].items()},
        fitness=float(pred["fitness"]),
        uncertainty=float(pred["uncertainty"]),
    )


def would_skip(
    fitness: float,
    uncertainty: float,
    *,
    min_predicted_fitness: float = MIN_FIT,
    max_uncertainty_to_skip: float = MAX_UNC,
    force_eval_empty: bool = False,
) -> bool:
    """Match live ``threshold_gate`` skip when empty-bin force-eval is known."""
    if force_eval_empty:
        return False
    return fitness < min_predicted_fitness and uncertainty <= max_uncertainty_to_skip


def _div_key(min_predicted_fitness: float) -> str:
    """Stable JSON key for the divergent-skip metric at a given threshold."""
    return f"divergent_skip_fraction_at_min_fit_{min_predicted_fitness:g}"


def replay_seed(
    path: Path,
    *,
    min_predicted_fitness: float = MIN_FIT,
    max_uncertainty_to_skip: float = MAX_UNC,
) -> dict[str, Any]:
    n = 0
    divergent = 0
    in_band = 0
    zero_flip = 0
    skip_a = 0
    skip_b = 0
    logged_skip = 0
    agree_logged_b = 0
    empty_logged = 0
    abs_diff = []
    emitter_div: dict[str, list[int]] = {}
    fitness_match_b = 0

    for record in _iter_jsonl(path):
        pred = _prediction(record)
        force_empty = record.get("decision_reason") == REASON_EMPTY
        if force_empty:
            empty_logged += 1

        fit_a = compute_fitness_from_prediction(pred, extinction_gate_threshold=GATE_A)
        fit_b = compute_fitness_from_prediction(pred, extinction_gate_threshold=GATE_B)
        if abs(fit_b - pred.fitness) <= 1e-9:
            fitness_match_b += 1

        s_a = would_skip(
            fit_a,
            pred.uncertainty,
            min_predicted_fitness=min_predicted_fitness,
            max_uncertainty_to_skip=max_uncertainty_to_skip,
            force_eval_empty=force_empty,
        )
        s_b = would_skip(
            fit_b,
            pred.uncertainty,
            min_predicted_fitness=min_predicted_fitness,
            max_uncertainty_to_skip=max_uncertainty_to_skip,
            force_eval_empty=force_empty,
        )
        diverged = s_a != s_b
        divergent += int(diverged)
        skip_a += int(s_a)
        skip_b += int(s_b)
        logged = record.get("decision") == "skip"
        logged_skip += int(logged)
        agree_logged_b += int(logged == s_b)

        p_ext = float(pred.components["early_extinction_prob"])
        in_band += int(GATE_A <= p_ext < GATE_B)
        zero_flip += int((fit_a <= 0.0) != (fit_b <= 0.0))
        abs_diff.append(abs(fit_b - fit_a))

        emitter = str(record.get("emitter_type") or "unknown")
        bucket = emitter_div.setdefault(emitter, [0, 0])
        bucket[0] += 1
        bucket[1] += int(diverged)
        n += 1

    if n == 0:
        raise ValueError(f"empty archive: {path}")

    div = divergent / n
    out: dict[str, Any] = {
        "path": str(path),
        "n": n,
        "min_predicted_fitness": min_predicted_fitness,
        "divergent_skip_fraction": div,
        _div_key(min_predicted_fitness): div,
        "frac_pred_ext_p_in_[0.5,0.95)": in_band / n,
        "frac_pred_zero_flips": zero_flip / n,
        "mean_abs_pred_diff": float(np.mean(abs_diff)),
        "median_abs_pred_diff": float(np.median(abs_diff)),
        "skip_rate_gate_0.5": skip_a / n,
        "skip_rate_gate_0.95": skip_b / n,
        "logged_skip_rate": logged_skip / n,
        "agree_logged_skip_vs_gate_0.95": agree_logged_b / n,
        "logged_empty_bin_explore_frac": empty_logged / n,
        "logged_fitness_matches_recompose_0.95": fitness_match_b / n,
        "by_emitter": {
            name: {
                "n": counts[0],
                "divergent_skip_fraction": counts[1] / counts[0],
            }
            for name, counts in sorted(emitter_div.items())
        },
    }
    # Backward-compatible alias for earlier B.2 consumers.
    if abs(min_predicted_fitness - 0.45) < 1e-12:
        out["divergent_skip_fraction_at_min_fit_0.45"] = div
    return out


def discover_seed_archives(filter_root: Path) -> list[Path]:
    paths = sorted(filter_root.glob("seed_*/surrogate_archive.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"no surrogate_archive.jsonl under {filter_root}/seed_*/"
        )
    return paths


def aggregate(
    seed_stats: list[dict[str, Any]],
    *,
    min_predicted_fitness: float,
) -> dict[str, Any]:
    divs = np.asarray(
        [s["divergent_skip_fraction"] for s in seed_stats],
        dtype=float,
    )
    n_total = int(sum(s["n"] for s in seed_stats))
    # pooled (proposal-weighted) divergent fraction
    div_count = sum(s["divergent_skip_fraction"] * s["n"] for s in seed_stats)
    pooled = div_count / n_total
    band = (
        sum(s["frac_pred_ext_p_in_[0.5,0.95)"] * s["n"] for s in seed_stats) / n_total
    )
    mean_abs = sum(s["mean_abs_pred_diff"] * s["n"] for s in seed_stats) / n_total
    agree = (
        sum(s["agree_logged_skip_vs_gate_0.95"] * s["n"] for s in seed_stats) / n_total
    )
    empty = (
        sum(s["logged_empty_bin_explore_frac"] * s["n"] for s in seed_stats) / n_total
    )
    fit_match = (
        sum(s["logged_fitness_matches_recompose_0.95"] * s["n"] for s in seed_stats)
        / n_total
    )
    out: dict[str, Any] = {
        "n_seeds": len(seed_stats),
        "n_proposals": n_total,
        "min_predicted_fitness": min_predicted_fitness,
        "divergent_skip_fraction": pooled,
        _div_key(min_predicted_fitness): pooled,
        "divergent_skip_mean_across_seeds": float(np.mean(divs)),
        "divergent_skip_min_across_seeds": float(np.min(divs)),
        "divergent_skip_max_across_seeds": float(np.max(divs)),
        "frac_pred_ext_p_in_[0.5,0.95)": band,
        "mean_abs_pred_diff": mean_abs,
        "agree_logged_skip_vs_gate_0.95": agree,
        "logged_empty_bin_explore_frac": empty,
        "logged_fitness_matches_recompose_0.95": fit_match,
        "RQ3_confirmatory_rule_div_le_0.05": pooled <= 0.05,
    }
    if abs(min_predicted_fitness - 0.45) < 1e-12:
        out["divergent_skip_fraction_at_min_fit_0.45"] = pooled
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filter-root",
        type=Path,
        default=DEFAULT_FILTER_ROOT,
        help="Directory containing seed_*/surrogate_archive.jsonl",
    )
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--min-fit", type=float, default=MIN_FIT)
    parser.add_argument("--max-unc", type=float, default=MAX_UNC)
    args = parser.parse_args()

    archives = discover_seed_archives(args.filter_root)
    print(f"Replaying {len(archives)} archives under {args.filter_root}", flush=True)
    seed_stats: list[dict[str, Any]] = []
    for path in archives:
        seed = path.parent.name
        stats = replay_seed(
            path,
            min_predicted_fitness=args.min_fit,
            max_uncertainty_to_skip=args.max_unc,
        )
        stats["seed"] = seed
        seed_stats.append(stats)
        print(
            f"  {seed}: n={stats['n']} div={stats['divergent_skip_fraction']:.4f} "
            f"(min_fit={args.min_fit:g}) "
            f"agree95={stats['agree_logged_skip_vs_gate_0.95']:.4f}",
            flush=True,
        )

    gate_sensitivity = aggregate(seed_stats, min_predicted_fitness=args.min_fit)
    payload = {
        "source": "live_proposal_replay",
        "filter_root": str(args.filter_root),
        "filter_min_predicted_fitness": args.min_fit,
        "max_uncertainty_to_skip": args.max_unc,
        "gates": [GATE_A, GATE_B],
        "policy_note": (
            "Skip iff fitness < min_fit and uncertainty <= max_unc; "
            "force-eval when logged decision_reason == empty_bin_explore. "
            "Q1 filter combat threshold is min_predicted_fitness=0.45 "
            "(raised from 0.10 after shadow); agree_logged≈1 only at that threshold."
        ),
        "gate_sensitivity": gate_sensitivity,
        "per_seed": seed_stats,
        "RQ3_confirmatory_rule_div_le_0.05": gate_sensitivity[
            "RQ3_confirmatory_rule_div_le_0.05"
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate_sensitivity, indent=2), flush=True)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

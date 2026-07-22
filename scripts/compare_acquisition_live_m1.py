#!/usr/bin/env python3
"""M1 Phase 3: live filter archive counterfactual — threshold_gate vs ucb_promote.

Replays acquisition policies on logged ``q1-full/filter/*/surrogate_archive.jsonl``
predictions (no new sims, no GP re-featurization — archives lack feature vectors).

Merges ``live_archive_replay`` into ``artifacts/surrogate/gp_ucb_ablation.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.illuminators.archive_factory import (  # noqa: E402
    ArchiveFactoryConfig,
    create_archive,
)
from worldspace.illuminators.scheduler import TargetBin  # noqa: E402
from worldspace.surrogate.acquisition import decide  # noqa: E402
from worldspace.surrogate.acquisition_config import AcquisitionConfig  # noqa: E402
from worldspace.surrogate.types import SurrogatePrediction  # noqa: E402

DEFAULT_FILTER_ROOT = ROOT / "artifacts/experiments/q1-full/filter"
DEFAULT_OUT = ROOT / "artifacts/surrogate/gp_ucb_ablation.json"
UCB_BETAS = (0.15, 0.5, 1.0)
SHADOW_SKIP_BAND = (0.25, 0.45)
FALSE_SKIP_MARGIN = 0.0  # match production min_fit gate for live proxy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter-root", type=Path, default=DEFAULT_FILTER_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-predicted-fitness", type=float, default=0.45)
    parser.add_argument("--max-uncertainty-to-skip", type=float, default=1.0)
    parser.add_argument(
        "--ucb-betas",
        type=str,
        default=",".join(str(b) for b in UCB_BETAS),
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=0,
        help="If >0, only the first N seed archives (sorted).",
    )
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def discover_seed_archives(filter_root: Path) -> list[Path]:
    paths = sorted(filter_root.glob("seed_*/surrogate_archive.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"no surrogate_archive.jsonl under {filter_root}/seed_*/"
        )
    return paths


def _prediction(record: dict[str, Any]) -> SurrogatePrediction:
    pred = record["prediction"]
    return SurrogatePrediction(
        components={k: float(v) for k, v in pred.get("components", {}).items()},
        measures={k: float(v) for k, v in pred.get("measures", {}).items()},
        fitness=float(pred["fitness"]),
        uncertainty=float(pred["uncertainty"]),
    )


def _target(archive, cell_id: int) -> TargetBin:
    return TargetBin(
        bin=archive.bin_from_cell_id(int(cell_id)),
        target_stability=0.5,
        target_diversity=0.5,
    )


def _in_shadow_band(skip_rate: float) -> bool:
    lo, hi = SHADOW_SKIP_BAND
    return lo <= float(skip_rate) <= hi


def _policy_stats(
    *,
    name: str,
    skip_count: int,
    n: int,
    agree_logged: int,
    flip_eval_to_skip: int,
    flip_skip_to_eval: int,
    false_skip_count: int,
    exploration_weight: float | None = None,
) -> dict[str, Any]:
    skip_rate = float(skip_count) / float(n) if n else float("nan")
    false_denom = flip_eval_to_skip  # only rows where we observe sim fitness
    false_rate = (
        float(false_skip_count) / float(false_denom) if false_denom > 0 else 0.0
    )
    return {
        "policy": name,
        "exploration_weight": exploration_weight,
        "n": n,
        "skip_count": skip_count,
        "skip_rate": skip_rate,
        "agree_logged_rate": float(agree_logged) / float(n) if n else float("nan"),
        "flip_eval_to_skip": flip_eval_to_skip,
        "flip_skip_to_eval": flip_skip_to_eval,
        "false_skip_count_on_eval_to_skip": false_skip_count,
        "false_skip_rate_on_eval_to_skip": false_rate,
        "in_shadow_skip_band_25_45": _in_shadow_band(skip_rate),
    }


def replay_seed(
    path: Path,
    policies: list[AcquisitionConfig],
    *,
    grid_resolution: int = 50,
) -> dict[str, Any]:
    """Replay each policy on one archive; empty-bin latch off (matches Q1 filter logs)."""
    archive = create_archive(
        ArchiveFactoryConfig(archive_type="grid", resolution=grid_resolution)
    )
    # Accumulators per policy index.
    skip_c = [0] * len(policies)
    agree_c = [0] * len(policies)
    flip_e2s = [0] * len(policies)
    flip_s2e = [0] * len(policies)
    false_c = [0] * len(policies)
    reason_counts: list[dict[str, int]] = [dict() for _ in policies]
    logged_skip = 0
    empty_logged = 0
    n = 0
    min_fit = float(policies[0].min_predicted_fitness)

    for record in _iter_jsonl(path):
        pred = _prediction(record)
        cell_id = int(record.get("target_cell_id", n % archive.n_cells))
        target = _target(archive, cell_id)
        logged_is_skip = record.get("decision") == "skip"
        logged_skip += int(logged_is_skip)
        if record.get("decision_reason") == "empty_bin_explore":
            empty_logged += 1
        eval_outcome = record.get("eval_outcome")
        actual_fit = (
            float(eval_outcome["fitness"])
            if isinstance(eval_outcome, dict) and "fitness" in eval_outcome
            else None
        )

        for i, policy in enumerate(policies):
            decision = decide(policy, pred, target, archive)
            is_skip = decision.action == "skip"
            skip_c[i] += int(is_skip)
            agree_c[i] += int(is_skip == logged_is_skip)
            reason_counts[i][decision.reason] = (
                reason_counts[i].get(decision.reason, 0) + 1
            )
            if (not logged_is_skip) and is_skip:
                flip_e2s[i] += 1
                if actual_fit is not None and actual_fit >= min_fit + FALSE_SKIP_MARGIN:
                    false_c[i] += 1
            if logged_is_skip and (not is_skip):
                flip_s2e[i] += 1
        n += 1

    if n == 0:
        raise ValueError(f"empty archive: {path}")

    policy_rows = []
    for i, policy in enumerate(policies):
        row = _policy_stats(
            name=policy.policy,
            skip_count=skip_c[i],
            n=n,
            agree_logged=agree_c[i],
            flip_eval_to_skip=flip_e2s[i],
            flip_skip_to_eval=flip_s2e[i],
            false_skip_count=false_c[i],
            exploration_weight=(
                float(policy.exploration_weight)
                if policy.policy == "ucb_promote"
                else None
            ),
        )
        row["decision_reasons"] = reason_counts[i]
        policy_rows.append(row)

    return {
        "path": str(path),
        "n": n,
        "logged_skip_count": logged_skip,
        "logged_skip_rate": float(logged_skip) / float(n),
        "logged_empty_bin_explore_frac": float(empty_logged) / float(n),
        "policies": policy_rows,
    }


def aggregate_seeds(seed_stats: list[dict[str, Any]]) -> dict[str, Any]:
    n_total = int(sum(s["n"] for s in seed_stats))
    logged_skip = int(sum(s["logged_skip_count"] for s in seed_stats))
    empty = sum(s["logged_empty_bin_explore_frac"] * s["n"] for s in seed_stats)
    n_policies = len(seed_stats[0]["policies"])
    pooled_rows: list[dict[str, Any]] = []
    for i in range(n_policies):
        skip = sum(s["policies"][i]["skip_count"] for s in seed_stats)
        agree = sum(s["policies"][i]["agree_logged_rate"] * s["n"] for s in seed_stats)
        e2s = sum(s["policies"][i]["flip_eval_to_skip"] for s in seed_stats)
        s2e = sum(s["policies"][i]["flip_skip_to_eval"] for s in seed_stats)
        false = sum(
            s["policies"][i]["false_skip_count_on_eval_to_skip"] for s in seed_stats
        )
        proto = seed_stats[0]["policies"][i]
        row = _policy_stats(
            name=proto["policy"],
            skip_count=skip,
            n=n_total,
            agree_logged=int(round(agree)),
            flip_eval_to_skip=e2s,
            flip_skip_to_eval=s2e,
            false_skip_count=false,
            exploration_weight=proto["exploration_weight"],
        )
        # Prefer exact agree count reconstruction.
        agree_exact = sum(
            int(round(s["policies"][i]["agree_logged_rate"] * s["n"]))
            for s in seed_stats
        )
        row["agree_logged_rate"] = float(agree_exact) / float(n_total)
        pooled_rows.append(row)

    # Per-seed skip rates for threshold / each UCB β.
    per_seed_skip: dict[str, list[float]] = {}
    for s in seed_stats:
        for pol in s["policies"]:
            key = pol["policy"]
            if pol["exploration_weight"] is not None:
                key = f"{pol['policy']}_beta_{pol['exploration_weight']}"
            per_seed_skip.setdefault(key, []).append(float(pol["skip_rate"]))
    per_seed_skip["logged"] = [float(s["logged_skip_rate"]) for s in seed_stats]

    return {
        "n_seeds": len(seed_stats),
        "n_proposals": n_total,
        "logged_skip_rate": float(logged_skip) / float(n_total),
        "logged_empty_bin_explore_frac": float(empty) / float(n_total),
        "policies": pooled_rows,
        "per_seed_skip_rates": {
            k: {
                "mean": float(np.mean(vs)),
                "min": float(np.min(vs)),
                "max": float(np.max(vs)),
                "values": vs,
            }
            for k, vs in per_seed_skip.items()
        },
    }


def build_policies(
    *,
    min_predicted_fitness: float,
    max_uncertainty_to_skip: float,
    betas: list[float],
) -> list[AcquisitionConfig]:
    base = AcquisitionConfig(
        mode="filter",
        policy="threshold_gate",
        min_predicted_fitness=min_predicted_fitness,
        max_uncertainty_to_skip=max_uncertainty_to_skip,
        never_skip_empty_bin=False,
    )
    policies = [base]
    for beta in betas:
        policies.append(
            replace(
                base,
                policy="ucb_promote",
                exploration_weight=float(beta),
            )
        )
    return policies


def main() -> None:
    args = parse_args()
    betas = [float(x.strip()) for x in args.ucb_betas.split(",") if x.strip()]
    policies = build_policies(
        min_predicted_fitness=float(args.min_predicted_fitness),
        max_uncertainty_to_skip=float(args.max_uncertainty_to_skip),
        betas=betas,
    )
    archives = discover_seed_archives(args.filter_root)
    if args.max_seeds > 0:
        archives = archives[: int(args.max_seeds)]

    print(
        f"M1 Phase 3: replaying {len(archives)} archives under {args.filter_root}",
        flush=True,
    )
    seed_stats: list[dict[str, Any]] = []
    for path in archives:
        seed = path.parent.name
        stats = replay_seed(path, policies)
        stats["seed"] = seed
        seed_stats.append(stats)
        tg = stats["policies"][0]
        print(
            f"  {seed}: n={stats['n']} logged_skip={stats['logged_skip_rate']:.3f} "
            f"tg_agree={tg['agree_logged_rate']:.4f} "
            + " ".join(
                f"ucbβ{p['exploration_weight']}={p['skip_rate']:.3f}"
                for p in stats["policies"]
                if p["policy"] == "ucb_promote"
            ),
            flush=True,
        )

    pooled = aggregate_seeds(seed_stats)
    tg = next(p for p in pooled["policies"] if p["policy"] == "threshold_gate")
    ucb_rows = [p for p in pooled["policies"] if p["policy"] == "ucb_promote"]
    verdict_notes = [
        (
            f"Logged filter skip_rate={pooled['logged_skip_rate']:.3f} "
            f"(n={pooled['n_proposals']}, {pooled['n_seeds']} seeds); "
            f"empty_bin_explore frac={pooled['logged_empty_bin_explore_frac']:.4f}."
        ),
        (
            f"threshold_gate replay agree_logged={tg['agree_logged_rate']:.4f} "
            f"(sanity; never_skip_empty_bin=false matches these logs)."
        ),
        (
            "ucb_promote on logged MLP μ/σ skip rates: "
            + ", ".join(
                f"β={r['exploration_weight']}→{r['skip_rate']:.3f}"
                f"{' (in-band)' if r['in_shadow_skip_band_25_45'] else ''}"
                for r in ucb_rows
            )
            + "."
        ),
        (
            "UCB only softens filtering on live MLP logs (flip skip→eval; "
            "eval→skip=0): higher β → more evals. No GP re-predict — archives "
            "have no feature vectors."
        ),
    ]

    payload_block: dict[str, Any] = {
        "phase": 3,
        "source": "live_filter_archive_counterfactual",
        "filter_root": str(args.filter_root.resolve()),
        "min_predicted_fitness": float(args.min_predicted_fitness),
        "max_uncertainty_to_skip": float(args.max_uncertainty_to_skip),
        "never_skip_empty_bin_replay": False,
        "shadow_skip_band": list(SHADOW_SKIP_BAND),
        "note": (
            "Counterfactual policies use logged MLP fitness/uncertainty only. "
            "GP+UCB not replayable here (no features/world_spec per proposal)."
        ),
        "pooled": pooled,
        "per_seed": seed_stats,
        "verdict_notes": verdict_notes,
    }

    out_path = args.output_json
    payload: dict[str, Any]
    if out_path.is_file():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        payload = {"family": "M1", "title": "MC-dropout MLP vs GP+UCB"}
    payload["phase"] = 3
    payload["live_archive_replay"] = payload_block
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pooled": pooled, "verdict_notes": verdict_notes}, indent=2))
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

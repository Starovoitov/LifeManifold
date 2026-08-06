#!/usr/bin/env python3
"""Offline oracle / live-policy replay on genetic_me_shadow proposal streams.

Counterfactual upper bound: skip when true_fitness < τ (empty-bin force-eval),
reconstruct archive coverage vs kept evaluations. Also replays logged would-skip
decisions on the same stream. See artifacts/Q1_H2_ORACLE_REPLAY.md.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.illuminators.archive import (
    ArchiveElite,
    GridArchive,
    load_and_collapse_jsonl,
    new_elite_metadata,
)

SHADOW_ROOT = ROOT / "artifacts/experiments/q1-h2-ranking-controls/genetic_me_shadow"
FILTER_ROOT = ROOT / "artifacts/experiments/q1-v3-genetic-me-filter/genetic_me_filter"
UNIFORM_ROOT = (
    ROOT / "artifacts/experiments/q1-v3-genetic-me-uniform/genetic_me_uniform"
)
RANDOM_SKIP_ROOT = (
    ROOT / "artifacts/experiments/q1-h2-ranking-controls/genetic_me_random_skip"
)
BASELINE = ROOT / "artifacts/map_elites_nightly/baseline/map_elites_archive.jsonl"
OUT = ROOT / "artifacts/experiments/h2-oracle-replay"
TAU = 0.45
RESOLUTION = 50
# Oracle keeps ~61% of 32.5k ≈ 19.9k; 20k is not always reached. Primary
# matched-eval horizon is 19k (all seeds). 20k reported when available.
EVAL_BUDGETS = (5000, 10000, 15000, 19000, 20000)
PRIMARY_HORIZON = 19000
HORIZON = 20000


def _finite(x: Any) -> bool:
    """True for real numbers; False for None / NaN / non-numeric."""
    if x is None:
        return False
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _mean_sd(xs: list[float]) -> dict[str, float | int]:
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def _clone_archive(src: GridArchive) -> GridArchive:
    dst = GridArchive(resolution=src.resolution)
    for cell_id in range(src.n_cells):
        elite = src.get_cell(cell_id)
        if elite is not None:
            dst.try_insert(copy.deepcopy(elite))
    return dst


def _coverage_pct(archive: GridArchive) -> float:
    return 100.0 * archive.filled_count() / float(archive.n_cells)


def _auc(xs: list[float], ys: list[float], horizon: int) -> float | None:
    if len(xs) < 2:
        return None
    order = np.argsort(xs)
    x = np.asarray(xs, dtype=float)[order]
    y = np.asarray(ys, dtype=float)[order]
    if x[0] > 0:
        x = np.concatenate([[0.0], x])
        y = np.concatenate([[y[0]], y])
    if x[-1] < horizon:
        x = np.concatenate([x, [float(horizon)]])
        y = np.concatenate([y, [y[-1]]])
    mask = x <= horizon
    x = x[mask]
    y = y[mask]
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trapz(y, x) / float(horizon))


def _cov_at_trace(trace: Path, budget: int) -> float | None:
    if not trace.is_file():
        return None
    best = None
    with trace.open() as fh:
        for line in fh:
            row = json.loads(line)
            ev = int(row.get("evaluations", -1))
            if 0 <= ev <= budget:
                cov = float(row["coverage"])
                best = cov * 100.0 if cov <= 1.5 else cov
    return best


def _auc_trace(trace: Path, horizon: int) -> float | None:
    if not trace.is_file():
        return None
    xs: list[float] = []
    ys: list[float] = []
    with trace.open() as fh:
        for line in fh:
            row = json.loads(line)
            ev = int(row.get("evaluations", -1))
            if ev < 0 or ev > horizon:
                continue
            cov = float(row["coverage"])
            if cov > 1.5:
                cov /= 100.0
            xs.append(float(ev))
            ys.append(cov)
    auc = _auc(xs, ys, horizon)
    return None if auc is None else auc * 100.0


def _elite_from_row(row: dict[str, Any]) -> ArchiveElite:
    eo = row["eval_outcome"]
    bin_ij = tuple(int(x) for x in row["target_bin"])
    assert len(bin_ij) == 2
    return ArchiveElite(
        bin=(bin_ij[0], bin_ij[1]),
        fitness=float(eo["fitness"]),
        world_spec=None,
        measures={
            "stability": float(eo["measures"]["stability"]),
            "diversity": float(eo["measures"]["diversity"]),
        },
        metadata=new_elite_metadata(
            generated_by="oracle_replay",
            emitter_type=str(row.get("emitter_type") or "unknown"),
        ),
    )


def _replay_seed(
    warm: GridArchive,
    archive_path: Path,
    *,
    mode: str,
) -> dict[str, Any]:
    """mode: 'oracle' (true fit < τ) or 'live' (logged decision==skip)."""
    archive = _clone_archive(warm)
    n = 0
    n_skip = 0
    n_force_empty = 0
    n_insert = 0
    evals = 0
    xs: list[float] = [0.0]
    ys: list[float] = [_coverage_pct(archive)]
    checkpoints: dict[int, float] = {}

    with archive_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            n += 1
            cell_id = int(row["target_cell_id"])
            true_fit = float(row["eval_outcome"]["fitness"])
            empty = archive.is_empty_cell(cell_id)

            if mode == "oracle":
                want_skip = (not empty) and (true_fit < TAU)
            else:
                # Live would-skip, but never skip empty in reconstructed archive
                want_skip = (row.get("decision") == "skip") and (not empty)

            if want_skip:
                n_skip += 1
                continue
            if empty and mode == "oracle" and true_fit < TAU:
                n_force_empty += 1

            evals += 1
            result = archive.try_insert(_elite_from_row(row))
            if result.accepted:
                n_insert += 1
            cov = _coverage_pct(archive)
            xs.append(float(evals))
            ys.append(cov)
            for b in EVAL_BUDGETS:
                if evals >= b and b not in checkpoints:
                    checkpoints[b] = cov

    # Hold last cov for any missed checkpoints if evals stopped early
    final_cov = ys[-1]
    for b in EVAL_BUDGETS:
        checkpoints.setdefault(b, final_cov if evals >= b else float("nan"))

    auc_primary = _auc(xs, [y / 100.0 for y in ys], PRIMARY_HORIZON)
    auc_20k = _auc(xs, [y / 100.0 for y in ys], HORIZON)
    return {
        "n_proposals": n,
        "n_skip": n_skip,
        "skip_rate_pct": 100.0 * n_skip / n if n else float("nan"),
        "n_force_empty_kept": n_force_empty,
        "n_evals": evals,
        "n_insert": n_insert,
        "terminal_coverage_pct": final_cov,
        "cov_at": {str(b): checkpoints[b] for b in EVAL_BUDGETS},
        "auc_cov_19k_pct": None if auc_primary is None else auc_primary * 100.0,
        "auc_cov_20k_pct": None if auc_20k is None else auc_20k * 100.0,
    }


def replay_seed(seed: int, warm: GridArchive) -> dict[str, Any]:
    arch = SHADOW_ROOT / f"seed_{seed}" / "surrogate_archive.jsonl"
    oracle = _replay_seed(warm, arch, mode="oracle")
    live = _replay_seed(warm, arch, mode="live")
    online = {
        "filter_cov_at_19000": _cov_at_trace(
            FILTER_ROOT / f"seed_{seed}" / "archive_trace.jsonl", PRIMARY_HORIZON
        ),
        "filter_cov_at_20000": _cov_at_trace(
            FILTER_ROOT / f"seed_{seed}" / "archive_trace.jsonl", 20000
        ),
        "filter_auc_19k": _auc_trace(
            FILTER_ROOT / f"seed_{seed}" / "archive_trace.jsonl", PRIMARY_HORIZON
        ),
        "filter_auc_20k": _auc_trace(
            FILTER_ROOT / f"seed_{seed}" / "archive_trace.jsonl", HORIZON
        ),
        "uniform_cov_at_19000": _cov_at_trace(
            UNIFORM_ROOT / f"seed_{seed}" / "archive_trace.jsonl", PRIMARY_HORIZON
        ),
        "uniform_cov_at_20000": _cov_at_trace(
            UNIFORM_ROOT / f"seed_{seed}" / "archive_trace.jsonl", 20000
        ),
        "random_skip_cov_at_19000": _cov_at_trace(
            RANDOM_SKIP_ROOT / f"seed_{seed}" / "archive_trace.jsonl", PRIMARY_HORIZON
        ),
        "random_skip_cov_at_20000": _cov_at_trace(
            RANDOM_SKIP_ROOT / f"seed_{seed}" / "archive_trace.jsonl", 20000
        ),
    }
    return {
        "seed": seed,
        "oracle": oracle,
        "live_policy_replay": live,
        "online": online,
    }


def main() -> int:
    if not BASELINE.is_file():
        raise SystemExit(f"missing warm-start baseline: {BASELINE}")
    warm_base = load_and_collapse_jsonl(
        BASELINE, archive_type="grid", resolution=RESOLUTION
    )
    if not isinstance(warm_base, GridArchive):
        raise SystemExit("baseline must load as GridArchive")

    seeds = sorted(
        int(p.name.split("_")[1])
        for p in SHADOW_ROOT.glob("seed_*")
        if p.is_dir() and (p / "surrogate_archive.jsonl").is_file()
    )
    if not seeds:
        raise SystemExit(f"no shadow seeds under {SHADOW_ROOT}")

    per_seed = [replay_seed(s, warm_base) for s in seeds]

    def collect(path: list[str]) -> list[float]:
        out: list[float] = []
        for row in per_seed:
            cur: Any = row
            for key in path:
                cur = cur[key]
            if cur is None or (isinstance(cur, float) and np.isnan(cur)):
                continue
            out.append(float(cur))
        return out

    summary = {
        "primary_horizon": PRIMARY_HORIZON,
        "oracle_n_evals": _mean_sd(collect(["oracle", "n_evals"])),
        "oracle_skip_rate_pct": _mean_sd(collect(["oracle", "skip_rate_pct"])),
        "oracle_cov_at_19000": _mean_sd(collect(["oracle", "cov_at", "19000"])),
        "oracle_auc_19k": _mean_sd(collect(["oracle", "auc_cov_19k_pct"])),
        "oracle_terminal_coverage_pct": _mean_sd(
            collect(["oracle", "terminal_coverage_pct"])
        ),
        "live_replay_skip_rate_pct": _mean_sd(
            collect(["live_policy_replay", "skip_rate_pct"])
        ),
        "live_replay_cov_at_19000": _mean_sd(
            collect(["live_policy_replay", "cov_at", "19000"])
        ),
        "live_replay_auc_19k": _mean_sd(
            collect(["live_policy_replay", "auc_cov_19k_pct"])
        ),
        "online_filter_cov_at_19000": _mean_sd(
            collect(["online", "filter_cov_at_19000"])
        ),
        "online_filter_auc_19k": _mean_sd(collect(["online", "filter_auc_19k"])),
        "online_random_skip_cov_at_19000": _mean_sd(
            collect(["online", "random_skip_cov_at_19000"])
        ),
        "online_uniform_cov_at_19000": _mean_sd(
            collect(["online", "uniform_cov_at_19000"])
        ),
    }

    # Same-stream gap: oracle − live would-skip replay @ 19k
    gap_live_cov = [
        float(r["oracle"]["cov_at"]["19000"])
        - float(r["live_policy_replay"]["cov_at"]["19000"])
        for r in per_seed
        if _finite(r["oracle"]["cov_at"]["19000"])
        and _finite(r["live_policy_replay"]["cov_at"]["19000"])
    ]
    gap_live_auc = [
        float(r["oracle"]["auc_cov_19k_pct"])
        - float(r["live_policy_replay"]["auc_cov_19k_pct"])
        for r in per_seed
        if _finite(r["oracle"]["auc_cov_19k_pct"])
        and _finite(r["live_policy_replay"]["auc_cov_19k_pct"])
    ]
    # Cross-stream descriptive: oracle@19k − online filter@19k (not an upper bound
    # on online filter — proposal streams differ).
    gap_online_cov = [
        float(r["oracle"]["cov_at"]["19000"])
        - float(r["online"]["filter_cov_at_19000"])
        for r in per_seed
        if _finite(r["oracle"]["cov_at"]["19000"])
        and _finite(r["online"]["filter_cov_at_19000"])
    ]
    summary["oracle_minus_live_replay_cov19k"] = dict(_mean_sd(gap_live_cov))
    summary["oracle_minus_live_replay_auc19k"] = dict(_mean_sd(gap_live_auc))
    summary["oracle_minus_live_replay_cov19k"]["n_positive"] = int(
        sum(1 for d in gap_live_cov if d > 0)
    )
    summary["oracle_minus_live_replay_auc19k"]["n_positive"] = int(
        sum(1 for d in gap_live_auc if d > 0)
    )
    summary["oracle_minus_online_filter_cov19k"] = dict(_mean_sd(gap_online_cov))
    summary["oracle_minus_online_filter_cov19k"]["n_positive"] = int(
        sum(1 for d in gap_online_cov if d > 0)
    )

    payload = {
        "tier": "h2-oracle-replay",
        "tau": TAU,
        "seeds": seeds,
        "n": len(seeds),
        "warm_start_filled": warm_base.filled_count(),
        "caveat": (
            "Proposal stream from shadow (full-eval parents); "
            "counterfactual upper bound, not online oracle policy."
        ),
        "summary": summary,
        "per_seed": per_seed,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "h2_oracle_replay_analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    def fmt(d: dict[str, float | int]) -> str:
        return f"{d['mean']:.2f}±{d['sd']:.2f}"

    lines = [
        "# H2 oracle replay analysis",
        "",
        f"**Status:** complete · seeds={seeds} · τ={TAU} · primary horizon={PRIMARY_HORIZON}",
        "",
        f"**Warm-start:** {warm_base.filled_count()} elites from baseline JSONL.",
        "",
        f"**Caveat:** {payload['caveat']}",
        "",
        f"Oracle keeps {fmt(summary['oracle_n_evals'])} evals/seed "
        f"(skip {fmt(summary['oracle_skip_rate_pct'])}%) — often <20k, hence horizon 19k.",
        "",
        "## Replay levels (same shadow stream)",
        "",
        "| Policy | Skip % | Cov@19k % | AUC cov@19k % | Terminal % |",
        "|--------|-------:|----------:|--------------:|-----------:|",
        f"| **Oracle** (true fit < τ) | {fmt(summary['oracle_skip_rate_pct'])} | "
        f"{fmt(summary['oracle_cov_at_19000'])} | {fmt(summary['oracle_auc_19k'])} | "
        f"{fmt(summary['oracle_terminal_coverage_pct'])} |",
        f"| Live would-skip replay | {fmt(summary['live_replay_skip_rate_pct'])} | "
        f"{fmt(summary['live_replay_cov_at_19000'])} | "
        f"{fmt(summary['live_replay_auc_19k'])} | — |",
        "",
        "## Online arms @19k real evals (archive_trace; different streams)",
        "",
        "| Arm | Cov@19k % |",
        "|-----|----------:|",
        f"| `genetic_me_filter` | {fmt(summary['online_filter_cov_at_19000'])} |",
        f"| `genetic_me_random_skip` | {fmt(summary['online_random_skip_cov_at_19000'])} |",
        f"| `genetic_me_uniform` | {fmt(summary['online_uniform_cov_at_19000'])} |",
        "",
        "## Gaps",
        "",
        "**Same-stream (fair ranking bound):** oracle − live would-skip replay",
        "",
        f"- Δ cov@19k: "
        f"{summary['oracle_minus_live_replay_cov19k']['mean']:+.2f}±"
        f"{summary['oracle_minus_live_replay_cov19k']['sd']:.2f} pp "
        f"({summary['oracle_minus_live_replay_cov19k']['n_positive']}/"
        f"{summary['oracle_minus_live_replay_cov19k']['n']})",
        f"- Δ AUC@19k: "
        f"{summary['oracle_minus_live_replay_auc19k']['mean']:+.2f}±"
        f"{summary['oracle_minus_live_replay_auc19k']['sd']:.2f} pp "
        f"({summary['oracle_minus_live_replay_auc19k']['n_positive']}/"
        f"{summary['oracle_minus_live_replay_auc19k']['n']})",
        "",
        "**Cross-stream (descriptive only):** oracle@19k − online filter@19k "
        f"= {summary['oracle_minus_online_filter_cov19k']['mean']:+.2f}±"
        f"{summary['oracle_minus_online_filter_cov19k']['sd']:.2f} pp "
        f"({summary['oracle_minus_online_filter_cov19k']['n_positive']}/"
        f"{summary['oracle_minus_online_filter_cov19k']['n']}). "
        "Online filter can exceed stream-oracle because skips change the proposal stream.",
        "",
        "## Reading notes",
        "",
        "- Primary fair contrast: oracle vs live would-skip on the **same** shadow stream.",
        "- Small same-stream gap ⇒ live MLP gate is near absolute-τ oracle on these candidates.",
        "- Online filter > stream-oracle is expected (extra proposals / different parents).",
        "",
        "Protocol: `artifacts/Q1_H2_ORACLE_REPLAY.md`",
        "Script: `scripts/analyze_h2_oracle_replay.py`",
        "",
    ]
    (OUT / "ANALYSIS.md").write_text("\n".join(lines))
    print((OUT / "ANALYSIS.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

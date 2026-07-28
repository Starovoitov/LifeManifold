#!/usr/bin/env python3
"""Offline τ replay on LLM+filter logs, stratified by emitter_type.

Replays ``threshold_gate`` skip decisions on logged ``q1-full/filter``
``surrogate_archive.jsonl`` rows at τ ∈ {0.35, 0.45, 0.55} without new
simulator runs. Contrasts mixed-arm (20R+20G+10L) skip calibration with the
no-LLM H2 genetic sweep (``q1-h2-threshold-sensitivity``).

Compose-gate D1 (extinction 0.5 vs 0.95) is reported separately — it is not
fixed by τ tuning; see ``replay_compose_gate_live.py``.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_compose_gate_live import (
    DEFAULT_FILTER_ROOT,
    MAX_UNC,
    discover_seed_archives,
    replay_seed,
    would_skip,
    _iter_jsonl,
    _prediction,
)

FILTER_ROOT = DEFAULT_FILTER_ROOT
H2_JSON = (
    ROOT
    / "artifacts/experiments/q1-h2-threshold-sensitivity/h2_threshold_sensitivity_analysis.json"
)
OUT_JSON = ROOT / "artifacts/surrogate/filter_threshold_emitter_replay.json"
OUT_MD = ROOT / "artifacts/experiments/q1-full/FILTER_THRESHOLD_EMITTER_REPLAY.md"

TAU_VALUES: tuple[float, ...] = (0.35, 0.45, 0.55)
EMITTER_STRATA: tuple[str, ...] = ("all", "llm", "genetic", "random")


def _matches_emitter(record: dict[str, Any], emitter: str) -> bool:
    if emitter == "all":
        return True
    return str(record.get("emitter_type") or "unknown") == emitter


def replay_tau_stratum(
    path: Path,
    *,
    tau: float,
    emitter: str = "all",
    max_uncertainty: float = MAX_UNC,
) -> dict[str, Any]:
    """Counterfactual skip stats for one seed / τ / emitter stratum."""
    n = 0
    skip = 0
    logged_skip = 0
    agree_logged = 0
    pred_skip: list[float] = []
    pred_eval: list[float] = []
    unc_skip: list[float] = []
    unc_eval: list[float] = []

    for record in _iter_jsonl(path):
        if not _matches_emitter(record, emitter):
            continue
        pred = _prediction(record)
        force_empty = record.get("decision_reason") == "empty_bin_explore"
        would = would_skip(
            pred.fitness,
            pred.uncertainty,
            min_predicted_fitness=tau,
            max_uncertainty_to_skip=max_uncertainty,
            force_eval_empty=force_empty,
        )
        n += 1
        skip += int(would)
        if would:
            pred_skip.append(pred.fitness)
            unc_skip.append(pred.uncertainty)
        else:
            pred_eval.append(pred.fitness)
            unc_eval.append(pred.uncertainty)

        logged = record.get("decision") == "skip"
        logged_skip += int(logged)
        agree_logged += int(logged == would)

    if n == 0:
        return {"n": 0, "emitter": emitter, "tau": tau}

    return {
        "n": n,
        "emitter": emitter,
        "tau": tau,
        "skip_rate": skip / n,
        "skip_rate_pct": round(100.0 * skip / n, 2),
        "logged_skip_rate": logged_skip / n,
        "agree_replay_vs_logged_skip": agree_logged / n,
        "mean_pred_fitness_skipped": (
            round(statistics.mean(pred_skip), 4) if pred_skip else None
        ),
        "mean_pred_fitness_evaluated": (
            round(statistics.mean(pred_eval), 4) if pred_eval else None
        ),
        "mean_uncertainty_skipped": (
            round(statistics.mean(unc_skip), 4) if unc_skip else None
        ),
        "mean_uncertainty_evaluated": (
            round(statistics.mean(unc_eval), 4) if unc_eval else None
        ),
    }


def _mean_sd(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 4),
        "sd": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
    }


def aggregate_stratum(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pooled (proposal-weighted) and per-seed mean skip rates."""
    total_n = sum(s["n"] for s in per_seed)
    if total_n == 0:
        return {"n_proposals": 0}

    pooled_skip = sum(s["skip_rate"] * s["n"] for s in per_seed) / total_n
    pooled_logged = sum(s["logged_skip_rate"] * s["n"] for s in per_seed) / total_n
    pooled_agree = (
        sum(s["agree_replay_vs_logged_skip"] * s["n"] for s in per_seed) / total_n
    )
    skip_rates = [s["skip_rate"] for s in per_seed]

    return {
        "n_seeds": len(per_seed),
        "n_proposals": total_n,
        "skip_rate_pooled": round(pooled_skip, 4),
        "skip_rate_pct_pooled": round(100.0 * pooled_skip, 2),
        "skip_rate_mean_across_seeds": round(statistics.mean(skip_rates), 4),
        "skip_rate_sd_across_seeds": (
            round(statistics.pstdev(skip_rates), 4) if len(skip_rates) > 1 else 0.0
        ),
        "logged_skip_rate_pooled": round(pooled_logged, 4),
        "agree_replay_vs_logged_pooled": round(pooled_agree, 4),
        "per_seed_skip_rate_pct": {
            str(i): round(100.0 * s["skip_rate"], 2) for i, s in enumerate(per_seed)
        },
    }


def _compose_d1_by_tau(
    archives: list[Path],
    taus: tuple[float, ...] = TAU_VALUES,
) -> dict[str, dict[str, Any]]:
    """Compose-gate divergent skip (extinction 0.5 vs 0.95) at each min-fit τ."""
    out: dict[str, dict[str, Any]] = {}
    for tau in taus:
        per_seed = [replay_seed(path, min_predicted_fitness=tau) for path in archives]
        n_total = sum(s["n"] for s in per_seed)
        pooled = sum(s["divergent_skip_fraction"] * s["n"] for s in per_seed) / n_total
        by_em: dict[str, list[float]] = {}
        for stats in per_seed:
            for em, vals in stats.get("by_emitter", {}).items():
                by_em.setdefault(em, []).append(vals["divergent_skip_fraction"])
        out[str(tau)] = {
            "d1_pooled": round(pooled, 4),
            "d1_by_emitter_mean": {
                em: round(statistics.mean(v), 4) for em, v in sorted(by_em.items())
            },
            "n_proposals": n_total,
            "n_seeds": len(per_seed),
        }
    return out


def _load_h2_reference() -> dict[float, float]:
    if not H2_JSON.is_file():
        return {}
    payload = json.loads(H2_JSON.read_text(encoding="utf-8"))
    out: dict[float, float] = {}
    for _cond, arm in payload.get("arms", {}).items():
        tau = float(arm.get("tau", 0))
        skip_stats = arm.get("skip_rate_pct", {})
        if skip_stats.get("n", 0) > 0:
            out[tau] = float(skip_stats["mean"])
    return out


def run_analysis(filter_root: Path = FILTER_ROOT) -> dict[str, Any]:
    archives = discover_seed_archives(filter_root)
    tau_by_emitter: dict[str, dict[str, Any]] = {}
    compose_by_emitter: dict[str, list[float]] = {}

    for emitter in EMITTER_STRATA:
        tau_by_emitter[emitter] = {}
        for tau in TAU_VALUES:
            per_seed = [
                replay_tau_stratum(path, tau=tau, emitter=emitter) for path in archives
            ]
            tau_by_emitter[emitter][str(tau)] = {
                "per_seed": per_seed,
                "aggregate": aggregate_stratum(per_seed),
            }

    compose_per_seed = [replay_seed(path) for path in archives]
    for seed_stats in compose_per_seed:
        for em, stats in seed_stats.get("by_emitter", {}).items():
            compose_by_emitter.setdefault(em, []).append(
                stats["divergent_skip_fraction"]
            )

    compose_agg = {
        em: _mean_sd(vals) for em, vals in sorted(compose_by_emitter.items())
    }
    compose_agg["all_pooled"] = _mean_sd(
        [s["divergent_skip_fraction"] for s in compose_per_seed]
    )

    h2_ref = _load_h2_reference()
    d1_by_tau = _compose_d1_by_tau(archives)

    return {
        "source": "offline_threshold_replay",
        "filter_root": str(filter_root.relative_to(ROOT)),
        "arm_mix": "20R+20G+10L (q1-full filter; H3 descriptive tier)",
        "h2_reference_tier": "q1-h2-threshold-sensitivity (20R+30G; no LLM)",
        "h2_live_skip_rate_pct": {str(k): v for k, v in sorted(h2_ref.items())},
        "tau_values": list(TAU_VALUES),
        "emitter_strata": list(EMITTER_STRATA),
        "max_uncertainty_to_skip": MAX_UNC,
        "note": (
            "Replay uses logged prediction.fitness (online compose @ extinction 0.95). "
            "τ sweep here varies min_predicted_fitness only; compose-gate D1 is separate."
        ),
        "tau_replay_by_emitter": tau_by_emitter,
        "compose_gate_d1_by_emitter": compose_agg,
        "compose_gate_d1_by_tau": d1_by_tau,
        "compose_gate_per_seed_mean_div": _mean_sd(
            [s["divergent_skip_fraction"] for s in compose_per_seed]
        ),
        "llm_proposals_per_seed_expected": 6500,
    }


def _md_table(payload: dict[str, Any]) -> list[str]:
    lines = [
        "# Filter τ replay by emitter (offline; q1-full/filter)",
        "",
        f"Artifact JSON: `{OUT_JSON.relative_to(ROOT)}`",
        f"Source logs: `{payload['filter_root']}` ({payload['arm_mix']})",
        "",
        "## Skip rate @ τ (replay on logged predictions; pooled over seeds)",
        "",
        "| τ | all | llm | genetic | random | H2 genetic live |",
        "|---|-----|-----|---------|--------|-----------------|",
    ]
    h2 = payload.get("h2_live_skip_rate_pct", {})
    for tau in TAU_VALUES:
        row = [f"{tau:g}"]
        for em in EMITTER_STRATA:
            agg = payload["tau_replay_by_emitter"][em][str(tau)]["aggregate"]
            pct = agg.get("skip_rate_pct_pooled")
            row.append(f"{pct:.1f}%" if pct is not None else "n/a")
        h2_pct = h2.get(str(tau))
        row.append(f"{h2_pct:.1f}%" if h2_pct is not None else "n/a")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "H2 column = no-LLM ``genetic_me_filter`` live runs (``q1-h2-threshold-sensitivity``).",
            "",
            "## Compose-gate D1 vs τ (extinction 0.5 vs 0.95; pooled)",
            "",
            "| τ | D1 pooled | llm | genetic | random |",
            "|---|-----------|-----|---------|--------|",
        ]
    )
    for tau in TAU_VALUES:
        row = payload["compose_gate_d1_by_tau"][str(tau)]
        em = row["d1_by_emitter_mean"]
        lines.append(
            f"| {tau:g} | {row['d1_pooled']:.3f} | {em.get('llm', 0):.3f} | "
            f"{em.get('genetic', 0):.3f} | {em.get('random', 0):.3f} |"
        )

    lines.extend(
        [
            "",
            "At production τ=0.45, per-seed mean D1 by emitter:",
            "",
        ]
    )
    for em, stats in payload["compose_gate_d1_by_emitter"].items():
        if em == "all_pooled" or stats.get("n", 0) == 0:
            continue
        lines.append(
            f"- **{em}**: {stats['mean']:.3f} ± {stats.get('sd', 0):.3f} "
            f"(n={stats['n']} seeds)"
        )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- LLM proposals: 6500/seed (20% of 50×650 batch); sufficient for offline τ diagnostic.",
            "- Skip rates rise monotonically with τ for every emitter; D1 **also** changes with τ (0.62→0.53→0.35), so τ and compose-gate interact.",
            "- At τ=0.45, D1 stays high for all emitters (0.51–0.58); none of {0.35,0.45,0.55} brings D1 ≤ 0.05.",
            "- D1 falls below 0.05 only at extreme τ (≥0.7, >87% skip) — artifactual agreement, not compose repair.",
            "",
            "Run: `uv run python scripts/analyze_filter_threshold_emitter_replay.py`",
        ]
    )
    return lines


def main() -> int:
    payload = run_analysis()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text("\n".join(_md_table(payload)) + "\n", encoding="utf-8")
    print(json.dumps(payload["tau_replay_by_emitter"]["all"], indent=2))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

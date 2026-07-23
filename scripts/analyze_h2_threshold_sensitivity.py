"""Analyze H2 threshold sweep: genetic_me_filter @ τ vs genetic_me_uniform (matched policy).

Tier: ``q1-h2-threshold-sensitivity`` (τ ∈ {0.35, 0.45, 0.55}).
Reference uniform arm: ``q1-v3-genetic-me-uniform`` (no re-run).
τ=0.45 may be reused from ``q1-v3-genetic-me-filter`` when absent in sweep root.

Primary endpoints (descriptive):
  - terminal coverage / fitness / evaluations / skip rate
  - anytime coverage @ 15k and 20k simulator evaluations (archive_trace)
  - paired Δ(filter − uniform) per τ
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SWEEP_ROOT = ROOT / "artifacts/experiments/q1-h2-threshold-sensitivity"
UNIFORM_ROOT = ROOT / "artifacts/experiments/q1-v3-genetic-me-uniform"
TAU045_LEGACY_ROOT = ROOT / "artifacts/experiments/q1-v3-genetic-me-filter"
OUT_JSON = SWEEP_ROOT / "h2_threshold_sensitivity_analysis.json"
OUT_MD = SWEEP_ROOT / "ANALYSIS.md"

TAU_ARMS: tuple[tuple[str, float, str], ...] = (
    ("genetic_me_filter_tau035", 0.35, "genetic_me_filter_tau035"),
    ("genetic_me_filter_tau045", 0.45, "genetic_me_filter"),
    ("genetic_me_filter_tau055", 0.55, "genetic_me_filter_tau055"),
)

ANYTIME_EVALS = (15_000, 20_000)


def _load_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _coverage_pct(payload: dict) -> float:
    cov = float(payload["coverage"])
    return cov * 100.0 if cov <= 1.0 else cov


def _fitness(payload: dict) -> float:
    val = payload.get("mean_best_fitness")
    return float("nan") if val is None else float(val)


def _evaluations(payload: dict) -> int:
    return int(payload.get("evaluations") or 0)


def _skip_rate_pct(payload: dict, run_dir: Path) -> float | None:
    val = payload.get("skip_rate_pct")
    if val is not None:
        return float(val)
    val = payload.get("skip_rate")
    if val is not None:
        f = float(val)
        return f * 100.0 if f <= 1.0 else f
    archive = run_dir / "surrogate_archive.jsonl"
    if not archive.is_file():
        return None
    skip = 0
    total = 0
    for line in archive.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        decision = row.get("decision")
        if isinstance(decision, dict):
            action = decision.get("action")
        else:
            action = decision
        if action == "skip":
            skip += 1
    return 100.0 * skip / total if total else None


def _coverage_at_eval(trace_path: Path, target_eval: int) -> float | None:
    if not trace_path.is_file():
        return None
    by_eval: dict[int, float] = {}
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("coverage") is None:
            continue
        by_eval[int(row["evaluations"])] = float(row["coverage"])
    if not by_eval:
        return None
    xs = sorted(by_eval)
    if target_eval <= xs[0]:
        cov = by_eval[xs[0]]
    elif target_eval >= xs[-1]:
        cov = by_eval[xs[-1]]
    else:
        lo = max(x for x in xs if x <= target_eval)
        hi = min(x for x in xs if x >= target_eval)
        if lo == hi:
            cov = by_eval[lo]
        else:
            t = (target_eval - lo) / (hi - lo)
            cov = by_eval[lo] * (1 - t) + by_eval[hi] * t
    return cov * 100.0 if cov <= 1.0 else cov


def _arm_stats(values: dict[int, float]) -> dict:
    if not values:
        return {"n": 0}
    vals = list(values.values())
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 4),
        "sd": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


def _paired(rows: list[tuple[int, float, float]]) -> dict:
    if not rows:
        return {"n": 0}
    deltas = [b - a for _, a, b in rows]
    n = len(deltas)
    return {
        "n": n,
        "mean_delta": round(statistics.mean(deltas), 4),
        "sd": round(statistics.pstdev(deltas), 4) if n > 1 else 0.0,
        "positive_seeds": sum(1 for d in deltas if d > 0),
        "seeds": [
            {
                "seed": s,
                "uniform": round(a, 4),
                "filter": round(b, 4),
                "delta": round(b - a, 4),
            }
            for s, a, b in rows
        ],
    }


def _filter_run_dir(condition: str, legacy_cond: str, seed: int) -> Path | None:
    sweep = SWEEP_ROOT / condition / f"seed_{seed}"
    if (sweep / "nightly_run_summary.json").is_file():
        return sweep
    if condition.endswith("_tau045"):
        legacy = TAU045_LEGACY_ROOT / legacy_cond / f"seed_{seed}"
        if (legacy / "nightly_run_summary.json").is_file():
            return legacy
    return None


def main() -> int:
    seeds = list(range(10))
    uniform_cov: dict[int, float] = {}
    uniform_anytime: dict[int, dict[int, float]] = {e: {} for e in ANYTIME_EVALS}

    for seed in seeds:
        summary = _load_summary(
            UNIFORM_ROOT
            / "genetic_me_uniform"
            / f"seed_{seed}"
            / "nightly_run_summary.json"
        )
        if summary is None:
            continue
        uniform_cov[seed] = _coverage_pct(summary)
        trace = (
            UNIFORM_ROOT / "genetic_me_uniform" / f"seed_{seed}" / "archive_trace.jsonl"
        )
        for ev in ANYTIME_EVALS:
            val = _coverage_at_eval(trace, ev)
            if val is not None:
                uniform_anytime[ev][seed] = val

    arm_rows: dict[str, dict] = {}
    contrasts: dict[str, dict] = {}
    anytime_contrasts: dict[str, dict[int, dict]] = {}

    for condition, tau, legacy_cond in TAU_ARMS:
        cov: dict[int, float] = {}
        fit: dict[int, float] = {}
        evals: dict[int, int] = {}
        skip: dict[int, float] = {}
        anytime: dict[int, dict[int, float]] = {e: {} for e in ANYTIME_EVALS}

        for seed in seeds:
            run_dir = _filter_run_dir(condition, legacy_cond, seed)
            if run_dir is None:
                continue
            summary = _load_summary(run_dir / "nightly_run_summary.json")
            if summary is None:
                continue
            cov[seed] = _coverage_pct(summary)
            fit[seed] = _fitness(summary)
            evals[seed] = _evaluations(summary)
            sr = _skip_rate_pct(summary, run_dir)
            if sr is not None:
                skip[seed] = sr
            trace = run_dir / "archive_trace.jsonl"
            for ev in ANYTIME_EVALS:
                val = _coverage_at_eval(trace, ev)
                if val is not None:
                    anytime[ev][seed] = val

        arm_rows[condition] = {
            "tau": tau,
            "coverage_pct": _arm_stats(cov),
            "fitness": _arm_stats({k: v for k, v in fit.items() if v == v}),
            "evaluations": _arm_stats({k: float(v) for k, v in evals.items()}),
            "skip_rate_pct": _arm_stats(skip),
            "anytime_coverage_pct": {
                str(ev): _arm_stats(anytime[ev]) for ev in ANYTIME_EVALS
            },
            "seeds_available": sorted(cov.keys()),
        }

        contrasts[f"{condition}_minus_uniform_terminal_cov"] = _paired(
            [
                (s, uniform_cov[s], cov[s])
                for s in seeds
                if s in uniform_cov and s in cov
            ]
        )
        anytime_contrasts[condition] = {}
        for ev in ANYTIME_EVALS:
            anytime_contrasts[condition][ev] = _paired(
                [
                    (s, uniform_anytime[ev][s], anytime[ev][s])
                    for s in seeds
                    if s in uniform_anytime[ev] and s in anytime[ev]
                ]
            )

    payload = {
        "tier": "q1-h2-threshold-sensitivity",
        "tau_values": [tau for _, tau, _ in TAU_ARMS],
        "uniform_reference": "q1-v3-genetic-me-uniform/genetic_me_uniform",
        "tau045_fallback": "q1-v3-genetic-me-filter/genetic_me_filter",
        "uniform_terminal_coverage_pct": _arm_stats(uniform_cov),
        "arms": arm_rows,
        "contrasts": contrasts,
        "anytime_contrasts_filter_minus_uniform": anytime_contrasts,
        "note": (
            "Descriptive H2 robustness: matched genetic_me_uniform vs filter at three "
            "min_predicted_fitness values; same checkpoint and uniform_frontier."
        ),
    }

    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# H2 threshold sensitivity (genetic_me_filter @ τ)",
        "",
        f"Artifact: `{SWEEP_ROOT.relative_to(ROOT)}`",
        f"Uniform reference: `{UNIFORM_ROOT.relative_to(ROOT)}/genetic_me_uniform`",
        "",
        "## Terminal coverage (%)",
        "",
        f"- **genetic_me_uniform** ($n={payload['uniform_terminal_coverage_pct'].get('n', 0)}$): "
        f"{payload['uniform_terminal_coverage_pct'].get('mean', float('nan')):.2f} "
        f"± {payload['uniform_terminal_coverage_pct'].get('sd', 0):.2f}",
        "",
    ]
    for condition, tau, _ in TAU_ARMS:
        stats = arm_rows.get(condition, {}).get("coverage_pct", {"n": 0})
        skip_stats = arm_rows.get(condition, {}).get("skip_rate_pct", {"n": 0})
        if stats.get("n", 0) == 0:
            lines.append(f"- **{condition}** (τ={tau:g}): no data")
            continue
        skip_txt = ""
        if skip_stats.get("n", 0):
            skip_txt = f"; skip {skip_stats['mean']:.1f}%"
        lines.append(
            f"- **{condition}** (τ={tau:g}, $n={stats['n']}$): "
            f"{stats['mean']:.2f} ± {stats['sd']:.2f}%{skip_txt}"
        )

    lines.extend(["", "## Paired Δ coverage (filter − uniform)", ""])
    for condition, tau, _ in TAU_ARMS:
        key = f"{condition}_minus_uniform_terminal_cov"
        c = contrasts.get(key, {"n": 0})
        if c.get("n", 0) == 0:
            lines.append(f"- **τ={tau:g}** terminal: no paired seeds")
            continue
        lines.append(
            f"- **τ={tau:g}** terminal: Δ={c['mean_delta']:+.2f} pp "
            f"(SD {c['sd']:.2f}; {c['positive_seeds']}/{c['n']} seeds)"
        )

    lines.extend(["", "## Anytime Δ coverage @ 15k / 20k evals (filter − uniform)", ""])
    for condition, tau, _ in TAU_ARMS:
        ac = anytime_contrasts.get(condition, {})
        parts = []
        for ev in ANYTIME_EVALS:
            c = ac.get(ev, {"n": 0})
            if c.get("n", 0) == 0:
                parts.append(f"{ev}: n/a")
            else:
                parts.append(
                    f"{ev}: {c['mean_delta']:+.2f} pp ({c['positive_seeds']}/{c['n']})"
                )
        lines.append(f"- **τ={tau:g}**: " + "; ".join(parts))

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "Run: `./scripts/run_experiment_batch.sh q1-h2-threshold-sensitivity [first_seed] [last_seed]`",
            "Analyze: `uv run python scripts/analyze_h2_threshold_sensitivity.py`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

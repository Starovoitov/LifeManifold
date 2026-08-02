#!/usr/bin/env python3
"""H2 gate decision-level diagnostics from archived genetic_me_filter logs.

Package A (offline): uses surrogate_archive.jsonl + archive_trace.jsonl.
Skipped proposals were not simulated, so true false-skip / missed-elite rates
on skips are unavailable without shadow/oracle runs.

Writes artifacts/experiments/h2-gate-diagnostics/.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILTER_ROOT = ROOT / "artifacts/experiments/q1-v3-genetic-me-filter/genetic_me_filter"
UNIFORM_ROOT = (
    ROOT / "artifacts/experiments/q1-v3-genetic-me-uniform/genetic_me_uniform"
)
OUT_DIR = ROOT / "artifacts/experiments/h2-gate-diagnostics"
TAU = 0.45
EXT_HIGH = 0.8
PHASE_CUTS = (0.33, 0.67)  # early / mid / late by iteration fraction


def _mean_sd(xs: list[float]) -> dict[str, float]:
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def _coverage_at_evals(trace_path: Path, budgets: list[int]) -> dict[int, float]:
    """Last coverage at or before each evaluation count (warm-start inclusive)."""
    if not trace_path.is_file():
        return {}
    points: list[tuple[int, float]] = []
    with trace_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            ev = int(row.get("evaluations", row.get("n_evaluations", -1)))
            cov = row.get("coverage")
            if cov is None and "coverage_pct" in row:
                cov = float(row["coverage_pct"]) / 100.0
            if ev < 0 or cov is None:
                continue
            points.append((ev, float(cov)))
    points.sort()
    out: dict[int, float] = {}
    for b in budgets:
        best = None
        for ev, cov in points:
            if ev <= b:
                best = cov
            else:
                break
        if best is not None:
            out[b] = best
    return out


def _auc_to_horizon(trace_path: Path, horizon: int) -> float | None:
    """Trapezoid AUC of coverage vs evaluations, normalized by horizon."""
    if not trace_path.is_file():
        return None
    xs: list[int] = []
    ys: list[float] = []
    with trace_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            ev = int(row.get("evaluations", row.get("n_evaluations", -1)))
            cov = row.get("coverage")
            if cov is None and "coverage_pct" in row:
                cov = float(row["coverage_pct"]) / 100.0
            if ev < 0 or cov is None or ev > horizon:
                continue
            xs.append(ev)
            ys.append(float(cov))
    if len(xs) < 2:
        return None
    order = np.argsort(xs)
    x = np.asarray(xs, dtype=float)[order]
    y = np.asarray(ys, dtype=float)[order]
    # ensure start at 0 if needed
    if x[0] > 0:
        x = np.concatenate([[0.0], x])
        y = np.concatenate([[y[0]], y])
    if x[-1] < horizon:
        x = np.concatenate([x, [float(horizon)]])
        y = np.concatenate([y, [y[-1]]])
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trapz(y, x) / float(horizon))


def _phase(iteration: int, max_iter: int) -> str:
    if max_iter <= 0:
        return "unknown"
    frac = iteration / float(max_iter)
    if frac <= PHASE_CUTS[0]:
        return "early"
    if frac <= PHASE_CUTS[1]:
        return "mid"
    return "late"


def analyze_seed(seed: int) -> dict[str, Any]:
    arch = FILTER_ROOT / f"seed_{seed}" / "surrogate_archive.jsonl"
    rows: list[dict[str, Any]] = []
    with arch.open() as fh:
        for line in fh:
            rows.append(json.loads(line))
    max_iter = max(int(r.get("iteration") or 0) for r in rows) if rows else 0

    n = len(rows)
    n_skip = sum(1 for r in rows if r.get("decision") == "skip")
    n_eval = n - n_skip

    # predicted anatomy
    skip_fit: list[float] = []
    eval_fit: list[float] = []
    skip_pext: list[float] = []
    eval_pext: list[float] = []
    skip_ext_high = 0
    skip_by_emitter: dict[str, int] = defaultdict(int)
    eval_by_emitter: dict[str, int] = defaultdict(int)
    slots_by_emitter: dict[str, int] = defaultdict(int)
    skip_by_phase: dict[str, int] = defaultdict(int)
    slots_by_phase: dict[str, int] = defaultdict(int)

    # evaluated-only decision proxies (true labels available)
    tp = fp = tn = fn = 0  # insertion event vs pred above tau (among evaluated)
    # "should insert" proxy: accepted OR improved (archive update)
    insert_events = 0
    low_true_below_tau = (
        0  # evaluated with true fit < tau (would-be correct skips if gated on truth)
    )
    extinction_skips = 0

    for r in rows:
        et = str(r.get("emitter_type") or "unknown")
        it = int(r.get("iteration") or 0)
        ph = _phase(it, max_iter)
        slots_by_emitter[et] += 1
        slots_by_phase[ph] += 1
        pred = r.get("prediction") or {}
        comps = pred.get("components") or {}
        pfit = pred.get("fitness")
        pext = comps.get("early_extinction_prob")
        decided_skip = r.get("decision") == "skip"

        if decided_skip:
            skip_by_emitter[et] += 1
            skip_by_phase[ph] += 1
            if pfit is not None:
                skip_fit.append(float(pfit))
            if pext is not None:
                skip_pext.append(float(pext))
                if float(pext) > EXT_HIGH:
                    skip_ext_high += 1
                    extinction_skips += 1
        else:
            eval_by_emitter[et] += 1
            if pfit is not None:
                eval_fit.append(float(pfit))
            if pext is not None:
                eval_pext.append(float(pext))
            eo = r.get("eval_outcome") or {}
            true_fit = eo.get("fitness")
            inserted = bool(eo.get("accepted") or eo.get("improved"))
            if inserted:
                insert_events += 1
            if true_fit is not None and float(true_fit) < TAU:
                low_true_below_tau += 1
            # proxy classifier: predict-eval iff pfit >= tau (always true for evaluated under threshold_gate)
            # Among evaluated: precision/recall for insertion using predicted fitness rank is not a gate
            # validation — report insertion rate and true-fitness < tau rate only.
            _ = tp, fp, tn, fn  # kept for schema stability; see notes

    # Proposal- vs eval-indexed coverage at checkpoints
    filt_trace = FILTER_ROOT / f"seed_{seed}" / "archive_trace.jsonl"
    uni_trace = UNIFORM_ROOT / f"seed_{seed}" / "archive_trace.jsonl"
    eval_budgets = [5000, 10000, 15000, 20000]

    return {
        "seed": seed,
        "n_proposals": n,
        "n_skip": n_skip,
        "n_eval": n_eval,
        "skip_rate": n_skip / n if n else float("nan"),
        "skip_ext_high_share": skip_ext_high / n_skip if n_skip else float("nan"),
        "extinction_share_of_skips": (
            extinction_skips / n_skip if n_skip else float("nan")
        ),
        "mean_pred_fit_skip": float(np.mean(skip_fit)) if skip_fit else float("nan"),
        "mean_pred_fit_eval": float(np.mean(eval_fit)) if eval_fit else float("nan"),
        "mean_pext_skip": float(np.mean(skip_pext)) if skip_pext else float("nan"),
        "mean_pext_eval": float(np.mean(eval_pext)) if eval_pext else float("nan"),
        "insert_rate_among_eval": insert_events / n_eval if n_eval else float("nan"),
        "true_fit_below_tau_among_eval": (
            low_true_below_tau / n_eval if n_eval else float("nan")
        ),
        "skip_rate_by_emitter": {
            et: skip_by_emitter[et] / slots_by_emitter[et]
            for et in sorted(slots_by_emitter)
        },
        "skip_rate_by_phase": {
            ph: skip_by_phase[ph] / slots_by_phase[ph]
            for ph in ("early", "mid", "late")
            if slots_by_phase.get(ph)
        },
        "slots_by_emitter": dict(slots_by_emitter),
        "coverage_eval_filter": _coverage_at_evals(filt_trace, eval_budgets),
        "coverage_eval_uniform": _coverage_at_evals(uni_trace, eval_budgets),
        "auc20k_filter": _auc_to_horizon(filt_trace, 20000),
        "auc20k_uniform": _auc_to_horizon(uni_trace, 20000),
        "notes": {
            "false_skip_on_skips": "unavailable (filter mode does not evaluate skips)",
            "empty_vs_occupied": (
                "empty_bin_explore never fires on this warm-start H2 tier; "
                "niche emptiness at proposal time not logged"
            ),
            "insertion_precision_recall": (
                "cannot score skips; among evaluated, gate already accepted all rows"
            ),
        },
    }


def main() -> None:
    seeds = sorted(
        int(p.name.split("_")[1])
        for p in FILTER_ROOT.glob("seed_*")
        if p.is_dir() and (p / "surrogate_archive.jsonl").is_file()
    )
    per_seed = [analyze_seed(s) for s in seeds]

    def collect(key: str) -> list[float]:
        vals = []
        for row in per_seed:
            v = row.get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            vals.append(float(v))
        return vals

    auc_f = collect("auc20k_filter")
    auc_u = collect("auc20k_uniform")
    auc_delta = [a - b for a, b in zip(auc_f, auc_u)]

    cov_deltas: dict[str, list[float]] = defaultdict(list)
    for row in per_seed:
        cf = row["coverage_eval_filter"]
        cu = row["coverage_eval_uniform"]
        for b, fv in cf.items():
            if b in cu:
                # coverage may be fraction or already percent — normalize to pp
                f = float(fv)
                u = float(cu[b])
                if f <= 1.5 and u <= 1.5:
                    cov_deltas[str(b)].append((f - u) * 100.0)
                else:
                    cov_deltas[str(b)].append(f - u)

    # emitter / phase skip rates across seeds
    emitter_rates: dict[str, list[float]] = defaultdict(list)
    phase_rates: dict[str, list[float]] = defaultdict(list)
    for row in per_seed:
        for et, rate in row["skip_rate_by_emitter"].items():
            emitter_rates[et].append(float(rate))
        for ph, rate in row["skip_rate_by_phase"].items():
            phase_rates[ph].append(float(rate))

    summary = {
        "tau": TAU,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "skip_rate": _mean_sd(collect("skip_rate")),
        "skip_ext_high_share": _mean_sd(collect("skip_ext_high_share")),
        "mean_pred_fit_skip": _mean_sd(collect("mean_pred_fit_skip")),
        "mean_pred_fit_eval": _mean_sd(collect("mean_pred_fit_eval")),
        "insert_rate_among_eval": _mean_sd(collect("insert_rate_among_eval")),
        "true_fit_below_tau_among_eval": _mean_sd(
            collect("true_fit_below_tau_among_eval")
        ),
        "auc20k_filter": _mean_sd(auc_f),
        "auc20k_uniform": _mean_sd(auc_u),
        "auc20k_delta": _mean_sd(auc_delta),
        "auc20k_delta_n_positive": int(sum(1 for d in auc_delta if d > 0)),
        "coverage_delta_pp_at_eval": {k: _mean_sd(v) for k, v in cov_deltas.items()},
        "skip_rate_by_emitter": {k: _mean_sd(v) for k, v in emitter_rates.items()},
        "skip_rate_by_phase": {k: _mean_sd(v) for k, v in phase_rates.items()},
        "limits": {
            "claim": (
                "H2 measures simulator efficiency under a fixed proposal budget "
                "with skips, not a pure surrogate-ranking effect"
            ),
            "missing_controls": [
                "rate-matched random skip with same force-eval logic",
                "oracle filter upper bound (true-fitness gate)",
                "extend filter arm to same real-eval count as baseline",
                "shadow genetic_me for false-skip / missed-elite on skips",
                "niche-relative (incumbent) gate instead of absolute tau",
            ],
            "endpoints": (
                "20k checkpoint and 50%/55% evals-to-threshold are descriptive "
                "reporting points from the matched ANALYSIS; normalized coverage "
                "AUC through 20k is the scalar companion already computed in-tier"
            ),
        },
        "per_seed": per_seed,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "h2_gate_diagnostics.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    md = OUT_DIR / "ANALYSIS.md"
    sr = summary["skip_rate"]
    lines = [
        "# H2 gate diagnostics (package A; offline)",
        "",
        f"**Tier:** `genetic_me_filter` vs `genetic_me_uniform` · seeds={seeds}",
        f"**τ:** {TAU}",
        "",
        "## Claim scope",
        "",
        summary["limits"]["claim"] + ".",
        "",
        "## Scalar companion (descriptive)",
        "",
        "| Metric | Mean ± SD |",
        "|--------|----------:|",
        f"| Skip rate | {sr['mean']*100:.1f}% ± {sr['sd']*100:.1f} |",
        f"| AUC@20k filter | {summary['auc20k_filter']['mean']:.4f} ± {summary['auc20k_filter']['sd']:.4f} |",
        f"| AUC@20k uniform | {summary['auc20k_uniform']['mean']:.4f} ± {summary['auc20k_uniform']['sd']:.4f} |",
        f"| ΔAUC@20k | {summary['auc20k_delta']['mean']:+.4f} ± {summary['auc20k_delta']['sd']:.4f} "
        f"({summary['auc20k_delta_n_positive']}/{len(auc_delta)}) |",
        f"| Insert rate among evaluated | "
        f"{summary['insert_rate_among_eval']['mean']*100:.2f}% ± "
        f"{summary['insert_rate_among_eval']['sd']*100:.2f} |",
        f"| True fitness < τ among evaluated | "
        f"{summary['true_fit_below_tau_among_eval']['mean']*100:.1f}% ± "
        f"{summary['true_fit_below_tau_among_eval']['sd']*100:.1f} |",
        f"| Skips with p_ext > {EXT_HIGH} | "
        f"{summary['skip_ext_high_share']['mean']*100:.1f}% ± "
        f"{summary['skip_ext_high_share']['sd']*100:.1f} |",
        "",
        "### Execution-indexed Δcov (pp; filter − uniform)",
        "",
        "| Eval budget | Δcov pp |",
        "|-------------|---------|",
    ]
    for k in sorted(summary["coverage_delta_pp_at_eval"], key=int):
        d = summary["coverage_delta_pp_at_eval"][k]
        lines.append(f"| {k} | {d['mean']:+.2f} ± {d['sd']:.2f} |")
    lines += [
        "",
        "### Skip rate by emitter / phase",
        "",
        "| Slice | Skip rate |",
        "|-------|----------:|",
    ]
    for et, d in sorted(summary["skip_rate_by_emitter"].items()):
        lines.append(f"| emitter={et} | {d['mean']*100:.1f}% ± {d['sd']*100:.1f} |")
    for ph in ("early", "mid", "late"):
        d = summary["skip_rate_by_phase"].get(ph)
        if d:
            lines.append(f"| phase={ph} | {d['mean']*100:.1f}% ± {d['sd']*100:.1f} |")
    lines += [
        "",
        "## Unavailable without new runs",
        "",
    ]
    for item in summary["limits"]["missing_controls"]:
        lines.append(f"- {item}")
    lines += [
        "",
        f"- Endpoints note: {summary['limits']['endpoints']}",
        "",
        "Script: `scripts/analyze_h2_gate_diagnostics.py`",
        f"JSON: `{out_json.relative_to(ROOT)}`",
        "",
    ]
    md.write_text("\n".join(lines))
    print(md.read_text())
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Confirmatory + sensitivity statistics for Q1 grid/CVT (Wilcoxon, bootstrap CI, Holm, TOST)."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
GRID_CSV = ROOT / "artifacts/experiments/q1-full/summary.csv"
CVT_CSV = ROOT / "artifacts/experiments/q1-cvt/summary.csv"
REPEAT_CSV = ROOT / "artifacts/experiments/q1-repeat/summary.csv"
COMPOSE_AB_JSON = ROOT / "artifacts/surrogate/compose_ab_check.json"
COMPOSE_GATE_JSON = ROOT / "artifacts/surrogate/compose_gate_0p5_vs_0p95.json"
COMPOSE_GATE_LIVE_JSON = ROOT / "artifacts/surrogate/compose_gate_live_0p5_vs_0p95.json"
COMPOSE_GATE_LIVE_MINFIT010_JSON = (
    ROOT / "artifacts/surrogate/compose_gate_live_0p5_vs_0p95_minfit0p10.json"
)
OUT_JSON = ROOT / "artifacts/Q1_GRID_CVT_ANALYSIS.json"
OUT_MD = ROOT / "artifacts/Q1_GRID_CVT_ANALYSIS.md"
COMBINED_CSV = ROOT / "artifacts/Q1_COMBINED_SUMMARY.csv"
STATS_MD_SECTION = "## 7. Confirmatory statistics (Wilcoxon, bootstrap CI, Holm, TOST)"

BOOTSTRAP_B = 10_000
RNG = random.Random(42)


@dataclass
class TestResult:
    name: str
    p: float
    local_ok: bool | None = None
    noise_indistinguishable: bool | None = None
    extras: dict[str, Any] | None = None


def load_arm_csv(path: Path, arm: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row = dict(row)
            row["arm"] = arm
            rows.append(row)
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def pooled_within_sd(values_by_group: list[list[float]]) -> float | None:
    variances: list[float] = []
    for group in values_by_group:
        if len(group) < 2:
            continue
        variances.append(float(np.var(group, ddof=1)))
    if not variances:
        return None
    return float(np.sqrt(np.mean(variances)))


def compute_variance_floor(repeat_rows: list[dict[str, str]]) -> dict[str, Any]:
    if not repeat_rows:
        return {
            "coverage": None,
            "fitness": None,
            "evals": None,
            "status": "SKIPPED (run ./scripts/run_experiment_batch.sh q1-repeat)",
        }

    metric_cols = {
        "coverage": "coverage_pct",
        "fitness": "mean_best_fitness",
        "evals": "evaluations",
    }
    groups: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in repeat_rows:
        rep = row.get("replicate")
        if rep in (None, ""):
            continue
        key = (row["condition"], int(row["seed"]))
        groups.setdefault(key, []).append(row)

    if not groups:
        return {
            "coverage": None,
            "fitness": None,
            "evals": None,
            "status": "SKIPPED (q1-repeat summary has no replicate column)",
        }

    floors: dict[str, float | None] = {}
    details: dict[str, Any] = {}
    for name, col in metric_cols.items():
        by_group: list[list[float]] = []
        per_group: list[dict[str, Any]] = []
        for (condition, seed), rows in sorted(groups.items()):
            if len(rows) < 2:
                continue
            vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
            if len(vals) >= 2:
                by_group.append(vals)
                per_group.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "n": len(vals),
                        "sd": float(np.std(vals, ddof=1)),
                        "values": vals,
                    }
                )
        pooled = pooled_within_sd(by_group)
        floors[name] = 2.0 * pooled if pooled is not None else None
        details[name] = {
            "pooled_within_sd": pooled,
            "floor_2x": floors[name],
            "groups": per_group,
        }

    return {
        "coverage": floors["coverage"],
        "fitness": floors["fitness"],
        "evals": floors["evals"],
        "status": "COMPUTED from q1-repeat",
        "source": str(REPEAT_CSV),
        "details": details,
    }


def paired_delta(
    rows: list[dict[str, str]],
    cond_t: str,
    cond_b: str,
    metric: str,
    *,
    relative: bool = False,
) -> np.ndarray:
    t_map = {int(r["seed"]): float(r[metric]) for r in rows if r["condition"] == cond_t}
    b_map = {int(r["seed"]): float(r[metric]) for r in rows if r["condition"] == cond_b}
    seeds = sorted(set(t_map) & set(b_map))
    deltas = []
    for seed in seeds:
        t_val = t_map[seed]
        b_val = b_map[seed]
        delta = t_val - b_val
        if relative:
            delta = delta / b_val if b_val != 0 else float("nan")
        deltas.append(delta)
    return np.asarray(deltas, dtype=float)


def signed_rank_1s(delta: np.ndarray, direction: Literal["greater", "less"]) -> float:
    if len(delta) < 1:
        return float("nan")
    if np.allclose(delta, 0):
        return 1.0
    result = cast(Any, stats.wilcoxon(delta, alternative=direction))
    return float(result.pvalue)


def dz(delta: np.ndarray) -> float:
    sd = float(np.std(delta, ddof=1))
    return float(np.mean(delta) / sd) if sd > 1e-12 else float("inf")


def bootstrap_ci(
    delta: np.ndarray,
    *,
    stat: str = "median",
    level: float = 0.95,
    b: int = BOOTSTRAP_B,
) -> tuple[float, float]:
    n = len(delta)
    if n == 0:
        return (float("nan"), float("nan"))
    alpha = 1.0 - level
    samples = []
    for _ in range(b):
        idx = [RNG.randrange(n) for _ in range(n)]
        draw = delta[idx]
        samples.append(float(np.median(draw) if stat == "median" else np.mean(draw)))
    samples.sort()
    lo = samples[int((alpha / 2) * b)]
    hi = samples[int((1 - alpha / 2) * b) - 1]
    return (lo, hi)


def shapiro_p(delta: np.ndarray) -> float:
    if len(delta) < 3:
        return 0.0
    return float(stats.shapiro(delta).pvalue)


def boundary_effect(delta: np.ndarray, eps: float = 1e-9) -> bool:
    return bool(np.any(np.abs(delta) < eps))


def symmetry_ok(delta: np.ndarray) -> bool:
    # crude symmetry check: skew not extreme
    if len(delta) < 3:
        return False
    return abs(float(stats.skew(delta))) < 1.0


def param_tost(delta: np.ndarray, eq_margin: float) -> dict[str, Any]:
    upper = float(
        cast(Any, stats.ttest_1samp(delta, eq_margin, alternative="less")).pvalue
    )
    lower = float(
        cast(Any, stats.ttest_1samp(delta, -eq_margin, alternative="greater")).pvalue
    )
    p_tost = max(upper, lower)
    mean = float(np.mean(delta))
    se = float(stats.sem(delta))
    df = len(delta) - 1
    tcrit = float(stats.t.ppf(0.95, df))
    ci90 = (mean - tcrit * se, mean + tcrit * se)
    return {
        "accepted": p_tost < 0.05,
        "p_tost": p_tost,
        "ci90": ci90,
        "level": "parametric_mean",
    }


def wilcoxon_tost(delta: np.ndarray, eq_margin: float) -> dict[str, Any]:
    # Hodges-Lehmann / binomial TOST not implemented; use bootstrap median TOST.
    return bootstrap_tost_median(delta, eq_margin)


def bootstrap_tost_median(delta: np.ndarray, eq_margin: float) -> dict[str, Any]:
    lo, hi = bootstrap_ci(delta, stat="median", level=0.90)
    accepted = hi <= eq_margin and lo >= -eq_margin
    # pseudo p: fraction of bootstrap medians outside band (two-sided)
    n = len(delta)
    meds = []
    for _ in range(BOOTSTRAP_B):
        idx = [RNG.randrange(n) for _ in range(n)]
        meds.append(float(np.median(delta[idx])))
    outside = sum(1 for m in meds if abs(m) > eq_margin)
    p_tost = (outside + 1) / (BOOTSTRAP_B + 1)
    return {
        "accepted": accepted,
        "p_tost": p_tost,
        "ci90": (lo, hi),
        "level": "bootstrap_median",
    }


def tost(delta: np.ndarray, eq_margin: float) -> dict[str, Any]:
    if shapiro_p(delta) >= 0.10 and not boundary_effect(delta):
        return param_tost(delta, eq_margin)
    if symmetry_ok(delta):
        return wilcoxon_tost(delta, eq_margin)
    return bootstrap_tost_median(delta, eq_margin)


def noninferiority(
    delta: np.ndarray,
    neg_margin: float,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """One-sided non-inferiority: H1 mean(delta) > neg_margin (e.g. > -0.05).

    Uses one-sided t-test when Shapiro p≥0.10; otherwise bootstrap CI lower
    bound at level 1-2α (equiv. one-sided α) vs ``neg_margin``.
    """
    if shapiro_p(delta) >= 0.10 and not boundary_effect(delta):
        p = float(
            cast(
                Any, stats.ttest_1samp(delta, neg_margin, alternative="greater")
            ).pvalue
        )
        mean = float(np.mean(delta))
        se = float(stats.sem(delta))
        df = len(delta) - 1
        # (1-2α) two-sided CI → lower bound is one-sided α for non-inferiority
        tcrit = float(stats.t.ppf(1.0 - alpha, df))
        ci_lo = mean - tcrit * se
        ci_hi = mean + tcrit * se
        level = "parametric_mean"
    else:
        # one-sided α via (1-2α) percentile CI on median
        lo, hi = bootstrap_ci(delta, stat="median", level=1.0 - 2.0 * alpha)
        ci_lo, ci_hi = lo, hi
        outside = 0
        n = len(delta)
        for _ in range(BOOTSTRAP_B):
            idx = [RNG.randrange(n) for _ in range(n)]
            if float(np.median(delta[idx])) <= neg_margin:
                outside += 1
        p = (outside + 1) / (BOOTSTRAP_B + 1)
        level = "bootstrap_median"
    accepted = (p < alpha) and (ci_lo > neg_margin)
    return {
        "accepted": accepted,
        "p": p,
        "ci": (ci_lo, ci_hi),
        "neg_margin": neg_margin,
        "level": level,
    }


def holm_step_down(p_by_name: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    names = list(p_by_name.keys())
    m = len(names)
    ordered = sorted(names, key=lambda k: p_by_name[k])
    reject: dict[str, bool] = {}
    prev_fail = False
    for rank, name in enumerate(ordered):
        adj = alpha / (m - rank)
        rej = (p_by_name[name] < adj) and (not prev_fail)
        reject[name] = rej
        if not rej:
            prev_fail = True
    return reject


def sign_consistency(delta: np.ndarray, grid_dir: int) -> tuple[int, float]:
    agree = int(np.sum(np.sign(delta) == grid_dir))
    # one-sided binom: agree > 5 expected under p=0.5
    result = stats.binomtest(agree, n=len(delta), p=0.5, alternative="greater")
    return agree, float(result.pvalue)


def fmt_ci(ci: tuple[float, float], digits: int = 3) -> str:
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def _div_from_gate_payload(gate: dict[str, Any]) -> float | None:
    """Read divergent_skip from live/proxy gate JSON (threshold-agnostic key preferred)."""
    sens = gate.get("gate_sensitivity") or {}
    if "divergent_skip_fraction" in sens:
        return sens["divergent_skip_fraction"]
    min_fit = gate.get(
        "filter_min_predicted_fitness", sens.get("min_predicted_fitness")
    )
    if min_fit is not None:
        key = f"divergent_skip_fraction_at_min_fit_{float(min_fit):g}"
        if key in sens:
            return sens[key]
    return sens.get("divergent_skip_fraction_at_min_fit_0.45")


def load_d1_gate() -> dict[str, Any]:
    """Load D1 compose/gate quantification if present; else documentation-path fallback.

    Prefer live-proposal replay (`compose_gate_live_0p5_vs_0p95.json`) over the
    buffer hold-out proxy when both exist. Primary combat threshold for Q1 is
    ``min_predicted_fitness=0.45`` (YAML + logged skips); ``0.10`` is a
    sensitivity / superseded calibration value.
    """
    ab = None
    if COMPOSE_AB_JSON.is_file():
        ab = json.loads(COMPOSE_AB_JSON.read_text(encoding="utf-8"))

    sens_010 = None
    div_010 = None
    if COMPOSE_GATE_LIVE_MINFIT010_JSON.is_file():
        gate_010 = json.loads(
            COMPOSE_GATE_LIVE_MINFIT010_JSON.read_text(encoding="utf-8")
        )
        sens_010 = gate_010.get("gate_sensitivity") or {}
        div_010 = _div_from_gate_payload(gate_010)

    if COMPOSE_GATE_LIVE_JSON.is_file():
        gate = json.loads(COMPOSE_GATE_LIVE_JSON.read_text(encoding="utf-8"))
        sens = gate.get("gate_sensitivity") or {}
        div = _div_from_gate_payload(gate)
        min_fit = float(gate.get("filter_min_predicted_fitness", 0.45))
        confirmatory = bool(
            sens.get(
                "RQ3_confirmatory_rule_div_le_0.05",
                gate.get("RQ3_confirmatory_rule_div_le_0.05", False),
            )
        )
        n_prop = sens.get("n_proposals")
        n_seeds = sens.get("n_seeds")
        status = (
            f"MEASURED on live filter proposals "
            f"(n={n_prop} across {n_seeds} seeds): "
            f"divergent_skip={div:.3f} at combat min_fit={min_fit:g} → "
            f"{'confirmatory' if confirmatory else 'EXPLORATORY'} "
            f"(rule: div ≤ 0.05). Source: {COMPOSE_GATE_LIVE_JSON.name}."
        )
        if div_010 is not None:
            status += (
                f" Sensitivity min_fit=0.10: divergent_skip={div_010:.3f} "
                f"(also >0.05; not combat)."
            )
        proxy_div = None
        if COMPOSE_GATE_JSON.is_file():
            proxy = json.loads(COMPOSE_GATE_JSON.read_text(encoding="utf-8"))
            proxy_div = _div_from_gate_payload(proxy)
        caveat = (
            "Primary D1 = live proposal replay at Q1 combat "
            f"min_predicted_fitness={min_fit:g} "
            "(filter YAML + agree_logged_skip=1.0). "
            "§3.5 historically listed 0.10; that value was raised after shadow "
            "calibration and is not the q1-full threshold. "
            "Empty-bin force-eval uses logged decision_reason only."
        )
        if div_010 is not None:
            caveat += (
                f" Sensitivity at min_fit=0.10 → div={div_010:.3f} "
                "(higher, not lower — band proposals become skip@0.5 / eval@0.95)."
            )
        if proxy_div is not None:
            caveat += (
                f" Buffer-holdout proxy div {proxy_div:.3f} @0.45 is superseded "
                "(not used for gate)."
            )
        return {
            "divergent_skip_fraction": div,
            "divergent_skip_fraction_min_fit_0.10": div_010,
            "min_predicted_fitness": min_fit,
            "RQ3_confirmatory": confirmatory,
            "status": status,
            "compose_ab": ab,
            "gate_sensitivity": sens,
            "gate_sensitivity_min_fit_0.10": sens_010,
            "paired_metrics": None,
            "source": "live_proposal_replay",
            "caveat": caveat,
        }

    if not COMPOSE_GATE_JSON.is_file():
        return {
            "divergent_skip_fraction": None,
            "RQ3_confirmatory": True,
            "status": (
                "ASSUMED confirmatory (documentation path §3.6; "
                "run scripts/replay_compose_gate_live.py or "
                "scripts/run_d1_compose_checks.py)"
            ),
            "compose_ab": ab,
            "gate_sensitivity": None,
        }
    gate = json.loads(COMPOSE_GATE_JSON.read_text(encoding="utf-8"))
    sens = gate.get("gate_sensitivity") or {}
    div = _div_from_gate_payload(gate)
    confirmatory = bool(sens.get("RQ3_confirmatory_rule_div_le_0.05", False))
    status = (
        f"MEASURED on buffer hold-out proxy (n={gate.get('n_holdout')}): "
        f"divergent_skip={div:.3f} at min_fit=0.45 → "
        f"{'confirmatory' if confirmatory else 'EXPLORATORY'} "
        f"(rule: div ≤ 0.05). Source: {COMPOSE_GATE_JSON.name}."
    )
    return {
        "divergent_skip_fraction": div,
        "RQ3_confirmatory": confirmatory,
        "status": status,
        "compose_ab": ab,
        "gate_sensitivity": sens,
        "paired_metrics": gate.get("paired_metrics"),
        "source": "buffer_holdout_proxy",
        "caveat": (
            "Proxy uses nightly buffer hold-out (extinction-proxy heavy), "
            "not live emitter proposals; run scripts/replay_compose_gate_live.py "
            "for confirmatory D1. Operational RQ3 results still reported."
        ),
    }


def run_statistics() -> dict[str, Any]:
    grid_rows = load_arm_csv(GRID_CSV, "grid")
    cvt_rows = load_arm_csv(CVT_CSV, "cvt")
    assert set(int(r["seed"]) for r in grid_rows if r["condition"] == "stub") == set(
        range(10)
    )

    floor = compute_variance_floor(load_csv(REPEAT_CSV))
    d1 = load_d1_gate()

    fam: dict[str, dict[str, Any]] = {}

    d_cov = paired_delta(grid_rows, "hints", "stub", "coverage_pct")
    d_fit = paired_delta(grid_rows, "hints", "stub", "mean_best_fitness")
    p1a = signed_rank_1s(d_cov, "greater")
    p1b = signed_rank_1s(d_fit, "greater")
    fam["RQ1"] = {
        "p": max(p1a, p1b),
        "p_cov": p1a,
        "p_fit": p1b,
        "local_ok": dz(d_cov) >= 0.5 and dz(d_fit) >= 0.5,
        "dz_cov": dz(d_cov),
        "dz_fit": dz(d_fit),
        "ci_cov_95": bootstrap_ci(d_cov, stat="median"),
        "ci_fit_95": bootstrap_ci(d_fit, stat="median"),
        "wilcoxon_cov": {"p": p1a, "alternative": "greater"},
        "wilcoxon_fit": {"p": p1b, "alternative": "greater"},
        "noise_indistinguishable": (
            floor.get("coverage") is not None
            and floor.get("fitness") is not None
            and (
                abs(float(np.median(d_cov))) < float(floor["coverage"])
                or abs(float(np.median(d_fit))) < float(floor["fitness"])
            )
        ),
        "delta_cov_pp": d_cov.tolist(),
        "delta_fit": d_fit.tolist(),
    }

    d_ev = paired_delta(grid_rows, "filter", "hints", "evaluations", relative=True)
    p_eval = signed_rank_1s(d_ev, "less")
    baseline_evals = float(
        np.median(
            [float(r["evaluations"]) for r in grid_rows if r["condition"] == "hints"]
        )
    )
    fam["RQ3_eval"] = {
        "p": p_eval,
        "local_ok": float(np.median(d_ev)) <= -0.20,
        "median_rel_change": float(np.median(d_ev)),
        "ci_95": bootstrap_ci(d_ev, stat="median"),
        "wilcoxon": {"p": p_eval, "alternative": "less"},
        "noise_indistinguishable": (
            floor.get("evals") is not None
            and abs(float(np.median(d_ev))) < (float(floor["evals"]) / baseline_evals)
        ),
        "delta_eval_rel": d_ev.tolist(),
    }

    d_cov3 = paired_delta(grid_rows, "filter", "hints", "coverage_pct")
    d_fit3 = paired_delta(
        grid_rows, "filter", "hints", "mean_best_fitness", relative=True
    )
    t_cov = tost(d_cov3, eq_margin=3.0)  # 3 percentage points
    t_fit = tost(d_fit3, eq_margin=0.05)  # 5% relative
    ni_cov = noninferiority(d_cov3, neg_margin=-3.0)
    ni_fit = noninferiority(d_fit3, neg_margin=-0.05)
    fam["RQ3_cov_TOST"] = {
        "p": t_cov["p_tost"],
        "local_ok": t_cov["accepted"],
        "ci90": t_cov["ci90"],
        "level": t_cov["level"],
        "eq_margin_pp": 3.0,
        "delta_pp": d_cov3.tolist(),
    }
    fam["RQ3_fit_TOST"] = {
        "p": t_fit["p_tost"],
        "local_ok": t_fit["accepted"],
        "ci90": t_fit["ci90"],
        "level": t_fit["level"],
        "eq_margin_rel": 0.05,
        "delta_rel": d_fit3.tolist(),
    }

    holm = holm_step_down({k: fam[k]["p"] for k in fam})

    # Amended RQ3 QD gates (post-hoc 2026-07-11): one-sided non-inferiority
    ni = {
        "coverage": ni_cov,
        "fitness": ni_fit,
        "note": (
            "Protocol amendment 2026-07-11: RQ3 QD interpreted as non-inferiority "
            "(Δcov > −3 pp, Δfit_rel > −5%), not symmetric TOST. Formal TOST family "
            "retained for transparency."
        ),
    }

    verdict: dict[str, str] = {}
    rq1_pass = (
        holm.get("RQ1", False)
        and fam["RQ1"]["local_ok"]
        and not fam["RQ1"].get("noise_indistinguishable")
    )
    verdict["RQ1"] = "PASS" if rq1_pass else "FAIL"

    rq3_formal = (
        holm.get("RQ3_eval", False)
        and fam["RQ3_eval"]["local_ok"]
        and holm.get("RQ3_cov_TOST", False)
        and fam["RQ3_cov_TOST"]["local_ok"]
        and holm.get("RQ3_fit_TOST", False)
        and fam["RQ3_fit_TOST"]["local_ok"]
    )
    if fam["RQ3_eval"].get("noise_indistinguishable"):
        rq3_formal = False
    rq3_amended = (
        holm.get("RQ3_eval", False)
        and fam["RQ3_eval"]["local_ok"]
        and ni_cov["accepted"]
        and ni_fit["accepted"]
        and not fam["RQ3_eval"].get("noise_indistinguishable")
    )
    formal_label = "PASS" if rq3_formal else "FAIL"
    amended_label = "PASS" if rq3_amended else "FAIL"

    if not d1["RQ3_confirmatory"]:
        div = d1.get("divergent_skip_fraction")
        div_txt = f"{div:.3f}" if div is not None else "?"
        verdict["RQ3_formal_TOST"] = (
            f"EXPLORATORY ({formal_label}; D1 div={div_txt}>0.05)"
        )
        verdict["RQ3_amended_noninferiority"] = (
            f"EXPLORATORY ({amended_label}; D1 div={div_txt}>0.05)"
        )
        verdict["RQ3"] = verdict["RQ3_amended_noninferiority"]
        d1_src = d1.get("source") or "d1"
        verdict["RQ3_note"] = (
            f"D1 ({d1_src}) fails confirmatory rule "
            f"(divergent_skip={div_txt} > 0.05) → RQ3 reported as exploratory. "
            f"Operational NI would be {amended_label}; formal TOST {formal_label}."
        )
    else:
        verdict["RQ3_formal_TOST"] = formal_label
        verdict["RQ3_amended_noninferiority"] = amended_label
        verdict["RQ3"] = amended_label
        verdict["RQ3_note"] = (
            "Primary RQ3 uses amended non-inferiority (2026-07-11). "
            f"Formal symmetric TOST family: {formal_label} "
            "(fitness TOST fails when filter improves fitness)."
        )

    ds_cov = paired_delta(cvt_rows, "hints", "stub", "coverage_pct")
    ds_ev = paired_delta(cvt_rows, "filter", "hints", "evaluations", relative=True)
    a1, pb1 = sign_consistency(ds_cov, grid_dir=+1)
    a3, pb3 = sign_consistency(ds_ev, grid_dir=-1)
    sens_ps = [("RQ1s", pb1), ("RQ3s", pb3)]
    sens_ps.sort(key=lambda x: x[1])
    sens_reject: dict[str, bool] = {}
    prev = False
    for rank, (name, pval) in enumerate(sens_ps):
        adj = 0.05 / (2 - rank)
        rej = (pval < adj) and (not prev)
        sens_reject[name] = rej
        if not rej:
            prev = True

    verdict["RQ1s"] = f"sign {a1}/10, holm={'reject' if sens_reject['RQ1s'] else 'no'}"
    verdict["RQ3s"] = f"sign {a3}/10, holm={'reject' if sens_reject['RQ3s'] else 'no'}"

    # descriptive CVT Wilcoxon (not in confirmatory family)
    cvt_descriptive = {
        "wilcoxon_delta_cov_greater": signed_rank_1s(ds_cov, "greater"),
        "wilcoxon_eval_rel_less": signed_rank_1s(ds_ev, "less"),
        "bootstrap_ci_cov_median_95": bootstrap_ci(ds_cov),
        "bootstrap_ci_eval_rel_median_95": bootstrap_ci(ds_ev),
    }

    prior_stats = "Wilcoxon/bootstrap only in ad-hoc checks (conversation 2026-07-09); not in repo before this script."

    return {
        "statistics_generated": True,
        "prior_runs_note": prior_stats,
        "variance_floor": floor,
        "d1_gate": d1,
        "confirmatory_family": fam,
        "holm_reject": holm,
        "noninferiority": ni,
        "verdict": verdict,
        "sensitivity": {
            "RQ1s_agree": a1,
            "RQ1s_binom_p": pb1,
            "RQ3s_agree": a3,
            "RQ3s_binom_p": pb3,
            "holm_reject": sens_reject,
            "cvt_descriptive": cvt_descriptive,
        },
        "bootstrap_B": BOOTSTRAP_B,
    }


def append_stats_section(stats: dict[str, Any]) -> None:
    fam = stats["confirmatory_family"]
    holm = stats["holm_reject"]
    v = stats["verdict"]

    lines = [
        "",
        STATS_MD_SECTION,
        "",
        f"*Generated by `scripts/analyze_q1_statistics.py` (bootstrap B={stats['bootstrap_B']}).*",
        "",
        f"**Prior runs:** {stats['prior_runs_note']}",
        "",
        "### B.1 Variance floor",
        "",
        f"- Status: **{stats['variance_floor']['status']}**",
    ]
    vf = stats["variance_floor"]
    if vf.get("coverage") is not None:
        lines.extend(
            [
                f"- Floor coverage (2× pooled within-SD): **{vf['coverage']:.3f} pp** (point estimate)",
                f"- Floor fitness: **{vf['fitness']:.4f}**",
                f"- Floor evals (absolute): **{vf['evals']:.1f}**",
                f"- Source: `{vf.get('source', REPEAT_CSV)}`",
                "- **Uncertainty:** few condition×seed groups → coarse scale only. "
                "Use for large effects (RQ1); do **not** adjudicate ~1 pp contrasts "
                "(seed diagnostics, 1-seed ablation) via `diff <? floor`.",
            ]
        )
    lines.extend(
        [
            "",
            "### B.2 D1 gate (RQ3 confirmatory vs exploratory)",
            "",
            f"- {stats['d1_gate']['status']}",
            f"- `RQ3_confirmatory`: **{stats['d1_gate']['RQ3_confirmatory']}**",
        ]
    )
    if stats["d1_gate"].get("caveat"):
        lines.append(f"- Caveat: {stats['d1_gate']['caveat']}")
    ab = (stats["d1_gate"].get("compose_ab") or {}).get("fitness_compose_ab")
    if ab:
        lines.extend(
            [
                "",
                "**Compose A/B (hard vs soft, gate 0.5 labels):**",
                f"- hard: R²={ab['hard']['r2_fitness']:.3f}, MAE={ab['hard']['mae_fitness']:.4f}",
                f"- soft: R²={ab['soft']['r2_fitness']:.3f}, MAE={ab['soft']['mae_fitness']:.4f}",
                "- Soft worse → keep `use_soft_extinction: false` (protocol §3.6).",
            ]
        )
    sens = stats["d1_gate"].get("gate_sensitivity") or {}
    if sens:
        gate_lines = [
            "",
            "**Gate 0.5 vs 0.95 (hard compose):**",
            (
                f"- divergent_skip @ combat min_fit="
                f"{stats['d1_gate'].get('min_predicted_fitness', 0.45):g}: "
                f"**{sens.get('divergent_skip_fraction', sens.get('divergent_skip_fraction_at_min_fit_0.45')):.3f}**"
            ),
        ]
        div_010 = stats["d1_gate"].get("divergent_skip_fraction_min_fit_0.10")
        if div_010 is not None:
            gate_lines.append(
                f"- divergent_skip @ min_fit=0.10 (sensitivity): **{div_010:.3f}**"
            )
        gate_lines.extend(
            [
                f"- frac pred p_ext in [0.5, 0.95): {sens.get('frac_pred_ext_p_in_[0.5,0.95)'):.3f}",
                f"- mean |Δ pred fitness|: {sens.get('mean_abs_pred_diff'):.3f}",
                (
                    "- Artifacts: `compose_ab_check.json`, "
                    "`compose_gate_live_0p5_vs_0p95.json` (primary @0.45), "
                    "`compose_gate_live_0p5_vs_0p95_minfit0p10.json` (sensitivity), "
                    "`compose_gate_0p5_vs_0p95.json` (buffer proxy)."
                    if stats["d1_gate"].get("source") == "live_proposal_replay"
                    else "- Artifacts: `compose_ab_check.json`, `compose_gate_0p5_vs_0p95.json`."
                ),
            ]
        )
        lines.extend(gate_lines)
    lines.extend(
        [
            "",
            "### B.3–B.5 Confirmatory family (grid primary, Holm m=4)",
            "",
            "| Test | p (family) | Holm reject @0.05 | local_ok | notes |",
            "|------|------------|-------------------|----------|-------|",
        ]
    )

    def row(name: str, notes: str) -> str:
        f = fam[name]
        return f"| {name} | {f['p']:.4g} | {holm.get(name, False)} | {f.get('local_ok')} | {notes} |"

    r1 = fam["RQ1"]
    lines.append(
        row(
            "RQ1",
            f"conjunctive max-p; p_cov={r1['p_cov']:.4g}, p_fit={r1['p_fit']:.4g}; "
            f"dz_cov={r1['dz_cov']:.2f}, dz_fit={r1['dz_fit']:.2f}; "
            f"CI95 cov {fmt_ci(r1['ci_cov_95'])}, fit {fmt_ci(r1['ci_fit_95'])}",
        )
    )
    r3e = fam["RQ3_eval"]
    lines.append(
        row(
            "RQ3_eval",
            f"median rel={r3e['median_rel_change']:.3f}; CI95 {fmt_ci(r3e['ci_95'])}",
        )
    )
    r3c = fam["RQ3_cov_TOST"]
    lines.append(
        row(
            "RQ3_cov_TOST",
            f"margin ±3 pp on mean paired Δ (not max-over-seeds); "
            f"{r3c['level']}; 90% CI {fmt_ci(r3c['ci90'])}",
        )
    )
    r3f = fam["RQ3_fit_TOST"]
    lines.append(
        row(
            "RQ3_fit_TOST",
            f"margin ±5% rel; {r3f['level']}; 90% CI {fmt_ci(r3f['ci_90'] if 'ci_90' in r3f else r3f['ci90'], 4)}",
        )
    )

    lines.extend(
        [
            "",
            "**Holm step-down order (by raw p):** "
            + ", ".join(
                f"{k} (p={fam[k]['p']:.4g}, reject={holm[k]})"
                for k in sorted(fam, key=lambda x: fam[x]["p"])
            ),
            "",
            "### B.5b Amended RQ3 QD gates (non-inferiority, post-hoc 2026-07-11)",
            "",
            stats["noninferiority"]["note"],
            "",
            "Confirmatory unit = **paired seed-level Δ** (n=10), same as TOST. "
            "Worst-seed Δcov (grid seed 6 = −3.32 pp) is descriptive only and does not gate RQ3.",
            "",
            "| Endpoint | neg. margin | p (one-sided) | CI lower | accepted |",
            "|----------|-------------|---------------|----------|----------|",
            (
                f"| Δcoverage (pp) | −3.0 | "
                f"{stats['noninferiority']['coverage']['p']:.4g} | "
                f"{stats['noninferiority']['coverage']['ci'][0]:.3f} | "
                f"**{stats['noninferiority']['coverage']['accepted']}** |"
            ),
            (
                f"| Δfitness (rel) | −5% | "
                f"{stats['noninferiority']['fitness']['p']:.4g} | "
                f"{stats['noninferiority']['fitness']['ci'][0]:.4f} | "
                f"**{stats['noninferiority']['fitness']['accepted']}** |"
            ),
            "",
            "### B.6 Hypothesis verdicts",
            "",
            "| Hypothesis | Verdict |",
            "|------------|---------|",
            f"| RQ1 | **{v['RQ1']}** |",
            f"| RQ3 formal (symmetric TOST Holm family) | **{v.get('RQ3_formal_TOST', v['RQ3'])}** |",
            f"| RQ3 amended (non-inferiority, paper claim) | **{v.get('RQ3_amended_noninferiority', v['RQ3'])}** |",
            f"| RQ1-s (CVT) | {v['RQ1s']} |",
            f"| RQ3-s (CVT) | {v['RQ3s']} |",
            "",
        ]
    )
    if v.get("RQ3_note"):
        lines.extend([f"*{v['RQ3_note']}*", ""])
    lines.extend(
        [
            "### B.7 CVT descriptive (not in confirmatory Holm family)",
            "",
            f"- Wilcoxon Δcoverage hints−stub (greater): p={stats['sensitivity']['cvt_descriptive']['wilcoxon_delta_cov_greater']:.4g}",
            f"- Wilcoxon Δeval rel filter−hints (less): p={stats['sensitivity']['cvt_descriptive']['wilcoxon_eval_rel_less']:.4g}",
            f"- Bootstrap 95% CI Δcoverage median: {fmt_ci(tuple(stats['sensitivity']['cvt_descriptive']['bootstrap_ci_cov_median_95']))}",
            "",
        ]
    )

    md = OUT_MD.read_text(encoding="utf-8")
    section8 = ""
    marker8 = "## 8. Prompt ablation"
    if marker8 in md:
        section8 = "\n" + md[md.index(marker8) :].rstrip() + "\n"
    if STATS_MD_SECTION in md:
        md = md.split(STATS_MD_SECTION)[0].rstrip() + "\n"
    OUT_MD.write_text(md + "\n".join(lines) + section8, encoding="utf-8")


def main() -> None:
    stats = run_statistics()

    combined: list[dict[str, str]] = []
    combined.extend(load_arm_csv(GRID_CSV, "grid"))
    combined.extend(load_arm_csv(CVT_CSV, "cvt"))
    if combined:
        fieldnames = list(combined[0].keys())
        with COMBINED_CSV.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined)

    if OUT_JSON.is_file():
        payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    else:
        payload = {}
    payload["statistics"] = stats
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    append_stats_section(stats)
    print(
        json.dumps(
            {
                "verdict": stats["verdict"],
                "holm": stats["holm_reject"],
                "noninferiority": {
                    "cov_accepted": stats["noninferiority"]["coverage"]["accepted"],
                    "fit_accepted": stats["noninferiority"]["fitness"]["accepted"],
                    "fit_ci": stats["noninferiority"]["fitness"]["ci"],
                    "fit_p": stats["noninferiority"]["fitness"]["p"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

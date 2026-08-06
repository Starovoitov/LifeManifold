#!/usr/bin/env python3
"""End-to-end numeric audit: regenerate claim-driving table values from archives.

Prints recomputed numbers for manuscript cross-checks (H1 bundled, matched H1,
H4 levels, multi-provider Δfit, H3-gray NI methods). Exit code 1 if any hard
check fails.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import stats as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_q1_statistics import (  # noqa: E402
    noninferiority,
    param_tost,
    vargha_delaney_a12_paired,
)

EPS = {
    "cov_pp": 0.02,
    "fit": 0.0005,
    "p_rel": 1e-9,
}


def _load_summary(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open()))


def _by_cond(rows: list[dict[str, str]], cond: str) -> dict[int, dict[str, str]]:
    return {int(r["seed"]): r for r in rows if r["condition"] == cond}


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


def _cov_from_run(path: Path) -> float:
    j = json.loads(path.read_text(encoding="utf-8"))
    cov = float(j.get("coverage_pct", j.get("coverage", 0.0)))
    return cov * 100.0 if cov <= 1.5 else cov


def check(name: str, ok: bool, detail: str, failures: list[str]) -> None:
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not ok:
        failures.append(name)


def main() -> int:
    failures: list[str] = []
    print("=== Manuscript numeric audit (from archived experiment data) ===\n")

    # --- Bundled hints − stub (q1-full) ---
    grid = _load_summary(ROOT / "artifacts/experiments/q1-full/summary.csv")
    hints = _by_cond(grid, "hints")
    stub = _by_cond(grid, "stub")
    seeds = sorted(set(hints) & set(stub))
    d_fit = np.array(
        [
            float(hints[s]["mean_best_fitness"]) - float(stub[s]["mean_best_fitness"])
            for s in seeds
        ]
    )
    d_cov = np.array(
        [
            float(hints[s]["coverage_pct"]) - float(stub[s]["coverage_pct"])
            for s in seeds
        ]
    )
    h_fit_m, h_fit_s = _mean_sd([float(hints[s]["mean_best_fitness"]) for s in seeds])
    s_fit_m, s_fit_s = _mean_sd([float(stub[s]["mean_best_fitness"]) for s in seeds])
    print("--- Table rq1 / levels ---")
    print(
        f"  hints fit {h_fit_m:.6f} ± {h_fit_s:.6f}; "
        f"stub fit {s_fit_m:.6f} ± {s_fit_s:.6f}"
    )
    print(
        f"  Δcov mean {d_cov.mean():+.4f}; Δfit mean {d_fit.mean():+.6f} "
        f"(median {np.median(d_fit):+.6f})"
    )
    check(
        "hints_fitness_not_0.539",
        abs(h_fit_m - 0.499) < 0.001,
        f"mean={h_fit_m:.6f} (must be ~0.499, not 0.539)",
        failures,
    )
    check(
        "qwen_delta_fit_tab_rq1",
        abs(d_fit.mean() - 0.047) < 0.001,
        f"mean Δfit={d_fit.mean():+.6f} → report +0.047 not +0.053",
        failures,
    )
    check(
        "stub_fit_sd_not_0.041",
        abs(s_fit_s - 0.012) < 0.002,
        f"sd={s_fit_s:.6f} → report ±0.012 not ±0.041",
        failures,
    )

    # --- Bundled floor: vanilla vs hints (must arithmetically match hints 0.499) ---
    print("\n--- Bundled floor vanilla − hints ---")
    van_rows = _load_summary(ROOT / "artifacts/experiments/q1-v3-vanilla/summary.csv")
    van = _by_cond(van_rows, "vanilla")
    v_fit = [float(van[s]["mean_best_fitness"]) for s in seeds]
    v_fit_m, v_fit_s = _mean_sd(v_fit)
    d_fit_v = np.array(
        [
            float(hints[s]["mean_best_fitness"]) - float(van[s]["mean_best_fitness"])
            for s in seeds
        ]
    )
    d_cov_v = np.array(
        [float(hints[s]["coverage_pct"]) - float(van[s]["coverage_pct"]) for s in seeds]
    )
    print(
        f"  vanilla fit {v_fit_m:.6f} ± {v_fit_s:.6f}; "
        f"hints−van Δfit {d_fit_v.mean():+.6f}; Δcov {d_cov_v.mean():+.4f}"
    )
    check(
        "vanilla_fit_not_0.373",
        abs(v_fit_m - 0.432) < 0.002,
        f"mean={v_fit_m:.6f} (must be ~0.432 from summary.csv, not 0.373)",
        failures,
    )
    check(
        "vanilla_delta_fit_not_0.166",
        abs(d_fit_v.mean() - 0.067) < 0.002,
        f"mean Δfit={d_fit_v.mean():+.6f} → report +0.067 not +0.166",
        failures,
    )
    check(
        "vanilla_plus_delta_equals_hints",
        abs(v_fit_m + d_fit_v.mean() - h_fit_m) < EPS["fit"],
        f"0.432+Δ implies hints={v_fit_m + d_fit_v.mean():.6f}",
        failures,
    )

    # --- H4 levels ---
    print("\n--- H4 pyribs levels vs frozen hints ---")
    py = _load_summary(ROOT / "artifacts/experiments/q1-v3-pyribs/summary.csv")
    for arm in ("cma_me", "cma_mae"):
        rows = _by_cond(py, arm)
        fit_m, fit_s = _mean_sd([float(rows[s]["mean_best_fitness"]) for s in seeds])
        cov_m, cov_s = _mean_sd([float(rows[s]["coverage_pct"]) for s in seeds])
        d_f = np.array(
            [
                float(hints[s]["mean_best_fitness"])
                - float(rows[s]["mean_best_fitness"])
                for s in seeds
            ]
        )
        print(
            f"  {arm}: cov {cov_m:.2f}±{cov_s:.2f}; fit {fit_m:.6f}±{fit_s:.6f}; "
            f"hints−arm Δfit {d_f.mean():+.6f}"
        )
        check(
            f"h4_hints_consistent_{arm}",
            abs(fit_m + d_f.mean() - h_fit_m) < EPS["fit"],
            f"cma+Δ implies hints={fit_m + d_f.mean():.6f}",
            failures,
        )

    # --- Matched H1 providers: mean-TOST ---
    print("\n--- Matched H1 mean-TOST (paired-t) ---")
    for slug, label in (
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("deepseek-v4-pro", "DeepSeek V4 Pro"),
    ):
        root = ROOT / "artifacts/experiments/q1-v3-llm" / slug
        deltas = []
        for s in seeds:
            su = _cov_from_run(
                root / "stub_uniform" / f"seed_{s}" / "nightly_run_summary.json"
            )
            hi = _cov_from_run(
                root / "hints" / f"seed_{s}" / "nightly_run_summary.json"
            )
            deltas.append(hi - su)
        d = np.asarray(deltas, dtype=float)
        t = param_tost(d, 2.0)
        print(
            f"  {label}: mean={d.mean():+.3f} sd={d.std(ddof=1):.3f}; "
            f"mean-TOST {'accept' if t['accepted'] else 'REJECT'} "
            f"90% CI [{t['ci90'][0]:+.2f},{t['ci90'][1]:+.2f}] "
            f"p={t['p_tost']:.4g}"
        )
        if slug == "deepseek-v4-pro":
            check(
                "deepseek_mean_tost_rejects",
                not t["accepted"],
                "paired-t 90% CI must leave [-2,2]; do not claim accept",
                failures,
            )
        if slug == "gpt-4o-mini":
            check(
                "gpt4o_mean_tost_accepts",
                t["accepted"],
                "paired-t mean-TOST should accept",
                failures,
            )

    # qwen matched
    su_root = ROOT / "artifacts/experiments/q1-stub-uniform-sensitivity"
    q_d = np.array(
        [
            float(hints[s]["coverage_pct"])
            - _cov_from_run(
                su_root / "stub_uniform" / f"seed_{s}" / "nightly_run_summary.json"
            )
            for s in seeds
        ]
    )
    qt = param_tost(q_d, 2.0)
    print(
        f"  qwen-turbo: mean={q_d.mean():+.3f} sd={q_d.std(ddof=1):.3f}; "
        f"mean-TOST {'accept' if qt['accepted'] else 'REJECT'} "
        f"90% CI [{qt['ci90'][0]:+.2f},{qt['ci90'][1]:+.2f}]"
    )
    check("qwen_mean_tost_accepts", qt["accepted"], "primary paired-t TOST", failures)

    # --- H3-gray: methods ---
    print("\n--- H3-gray confirmatory (methods + floor on Wilcoxon) ---")
    h3 = json.loads(
        (
            ROOT
            / "artifacts/experiments/q1-v3-h3-gray-zone"
            / "h3_gray_zone_confirmatory_holm.json"
        ).read_text(encoding="utf-8")
    )
    d_eval = np.array([r["delta_eval"] for r in h3["per_seed"]], dtype=float)
    d_c = np.array([r["delta_cov_pp"] for r in h3["per_seed"]], dtype=float)
    d_fr = np.array([r["delta_fit_rel"] for r in h3["per_seed"]], dtype=float)
    p_w = float(cast(Any, sp.wilcoxon(d_eval, alternative="less")).pvalue)
    ni_c = noninferiority(d_c, -3.0)
    ni_f = noninferiority(d_fr, -0.05)
    wilcoxon_floor = 1.0 / (2 ** len(seeds))
    print(
        f"  eval Wilcoxon p={p_w:.4g} (floor {wilcoxon_floor:.4g}); "
        f"PPS={vargha_delaney_a12_paired(d_eval, direction='less'):.2f}"
    )
    print(
        f"  cov NI: p={ni_c['p']:.4g} level={ni_c['level']} "
        f"accepted={ni_c['accepted']}"
    )
    print(
        f"  fit NI: p={ni_f['p']:.4g} level={ni_f['level']} "
        f"accepted={ni_f['accepted']}"
    )
    check(
        "h3_eval_p_is_wilcoxon_floor",
        abs(p_w - wilcoxon_floor) < 1e-12 or p_w >= wilcoxon_floor - 1e-15,
        f"p={p_w:.4g}",
        failures,
    )
    check(
        "h3_ni_p_may_be_below_wilcoxon_floor",
        ni_c["level"] != "wilcoxon" and ni_f["level"] != "wilcoxon",
        "disclose NI methods are t/bootstrap, not exact Wilcoxon",
        failures,
    )
    check(
        "h3_ni_p_below_floor_as_reported",
        ni_c["p"] < wilcoxon_floor and ni_f["p"] < wilcoxon_floor,
        "explains manuscript p=1e-4 / 1.6e-5 vs Wilcoxon floor",
        failures,
    )

    print("\n=== Summary ===")
    if failures:
        print(f"{len(failures)} hard check(s) failed: {', '.join(failures)}")
        return 1
    print("All hard checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

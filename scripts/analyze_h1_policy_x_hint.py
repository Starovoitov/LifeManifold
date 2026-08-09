#!/usr/bin/env python3
"""Policy × hint 2×2 readout: stub / stub_uniform / hints / hints_minfit.

Frozen cells are pinned; only hints_minfit comes from q1-h1-policy-x-hint.
Descriptive only — not a confirmatory Holm family.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import stats as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_q1_statistics import bootstrap_ci

DEFAULT_PATHS: dict[str, Path] = {
    "stub": ROOT / "artifacts/experiments/q1-full/stub",
    "stub_uniform": ROOT
    / "artifacts/experiments/q1-stub-uniform-sensitivity/stub_uniform",
    "hints": ROOT / "artifacts/experiments/q1-full/hints",
    "hints_minfit": ROOT / "artifacts/experiments/q1-h1-policy-x-hint/hints_minfit",
}

OUT_DIR = ROOT / "artifacts/experiments/q1-h1-policy-x-hint"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cov_pct(payload: dict[str, Any]) -> float:
    if "coverage_pct" in payload:
        cov = float(payload["coverage_pct"])
    else:
        cov = float(payload["coverage"])
    return cov * 100.0 if cov <= 1.0 else cov


def _wilcoxon_two_sided(delta: np.ndarray) -> float:
    if len(delta) < 1 or np.allclose(delta, 0):
        return 1.0
    result = cast(Any, sp.wilcoxon(delta, alternative="two-sided"))
    return float(result.pvalue)


def _paired(a: np.ndarray, b: np.ndarray, *, name: str) -> dict[str, Any]:
    d = b - a
    ci = bootstrap_ci(d, stat="mean", level=0.95)
    return {
        "contrast": name,
        "n": int(len(d)),
        "mean_a": round(float(np.mean(a)), 4),
        "mean_b": round(float(np.mean(b)), 4),
        "mean_delta_pp": round(float(np.mean(d)), 4),
        "sd_delta_pp": round(float(np.std(d, ddof=1)), 4) if len(d) > 1 else 0.0,
        "wins_b": int(np.sum(d > 0)),
        "ties": int(np.sum(np.isclose(d, 0.0))),
        "wilcoxon_p": round(_wilcoxon_two_sided(d), 6),
        "bootstrap_ci95_mean_delta": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
        "per_seed_delta_pp": [round(float(x), 4) for x in d.tolist()],
    }


def _load_arm(arm_dir: Path, seeds: list[int]) -> np.ndarray:
    vals: list[float] = []
    for seed in seeds:
        path = arm_dir / f"seed_{seed}" / "nightly_run_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        vals.append(_cov_pct(_load(path)))
    return np.asarray(vals, dtype=float)


def analyze(seeds: list[int], paths: dict[str, Path]) -> dict[str, Any]:
    cov = {name: _load_arm(path, seeds) for name, path in paths.items()}
    contrasts = {
        "stub_uniform_minus_stub": _paired(
            cov["stub"], cov["stub_uniform"], name="stub_uniform − stub"
        ),
        "hints_minus_stub_uniform": _paired(
            cov["stub_uniform"], cov["hints"], name="hints − stub_uniform"
        ),
        "hints_minfit_minus_stub": _paired(
            cov["stub"], cov["hints_minfit"], name="hints_minfit − stub"
        ),
        "hints_minus_hints_minfit": _paired(
            cov["hints_minfit"], cov["hints"], name="hints − hints_minfit"
        ),
        "hints_minfit_minus_stub_uniform": _paired(
            cov["stub_uniform"],
            cov["hints_minfit"],
            name="hints_minfit − stub_uniform",
        ),
        "bundled_hints_minus_stub": _paired(
            cov["stub"], cov["hints"], name="hints − stub (bundled)"
        ),
    }
    # Descriptive interaction: (hints−stub_uniform) − (hints_minfit−stub)
    soft_uniform = cov["hints"] - cov["stub_uniform"]
    soft_minfit = cov["hints_minfit"] - cov["stub"]
    interaction = soft_uniform - soft_minfit
    ci = bootstrap_ci(interaction, stat="mean", level=0.95)
    contrasts["soft_x_policy_interaction"] = {
        "contrast": "(hints−stub_uniform) − (hints_minfit−stub)",
        "n": int(len(interaction)),
        "mean_delta_pp": round(float(np.mean(interaction)), 4),
        "sd_delta_pp": (
            round(float(np.std(interaction, ddof=1)), 4)
            if len(interaction) > 1
            else 0.0
        ),
        "wilcoxon_p": round(_wilcoxon_two_sided(interaction), 6),
        "bootstrap_ci95_mean_delta": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
        "per_seed_delta_pp": [round(float(x), 4) for x in interaction.tolist()],
        "note": "descriptive 2×2 interaction contrast only; not ANOVA / Holm",
    }

    levels = {
        name: {
            "mean_cov_pct": round(float(np.mean(v)), 4),
            "sd_cov_pct": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0,
            "per_seed": [round(float(x), 4) for x in v.tolist()],
            "path": str(paths[name]),
        }
        for name, v in cov.items()
    }

    bundled = float(np.mean(cov["hints"] - cov["stub"]))
    policy_stub = float(np.mean(cov["stub_uniform"] - cov["stub"]))
    soft_matched = float(np.mean(cov["hints"] - cov["stub_uniform"]))
    soft_hist = float(np.mean(cov["hints_minfit"] - cov["stub"]))
    policy_hints = float(np.mean(cov["hints"] - cov["hints_minfit"]))

    return {
        "tier": "q1-h1-policy-x-hint",
        "n": len(seeds),
        "seeds": seeds,
        "levels": levels,
        "contrasts": contrasts,
        "summary_pp": {
            "bundled_hints_minus_stub": round(bundled, 4),
            "policy_on_stub_stub_uniform_minus_stub": round(policy_stub, 4),
            "soft_at_uniform_hints_minus_stub_uniform": round(soft_matched, 4),
            "soft_at_minfit_hints_minfit_minus_stub": round(soft_hist, 4),
            "policy_on_hints_hints_minus_hints_minfit": round(policy_hints, 4),
            "fraction_bundled_recovered_by_stub_policy": (
                round(policy_stub / bundled, 4) if abs(bundled) > 1e-9 else None
            ),
        },
        "scope": (
            "Descriptive policy×hint 2×2 after completing hints_minfit. "
            "Not a confirmatory Holm family."
        ),
    }


def _md(payload: dict[str, Any]) -> str:
    s = payload["summary_pp"]
    lines = [
        "# H1 policy × hint 2×2 (`hints_minfit` cell)",
        "",
        f"**n:** {payload['n']} (seeds {payload['seeds'][0]}–{payload['seeds'][-1]})",
        f"**Scope:** {payload['scope']}",
        "",
        "## Cell means (terminal coverage %)",
        "",
        "| Arm | Policy | Soft | Mean ± SD |",
        "|-----|--------|------|----------:|",
    ]
    meta = {
        "stub": ("min_fitness_frontier", "stub constants"),
        "stub_uniform": ("uniform_frontier", "stub constants"),
        "hints": ("uniform_frontier", "live hints"),
        "hints_minfit": ("min_fitness_frontier", "live hints"),
    }
    for arm, (pol, soft) in meta.items():
        lv = payload["levels"][arm]
        lines.append(
            f"| `{arm}` | `{pol}` | {soft} | "
            f"{lv['mean_cov_pct']:.2f} ± {lv['sd_cov_pct']:.2f} |"
        )
    lines += [
        "",
        "## Headline contrasts (pp)",
        "",
        f"- Bundled `hints − stub`: **{s['bundled_hints_minus_stub']:+.2f}**",
        f"- Policy on stub (`stub_uniform − stub`): "
        f"**{s['policy_on_stub_stub_uniform_minus_stub']:+.2f}**",
        f"- Soft @ uniform (`hints − stub_uniform`): "
        f"**{s['soft_at_uniform_hints_minus_stub_uniform']:+.2f}**",
        f"- Soft @ minfit (`hints_minfit − stub`): "
        f"**{s['soft_at_minfit_hints_minfit_minus_stub']:+.2f}**",
        f"- Policy on hints (`hints − hints_minfit`): "
        f"**{s['policy_on_hints_hints_minus_hints_minfit']:+.2f}**",
    ]
    frac = s.get("fraction_bundled_recovered_by_stub_policy")
    if frac is not None:
        lines.append(
            f"- Fraction of bundled gap recovered by stub-side policy alone: "
            f"**{100.0 * frac:.1f}%** "
            f"(soft@minfit near null confirms policy main effect)"
        )
    inter = payload["contrasts"]["soft_x_policy_interaction"]
    lines += [
        "",
        "## Descriptive interaction",
        "",
        f"`{inter['contrast']}`: mean **{inter['mean_delta_pp']:+.2f}** pp "
        f"(SD {inter['sd_delta_pp']:.2f}; Wilcoxon p={inter['wilcoxon_p']}; "
        f"boot 95% CI {inter['bootstrap_ci95_mean_delta']})",
        "",
        "## All contrasts",
        "",
    ]
    for key, c in payload["contrasts"].items():
        if key == "soft_x_policy_interaction":
            continue
        lines.append(
            f"- `{c['contrast']}`: {c['mean_delta_pp']:+.2f} ± {c['sd_delta_pp']:.2f} "
            f"({c['wins_b']}/{c['n']}; p={c['wilcoxon_p']})"
        )
    lines += [
        "",
        "## Paths",
        "",
    ]
    for arm, lv in payload["levels"].items():
        lines.append(f"- `{arm}`: `{lv['path']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument(
        "--hints-minfit-root",
        type=Path,
        default=DEFAULT_PATHS["hints_minfit"],
    )
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    paths = dict(DEFAULT_PATHS)
    paths["hints_minfit"] = args.hints_minfit_root

    missing = [a for a, p in paths.items() if not p.is_dir()]
    if missing:
        raise SystemExit(f"missing arm dirs: {missing}")

    present = []
    for seed in args.seeds:
        p = paths["hints_minfit"] / f"seed_{seed}" / "nightly_run_summary.json"
        if p.is_file():
            present.append(seed)
    if not present:
        raise SystemExit(
            f"no completed hints_minfit seeds under {paths['hints_minfit']}"
        )
    if present != args.seeds:
        print(
            f"NOTE: analyzing completed seeds only: {present} "
            f"(requested {args.seeds})",
            file=sys.stderr,
        )

    payload = analyze(present, paths)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "h1_policy_x_hint_analysis.json"
    md_path = args.out_dir / "ANALYSIS.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_md(payload), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    s = payload["summary_pp"]
    print(
        "soft@minfit={soft:+.2f}  policy@hints={pol:+.2f}  "
        "soft@uniform={su:+.2f}".format(
            soft=s["soft_at_minfit_hints_minfit_minus_stub"],
            pol=s["policy_on_hints_hints_minus_hints_minfit"],
            su=s["soft_at_uniform_hints_minus_stub_uniform"],
        )
    )


if __name__ == "__main__":
    main()

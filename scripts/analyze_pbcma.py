"""Analyze pbCMA CMA-ME vs native discrete, Bernoulli, rint, hints, genetic ME."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PBCMA_ROOT = ROOT / "artifacts/experiments/q1-v3-pyribs-pbcma"
NATIVE_ROOT = ROOT / "artifacts/experiments/q1-v3-pyribs-native-discrete-cma"
BERNOULLI_ROOT = ROOT / "artifacts/experiments/q1-v3-pyribs-discrete-cma"
CONTINUOUS_ROOT = ROOT / "artifacts/experiments/q1-v3-pyribs"
HINTS_ROOT = ROOT / "artifacts/experiments/q1-full"
GENETIC_ROOT = ROOT / "artifacts/experiments/q1-v3-genetic-me-uniform"
OUT_JSON = PBCMA_ROOT / "pbcma_analysis.json"
OUT_MD = PBCMA_ROOT / "ANALYSIS.md"


def _load_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _coverage_pct(payload: dict) -> float:
    cov = payload.get("coverage")
    if cov is None:
        return float("nan")
    cov_f = float(cov)
    return cov_f * 100.0 if cov_f <= 1.0 else cov_f


def _fitness(payload: dict) -> float:
    val = payload.get("mean_best_fitness")
    return float("nan") if val is None else float(val)


def _arm_stats(values: dict[int, float]) -> dict:
    if not values:
        return {"n": 0}
    vals = list(values.values())
    return {
        "n": len(vals),
        "mean_pct": round(statistics.mean(vals), 4),
        "sd_pct": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
        "min_pct": round(min(vals), 4),
        "max_pct": round(max(vals), 4),
    }


def _paired(rows: list[tuple[int, float, float]]) -> dict:
    if not rows:
        return {"n": 0}
    deltas = [b - a for _, a, b in rows]
    n = len(deltas)
    mean = statistics.mean(deltas)
    sd = statistics.pstdev(deltas) if n > 1 else 0.0
    pos = sum(1 for d in deltas if d > 0)
    return {
        "n": n,
        "mean_delta_pp": round(mean, 4),
        "sd_pp": round(sd, 4),
        "positive_seeds": pos,
        "seeds": [
            {"seed": s, "a": round(a, 4), "b": round(b, 4), "delta_pp": round(b - a, 4)}
            for s, a, b in rows
        ],
    }


def main() -> int:
    seeds = list(range(10))
    arms_cov: dict[str, dict[int, float]] = {}
    arms_fit: dict[str, dict[int, float]] = {}
    for label, root, cond in [
        ("cma_me_pbcma", PBCMA_ROOT, "cma_me_pbcma"),
        ("cma_me_discrete", NATIVE_ROOT, "cma_me_discrete"),
        ("cma_me_bernoulli", BERNOULLI_ROOT, "cma_me_bernoulli"),
        ("cma_me_rint", CONTINUOUS_ROOT, "cma_me"),
        ("hints", HINTS_ROOT, "hints"),
        ("genetic_me_uniform", GENETIC_ROOT, "genetic_me_uniform"),
    ]:
        arms_cov[label] = {}
        arms_fit[label] = {}
        for seed in seeds:
            summary = _load_summary(
                root / cond / f"seed_{seed}" / "nightly_run_summary.json"
            )
            if summary is not None:
                arms_cov[label][seed] = _coverage_pct(summary)
                arms_fit[label][seed] = _fitness(summary)

    contrasts = {
        "pbcma_minus_hints": _paired(
            [
                (s, arms_cov["hints"][s], arms_cov["cma_me_pbcma"][s])
                for s in seeds
                if s in arms_cov["hints"] and s in arms_cov["cma_me_pbcma"]
            ]
        ),
        "pbcma_minus_genetic_uniform": _paired(
            [
                (s, arms_cov["genetic_me_uniform"][s], arms_cov["cma_me_pbcma"][s])
                for s in seeds
                if s in arms_cov["genetic_me_uniform"] and s in arms_cov["cma_me_pbcma"]
            ]
        ),
        "pbcma_minus_native": _paired(
            [
                (s, arms_cov["cma_me_discrete"][s], arms_cov["cma_me_pbcma"][s])
                for s in seeds
                if s in arms_cov["cma_me_discrete"] and s in arms_cov["cma_me_pbcma"]
            ]
        ),
        "pbcma_minus_bernoulli": _paired(
            [
                (s, arms_cov["cma_me_bernoulli"][s], arms_cov["cma_me_pbcma"][s])
                for s in seeds
                if s in arms_cov["cma_me_bernoulli"] and s in arms_cov["cma_me_pbcma"]
            ]
        ),
        "pbcma_minus_rint": _paired(
            [
                (s, arms_cov["cma_me_rint"][s], arms_cov["cma_me_pbcma"][s])
                for s in seeds
                if s in arms_cov["cma_me_rint"] and s in arms_cov["cma_me_pbcma"]
            ]
        ),
    }

    payload = {
        "package": "pbcma",
        "tier": "q1-v3-pyribs-pbcma",
        "condition": "cma_me_pbcma",
        "emitter_kind": "pbcma",
        "note": (
            "Latent-Gaussian (μ/μ_w, λ)-CMA with bit threshold at 0.5 and margin "
            "correction; discrete {0,1} rule bits stored in archive; CMA updated "
            "from latent parents (not discrete archive vectors)."
        ),
        "arm_stats_coverage_pct": {k: _arm_stats(v) for k, v in arms_cov.items()},
        "arm_stats_fitness": {
            k: _arm_stats({i: x for i, x in v.items() if x == x})
            for k, v in arms_fit.items()
        },
        "seeds_available": {k: sorted(v.keys()) for k, v in arms_cov.items()},
        "contrasts": contrasts,
    }

    PBCMA_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    pbcma = _arm_stats(arms_cov.get("cma_me_pbcma", {}))
    lines = [
        "# pbCMA CMA-ME (latent Gaussian + bit threshold + margin)",
        "",
        f"Artifact: `{PBCMA_ROOT.relative_to(ROOT)}`",
        "",
        "## Arm levels (terminal coverage %)",
        "",
    ]
    for label in [
        "cma_me_pbcma",
        "cma_me_discrete",
        "cma_me_bernoulli",
        "cma_me_rint",
        "hints",
        "genetic_me_uniform",
    ]:
        stats = payload["arm_stats_coverage_pct"].get(label, {"n": 0})
        if stats.get("n", 0) == 0:
            lines.append(f"- **{label}**: no data")
            continue
        lines.append(
            f"- **{label}** ($n={stats['n']}$): "
            f"{stats['mean_pct']:.2f} ± {stats['sd_pct']:.2f}% "
            f"(range {stats['min_pct']:.2f}–{stats['max_pct']:.2f})"
        )
    lines.extend(["", "## Paired coverage deltas (pp)", ""])
    for name, stats in contrasts.items():
        if stats.get("n", 0) == 0:
            lines.append(f"- **{name}**: no paired seeds yet")
            continue
        lines.append(
            f"- **{name}** ($n={stats['n']}$): mean $\\Delta$={stats['mean_delta_pp']:+.2f} pp "
            f"(SD {stats['sd_pp']:.2f}; {stats['positive_seeds']}/{stats['n']} seeds positive)"
        )
    ph = contrasts["pbcma_minus_hints"]
    pn = contrasts["pbcma_minus_native"]
    pb = contrasts["pbcma_minus_bernoulli"]
    pr = contrasts["pbcma_minus_rint"]
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"pbCMA mean coverage: {pbcma.get('mean_pct', float('nan')):.2f}% "
            f"(SD {pbcma.get('sd_pct', 0):.2f}).",
        ]
    )
    if ph.get("n"):
        lines.append(
            f"pbCMA vs hints: {ph['mean_delta_pp']:+.2f} pp ({ph['positive_seeds']}/{ph['n']} seeds)."
        )
    if pn.get("n"):
        lines.append(
            f"pbCMA vs native bit-flip: {pn['mean_delta_pp']:+.2f} pp ({pn['positive_seeds']}/{pn['n']} seeds)."
        )
    if pb.get("n"):
        lines.append(
            f"pbCMA vs Bernoulli: {pb['mean_delta_pp']:+.2f} pp ({pb['positive_seeds']}/{pb['n']} seeds)."
        )
    if pr.get("n"):
        lines.append(
            f"pbCMA vs rint CMA: {pr['mean_delta_pp']:+.2f} pp ({pr['positive_seeds']}/{pr['n']} seeds)."
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

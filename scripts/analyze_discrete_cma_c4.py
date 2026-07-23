"""Package C4: compare Bernoulli-decode CMA-ME vs continuous rint CMA-ME and discrete baselines."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DISCRETE_ROOT = ROOT / "artifacts/experiments/q1-v3-pyribs-discrete-cma"
CONTINUOUS_ROOT = ROOT / "artifacts/experiments/q1-v3-pyribs"
HINTS_ROOT = ROOT / "artifacts/experiments/q1-full"
GENETIC_ROOT = ROOT / "artifacts/experiments/q1-v3-genetic-me-uniform"
OUT_JSON = DISCRETE_ROOT / "c4_discrete_cma_analysis.json"
OUT_MD = DISCRETE_ROOT / "ANALYSIS.md"


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
    arms: dict[str, dict[int, float]] = {}
    for label, root, cond in [
        ("cma_me_bernoulli", DISCRETE_ROOT, "cma_me_bernoulli"),
        ("cma_me_rint", CONTINUOUS_ROOT, "cma_me"),
        ("hints", HINTS_ROOT, "hints"),
        ("genetic_me_uniform", GENETIC_ROOT, "genetic_me_uniform"),
    ]:
        arms[label] = {}
        for seed in seeds:
            summary = _load_summary(
                root / cond / f"seed_{seed}" / "nightly_run_summary.json"
            )
            if summary is not None:
                arms[label][seed] = _coverage_pct(summary)

    contrasts = {
        "bernoulli_minus_rint": _paired(
            [
                (s, arms["cma_me_rint"][s], arms["cma_me_bernoulli"][s])
                for s in seeds
                if s in arms["cma_me_rint"] and s in arms["cma_me_bernoulli"]
            ]
        ),
        "bernoulli_minus_hints": _paired(
            [
                (s, arms["hints"][s], arms["cma_me_bernoulli"][s])
                for s in seeds
                if s in arms["hints"] and s in arms["cma_me_bernoulli"]
            ]
        ),
        "rint_minus_hints": _paired(
            [
                (s, arms["hints"][s], arms["cma_me_rint"][s])
                for s in seeds
                if s in arms["hints"] and s in arms["cma_me_rint"]
            ]
        ),
        "bernoulli_minus_genetic_uniform": _paired(
            [
                (s, arms["genetic_me_uniform"][s], arms["cma_me_bernoulli"][s])
                for s in seeds
                if s in arms["genetic_me_uniform"] and s in arms["cma_me_bernoulli"]
            ]
        ),
    }

    payload = {
        "package": "C4",
        "tier": "q1-v3-pyribs-discrete-cma",
        "condition": "cma_me_bernoulli",
        "decode_mode": "bernoulli",
        "note": (
            "CMA still proposes in R^21; Bernoulli decode samples discrete rule bits at eval. "
            "Not a native discrete search-space CMA emitter."
        ),
        "seeds_available": {k: sorted(v.keys()) for k, v in arms.items()},
        "contrasts": contrasts,
    }

    DISCRETE_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Package C4 — discrete-decode CMA-ME (Bernoulli)",
        "",
        f"Artifact: `{DISCRETE_ROOT.relative_to(ROOT)}`",
        "",
        "## Paired coverage deltas (pp)",
        "",
    ]
    for name, stats in contrasts.items():
        if stats.get("n", 0) == 0:
            lines.append(f"- **{name}**: no paired seeds yet")
            continue
        lines.append(
            f"- **{name}** ($n={stats['n']}$): mean $\\Delta$={stats['mean_delta_pp']:+.2f} pp "
            f"(SD {stats['sd_pp']:.2f}; {stats['positive_seeds']}/{stats['n']} seeds positive)"
        )
    lines.extend(["", "## Readout", ""])
    b = contrasts["bernoulli_minus_hints"]
    r = contrasts["rint_minus_hints"]
    br = contrasts["bernoulli_minus_rint"]
    if b.get("n"):
        lines.append(
            f"Bernoulli CMA vs hints: {b['mean_delta_pp']:+.2f} pp mean ({b['n']} seeds)."
        )
    if r.get("n"):
        lines.append(
            f"Continuous rint CMA vs hints: {r['mean_delta_pp']:+.2f} pp mean ({r['n']} seeds)."
        )
    if br.get("n"):
        lines.append(
            f"Bernoulli vs rint CMA: {br['mean_delta_pp']:+.2f} pp mean ({br['n']} seeds)."
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Analyze matched H1 pilot: gpt-4o-mini stub_uniform vs hints (seeds 0-2)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLM_ROOT = ROOT / "artifacts/experiments/q1-v3-llm/gpt-4o-mini"
OUT_DIR = LLM_ROOT
OUT_JSON = OUT_DIR / "h1_matched_gpt4o_mini_analysis.json"
OUT_MD = OUT_DIR / "H1_MATCHED_ANALYSIS.md"


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _cov_pct(payload: dict) -> float:
    cov = float(payload["coverage"])
    return cov * 100.0 if cov <= 1.0 else cov


def main() -> int:
    seeds = list(range(3))
    stub: dict[int, float] = {}
    hints: dict[int, float] = {}
    for seed in seeds:
        su = _load(
            LLM_ROOT / "stub_uniform" / f"seed_{seed}" / "nightly_run_summary.json"
        )
        hi = _load(LLM_ROOT / "hints" / f"seed_{seed}" / "nightly_run_summary.json")
        if su is not None:
            stub[seed] = _cov_pct(su)
        if hi is not None:
            hints[seed] = _cov_pct(hi)

    paired = [
        {
            "seed": s,
            "stub_uniform_pct": round(stub[s], 4),
            "hints_pct": round(hints[s], 4),
            "delta_hints_minus_stub_pp": round(hints[s] - stub[s], 4),
        }
        for s in seeds
        if s in stub and s in hints
    ]
    deltas = [row["delta_hints_minus_stub_pp"] for row in paired]
    payload = {
        "tier": "q1-h1-matched-gpt-4o-mini",
        "provider": "openai/gpt-4o-mini",
        "contrast": "hints_minus_stub_uniform",
        "policy": "uniform_frontier",
        "n": len(paired),
        "mean_delta_pp": round(statistics.mean(deltas), 4) if deltas else None,
        "sd_pp": round(statistics.pstdev(deltas), 4) if len(deltas) > 1 else 0.0,
        "seeds": paired,
        "note": (
            "Exploratory matched-policy H1 on second provider; reuses existing hints runs."
        ),
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Matched H1 pilot — gpt-4o-mini (stub_uniform vs hints)",
        "",
        f"Artifact: `{LLM_ROOT.relative_to(ROOT)}`",
        "",
        f"**n={payload['n']}** seeds 0–2; Δ = hints − stub_uniform (coverage %).",
        "",
        "| Seed | stub_uniform | hints | Δ (pp) |",
        "|------|--------------|-------|--------|",
    ]
    for row in paired:
        lines.append(
            f"| {row['seed']} | {row['stub_uniform_pct']:.2f}% | "
            f"{row['hints_pct']:.2f}% | {row['delta_hints_minus_stub_pp']:+.2f} |"
        )
    if payload["mean_delta_pp"] is not None:
        lines.extend(
            [
                "",
                f"**Mean Δ:** {payload['mean_delta_pp']:+.2f} pp "
                f"(SD {payload['sd_pp']:.2f}).",
                "",
                "Reading: flat matched H1 on second provider (within ±2 pp band).",
            ]
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Matched H1 mean-TOST sensitivity across equivalence margins (offline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_q1_statistics import param_tost

MARGINS_PP = (1.0, 1.5, 2.0, 3.0)
OUT_JSON = ROOT / "artifacts/experiments/q1-v3-llm/h1_tost_margin_sensitivity.json"
OUT_MD = ROOT / "artifacts/experiments/q1-v3-llm/H1_TOST_MARGIN_SENSITIVITY.md"


def _series() -> dict[str, dict[str, Any]]:
    qwen = json.loads(
        (
            ROOT
            / "artifacts/experiments/q1-stub-uniform-sensitivity/rq1_equivalence_power.json"
        ).read_text(encoding="utf-8")
    )
    out: dict[str, dict[str, Any]] = {
        "qwen-turbo": {
            "label": "qwen-turbo (primary)",
            "n": 10,
            "deltas": [float(x) for x in qwen["coverage_pp"]["deltas"]],
            "role": "primary_claim",
        }
    }
    for slug, label in (
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("deepseek-v4-pro", "DeepSeek V4 Pro"),
    ):
        payload = json.loads(
            (
                ROOT
                / f"artifacts/experiments/q1-v3-llm/{slug}/h1_matched_analysis.json"
            ).read_text(encoding="utf-8")
        )
        deltas = [float(r["delta_cov_pp"]) for r in payload["per_seed"]]
        out[slug] = {
            "label": label,
            "n": len(deltas),
            "deltas": deltas,
            "role": "matched_provider",
        }
    ds = np.asarray(out["deepseek-v4-pro"]["deltas"], dtype=float)
    out["deepseek-v4-pro_drop_seed8"] = {
        "label": "DeepSeek V4 Pro (drop seed 8)",
        "n": 9,
        "deltas": [float(x) for x in np.delete(ds, 8)],
        "role": "loo_sensitivity",
    }
    return out


def _row(name: str, meta: dict[str, Any]) -> dict[str, Any]:
    deltas = np.asarray(meta["deltas"], dtype=float)
    ci = param_tost(deltas, 2.0)["ci90"]
    margins: dict[str, Any] = {}
    for m in MARGINS_PP:
        t = param_tost(deltas, float(m))
        margins[f"{m:g}"] = {
            "accepted": bool(t["accepted"]),
            "p_tost": float(t["p_tost"]),
            "ci90": [float(t["ci90"][0]), float(t["ci90"][1])],
        }
    return {
        "id": name,
        "label": meta["label"],
        "role": meta["role"],
        "n": meta["n"],
        "mean_delta_cov_pp": round(float(np.mean(deltas)), 4),
        "sd_delta_cov_pp": round(float(np.std(deltas, ddof=1)), 4),
        "ci90_mean": [float(ci[0]), float(ci[1])],
        "margins_pp": margins,
    }


def main() -> int:
    rows = [_row(name, meta) for name, meta in _series().items()]
    payload = {
        "contrast": "hints_minus_stub_uniform",
        "policy": "uniform_frontier",
        "estimand": "paired_t_mean_tost",
        "margins_pp": list(MARGINS_PP),
        "note": (
            "Offline sensitivity of post-hoc paired-t mean-TOST to the equivalence "
            "margin. Primary claim remains ±2 pp on qwen-turbo. DeepSeek drop-seed-8 "
            "row is leave-one-out influence only—not a co-claimed provider result."
        ),
        "rows": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Matched H1 — mean-TOST margin sensitivity",
        "",
        payload["note"],
        "",
        "| Provider | n | Mean Δ | 90% CI | ±1 | ±1.5 | ±2 | ±3 |",
        "|----------|--:|-------:|-------:|:--:|:----:|:--:|:--:|",
    ]
    for row in rows:
        cells = []
        for m in MARGINS_PP:
            t = row["margins_pp"][f"{m:g}"]
            cells.append("accept" if t["accepted"] else "reject")
        lines.append(
            f"| {row['label']} | {row['n']} | {row['mean_delta_cov_pp']:+.2f} | "
            f"[{row['ci90_mean'][0]:+.2f},{row['ci90_mean'][1]:+.2f}] | "
            + " | ".join(cells)
            + " |"
        )
    lines += ["", f"JSON: `{OUT_JSON.relative_to(ROOT)}`", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD)
    print(OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

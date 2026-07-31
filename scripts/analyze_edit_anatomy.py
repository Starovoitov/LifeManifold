#!/usr/bin/env python3
"""Computational edit-anatomy case study: LLM vs genetic elites (descriptive).

Compares parent→child WorldSpec diffs among accepted archive elites with a
resolvable parent inside the same run (default: q1-full/hints, seeds 0–9).
Does not claim a QD win or human preference — only that LLM edits are more
compact / named-field-local than genetic bit-flip+nudge on this genome.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "artifacts/experiments/q1-full/hints"
OUT_DIR = ROOT / "artifacts/experiments/edit-anatomy"
SCALAR_KEYS = ("noise", "resource_regen", "predation")
SCALAR_EPS = 1e-6


def _rule_bits(spec: dict[str, Any]) -> tuple[list[int], list[int]]:
    birth = [0] * 9
    survival = [0] * 9
    for x in spec.get("birth") or []:
        i = int(x)
        if 0 <= i <= 8:
            birth[i] = 1
    for x in spec.get("survival") or []:
        i = int(x)
        if 0 <= i <= 8:
            survival[i] = 1
    return birth, survival


def _set_ops(child: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    cb, cs = _rule_bits(child)
    pb, ps = _rule_bits(parent)
    birth_add = [i for i in range(9) if cb[i] and not pb[i]]
    birth_rem = [i for i in range(9) if pb[i] and not cb[i]]
    surv_add = [i for i in range(9) if cs[i] and not ps[i]]
    surv_rem = [i for i in range(9) if ps[i] and not cs[i]]
    rule_ham = len(birth_add) + len(birth_rem) + len(surv_add) + len(surv_rem)
    scalar_delta: dict[str, float] = {}
    scalar_l1 = 0.0
    for key in SCALAR_KEYS:
        d = float(child.get(key, 0.0)) - float(parent.get(key, 0.0))
        if abs(d) > SCALAR_EPS:
            scalar_delta[key] = d
            scalar_l1 += abs(d)
    return {
        "rule_hamming": rule_ham,
        "birth_add": birth_add,
        "birth_rem": birth_rem,
        "survival_add": surv_add,
        "survival_rem": surv_rem,
        "n_set_ops": rule_ham,
        "scalar_changed": len(scalar_delta),
        "scalar_l1": scalar_l1,
        "scalar_delta": scalar_delta,
    }


def _phrase(ops: dict[str, Any]) -> str:
    parts: list[str] = []
    if ops["birth_add"]:
        parts.append("add birth " + ",".join(map(str, ops["birth_add"])))
    if ops["birth_rem"]:
        parts.append("drop birth " + ",".join(map(str, ops["birth_rem"])))
    if ops["survival_add"]:
        parts.append("add survival " + ",".join(map(str, ops["survival_add"])))
    if ops["survival_rem"]:
        parts.append("drop survival " + ",".join(map(str, ops["survival_rem"])))
    for key, d in ops["scalar_delta"].items():
        if abs(d) < 5e-4:
            continue
        parts.append(f"{key}{d:+.3f}")
    return "; ".join(parts) if parts else "(no-op)"


def _load_archive(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for line in path.open():
        row = json.loads(line)
        eid = str(row["metadata"]["id"])
        by_id[eid] = row
        rows.append(row)
    return by_id, rows


def _collect_edits(
    by_id: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    emitter: str,
) -> list[dict[str, Any]]:
    edits: list[dict[str, Any]] = []
    for row in rows:
        meta = row["metadata"]
        if str(meta.get("emitter_type") or "") != emitter:
            continue
        pid = meta.get("parent_id")
        if not pid or str(pid) not in by_id:
            continue
        parent = by_id[str(pid)]
        ops = _set_ops(row["world_spec"], parent["world_spec"])
        if ops["rule_hamming"] == 0 and ops["scalar_changed"] == 0:
            continue  # identical after eps; skip
        edits.append(
            {
                "child_id": meta["id"],
                "parent_id": pid,
                "fitness": float(row.get("fitness") or 0.0),
                "parent_fitness": float(parent.get("fitness") or 0.0),
                "stability": float((row.get("measures") or {}).get("stability") or 0.0),
                "parent_stability": float(
                    (parent.get("measures") or {}).get("stability") or 0.0
                ),
                **ops,
                "phrase": _phrase(ops),
            }
        )
    return edits


def _summarize(edits: list[dict[str, Any]]) -> dict[str, Any]:
    if not edits:
        return {"n": 0}
    ham = np.asarray([e["rule_hamming"] for e in edits], dtype=float)
    sc = np.asarray([e["scalar_changed"] for e in edits], dtype=float)
    l1 = np.asarray([e["scalar_l1"] for e in edits], dtype=float)
    return {
        "n": int(len(edits)),
        "mean_rule_hamming": round(float(np.mean(ham)), 4),
        "median_rule_hamming": round(float(np.median(ham)), 4),
        "frac_rule_ham_le1": round(float(np.mean(ham <= 1)), 4),
        "frac_rule_ham_le2": round(float(np.mean(ham <= 2)), 4),
        "frac_scalar_only": round(float(np.mean(ham == 0)), 4),
        "mean_scalar_changed": round(float(np.mean(sc)), 4),
        "mean_scalar_l1": round(float(np.mean(l1)), 4),
        "median_scalar_l1": round(float(np.median(l1)), 4),
    }


def _pick_examples(edits: list[dict[str, Any]], k: int = 5) -> list[dict[str, Any]]:
    """Prefer compact rule edits with a short phrase."""
    ranked = sorted(
        edits,
        key=lambda e: (
            e["rule_hamming"],
            e["scalar_l1"],
            -abs(e["stability"] - e["parent_stability"]),
        ),
    )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for e in ranked:
        if e["rule_hamming"] == 0:
            continue
        key = e["phrase"]
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "rule_hamming": e["rule_hamming"],
                "scalar_l1": round(e["scalar_l1"], 4),
                "stability_parent_child": [
                    round(e["parent_stability"], 3),
                    round(e["stability"], 3),
                ],
                "phrase": e["phrase"],
            }
        )
        if len(out) >= k:
            break
    return out


def analyze_seed(archive: Path) -> dict[str, Any]:
    by_id, rows = _load_archive(archive)
    llm = _collect_edits(by_id, rows, emitter="llm")
    genetic = _collect_edits(by_id, rows, emitter="genetic")
    return {
        "archive": str(archive),
        "n_elites": len(rows),
        "llm": _summarize(llm),
        "genetic": _summarize(genetic),
        "llm_hammings": [e["rule_hamming"] for e in llm],
        "genetic_hammings": [e["rule_hamming"] for e in genetic],
        "llm_scalar_l1": [e["scalar_l1"] for e in llm],
        "genetic_scalar_l1": [e["scalar_l1"] for e in genetic],
        "llm_examples": _pick_examples(llm),
    }


def _pool_mean_sd(
    per_seed: list[dict[str, Any]], arm: str, key: str
) -> dict[str, float]:
    vals = [float(s[arm][key]) for s in per_seed if s[arm].get("n", 0) > 0]
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "sd": round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else 0.0,
        "n_seeds": len(arr),
    }


def write_figure(
    per_seed: list[dict[str, Any]],
    path: Path,
) -> None:
    llm = np.concatenate([np.asarray(s["llm_hammings"], dtype=float) for s in per_seed])
    gen = np.concatenate(
        [np.asarray(s["genetic_hammings"], dtype=float) for s in per_seed]
    )
    bins = np.arange(-0.5, 19.5, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    for ax, data, title, color in (
        (axes[0], llm, "LLM elites", "#2a6f97"),
        (axes[1], gen, "Genetic elites", "#8b5e34"),
    ):
        ax.hist(
            data, bins=bins, density=True, color=color, edgecolor="white", linewidth=0.4
        )
        ax.axvline(float(np.mean(data)), color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Rule Hamming (0–18)")
        ax.set_title(f"{title}\nmean={np.mean(data):.2f}, med={np.median(data):.0f}")
        ax.set_xlim(-0.5, 12.5)
    axes[0].set_ylabel("Density")
    fig.suptitle(
        "Edit anatomy on q1-full/hints (accepted elites, resolvable parent)",
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    p = payload["pooled"]
    t = payload["tests"]
    lines = [
        "# Edit anatomy case study (computational)",
        "",
        f"**Tier:** `{payload['tier']}` · **Seeds:** {payload['seeds']} · "
        f"**n elites with resolvable parent+diff:** "
        f"LLM {payload['totals']['llm']} / genetic {payload['totals']['genetic']}",
        "",
        "Descriptive only — not a confirmatory Holm family; not a user preference study.",
        "",
        "## Pooled per-seed means ($n=10$)",
        "",
        "| Metric | LLM | Genetic |",
        "|--------|----:|--------:|",
        (
            f"| Mean rule Hamming | "
            f"**{p['llm']['mean_rule_hamming']['mean']:.2f}** ± "
            f"{p['llm']['mean_rule_hamming']['sd']:.2f} | "
            f"{p['genetic']['mean_rule_hamming']['mean']:.2f} ± "
            f"{p['genetic']['mean_rule_hamming']['sd']:.2f} |"
        ),
        (
            f"| Median rule Hamming (mean of seed medians) | "
            f"**{p['llm']['median_rule_hamming']['mean']:.2f}** | "
            f"{p['genetic']['median_rule_hamming']['mean']:.2f} |"
        ),
        (
            f"| Frac rule Hamming ≤ 1 | "
            f"**{100*p['llm']['frac_rule_ham_le1']['mean']:.1f}%** | "
            f"{100*p['genetic']['frac_rule_ham_le1']['mean']:.1f}% |"
        ),
        (
            f"| Frac rule Hamming ≤ 2 | "
            f"**{100*p['llm']['frac_rule_ham_le2']['mean']:.1f}%** | "
            f"{100*p['genetic']['frac_rule_ham_le2']['mean']:.1f}% |"
        ),
        (
            f"| Frac scalar-only (0 rule bits) | "
            f"**{100*p['llm']['frac_scalar_only']['mean']:.1f}%** | "
            f"{100*p['genetic']['frac_scalar_only']['mean']:.1f}% |"
        ),
        (
            f"| Mean scalar L1 | "
            f"**{p['llm']['mean_scalar_l1']['mean']:.3f}** ± "
            f"{p['llm']['mean_scalar_l1']['sd']:.3f} | "
            f"{p['genetic']['mean_scalar_l1']['mean']:.3f} ± "
            f"{p['genetic']['mean_scalar_l1']['sd']:.3f} |"
        ),
        "",
        "## Paired tests (seed-level means, LLM − genetic)",
        "",
        (
            f"- Mean rule Hamming: Δ = {t['delta_mean_rule_hamming']:+.3f}; "
            f"Wilcoxon one-sided less p = {t['wilcoxon_hamming_less_p']:.4g}"
        ),
        (
            f"- Mean scalar L1: Δ = {t['delta_mean_scalar_l1']:+.3f}; "
            f"Wilcoxon one-sided less p = {t['wilcoxon_scalar_l1_less_p']:.4g}"
        ),
        "",
        "## Example LLM phrases (seed 4, compact)",
        "",
    ]
    for ex in payload.get("seed4_examples", []):
        lines.append(
            f"- Hamming {ex['rule_hamming']}, L1={ex['scalar_l1']}: `{ex['phrase']}`"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "On the same `hints` runs, accepted LLM elites differ from their parents by "
        "**fewer rule-bit flips** and **smaller scalar L1** than accepted genetic elites. "
        "Edits remain named-field set operations (birth/survival neighbour counts + scalars), "
        "supporting a *structured / human-readable variation* niche without claiming archive "
        "coverage superiority.",
        "",
        f"Figure: `{payload['figure']}`",
        "",
        "## Regenerate",
        "",
        "```bash",
        "uv run python scripts/analyze_edit_anatomy.py",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    per_seed: list[dict[str, Any]] = []
    for seed in args.seeds:
        archive = args.root / f"seed_{seed}" / "map_elites_archive.jsonl"
        if not archive.exists():
            raise FileNotFoundError(archive)
        per_seed.append({"seed": seed, **analyze_seed(archive)})

    metrics = [
        "mean_rule_hamming",
        "median_rule_hamming",
        "frac_rule_ham_le1",
        "frac_rule_ham_le2",
        "frac_scalar_only",
        "mean_scalar_l1",
    ]
    pooled = {
        arm: {m: _pool_mean_sd(per_seed, arm, m) for m in metrics}
        for arm in ("llm", "genetic")
    }

    ham_llm = np.asarray([s["llm"]["mean_rule_hamming"] for s in per_seed], dtype=float)
    ham_gen = np.asarray(
        [s["genetic"]["mean_rule_hamming"] for s in per_seed], dtype=float
    )
    l1_llm = np.asarray([s["llm"]["mean_scalar_l1"] for s in per_seed], dtype=float)
    l1_gen = np.asarray([s["genetic"]["mean_scalar_l1"] for s in per_seed], dtype=float)
    w_ham = cast(Any, wilcoxon(ham_llm - ham_gen, alternative="less"))
    w_l1 = cast(Any, wilcoxon(l1_llm - l1_gen, alternative="less"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = args.out_dir / "edit_anatomy_hamming.png"
    write_figure(per_seed, fig_path)

    # Also copy into manuscript figures if present.
    ms_fig = ROOT / "artifacts/manuscript/figures/fig_edit_anatomy_hamming.pdf"
    if (args.out_dir / "edit_anatomy_hamming.pdf").exists():
        ms_fig.parent.mkdir(parents=True, exist_ok=True)
        ms_fig.write_bytes((args.out_dir / "edit_anatomy_hamming.pdf").read_bytes())

    seed4 = next((s for s in per_seed if s["seed"] == 4), per_seed[0])
    payload: dict[str, Any] = {
        "family": "edit-anatomy-descriptive",
        "confirmatory": False,
        "tier": (
            str(args.root.relative_to(ROOT))
            if args.root.is_relative_to(ROOT)
            else str(args.root)
        ),
        "seeds": args.seeds,
        "note": (
            "Parent→child diffs among accepted elites with resolvable parent in the "
            "same archive; descriptive compactness of named-field edits."
        ),
        "totals": {
            "llm": int(sum(s["llm"]["n"] for s in per_seed)),
            "genetic": int(sum(s["genetic"]["n"] for s in per_seed)),
        },
        "pooled": pooled,
        "tests": {
            "delta_mean_rule_hamming": round(float(np.mean(ham_llm - ham_gen)), 4),
            "wilcoxon_hamming_less_p": float(w_ham.pvalue),
            "delta_mean_scalar_l1": round(float(np.mean(l1_llm - l1_gen)), 4),
            "wilcoxon_scalar_l1_less_p": float(w_l1.pvalue),
        },
        "per_seed": [
            {
                "seed": s["seed"],
                "llm": s["llm"],
                "genetic": s["genetic"],
            }
            for s in per_seed
        ],
        "seed4_examples": seed4.get("llm_examples", []),
        "figure": str(fig_path.relative_to(ROOT)),
    }

    # Drop bulky per-edit arrays from JSON.
    json_path = args.out_dir / "edit_anatomy.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path = args.out_dir / "ANALYSIS.md"
    write_markdown(payload, md_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

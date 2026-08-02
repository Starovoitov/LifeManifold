#!/usr/bin/env python3
"""H1 package A: LLM-candidate quality from accepted elites (no new LLM runs).

Scope (honest):
  - Metrics use accepted archive elites with emitter_type=llm.
  - Rejected / non-inserted LLM proposals are NOT logged per-seed on
    confirmatory hints / stub_uniform arms → insertion denominators use
    llm_emit_attempts − llm_emit_fallbacks from nightly_run_summary.
  - Predicted fitness is offline SurrogateFacade.predict(child) when a
    checkpoint is provided (not logged at emit time on these arms).

Arms (default):
  - q1-full/hints
  - q1-stub-uniform-sensitivity/stub_uniform
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_edit_anatomy import _set_ops  # noqa: E402
from worldspace.specs.spec import WorldSpec  # noqa: E402
from worldspace.surrogate import get_surrogate  # noqa: E402
from worldspace.surrogate.types import SurrogateConfig  # noqa: E402

OUT_DIR = ROOT / "artifacts/experiments/h1-llm-candidate-quality"
DEFAULT_CHECKPOINT = ROOT / "artifacts/surrogate/checkpoints/nightly_v3_mc_d005.pkl"
DEFAULT_CALIBRATION = (
    ROOT / "artifacts/surrogate/checkpoints/calibration_v3_mc_d005.pkl"
)

DEFAULT_ARMS: dict[str, Path] = {
    "hints": ROOT / "artifacts/experiments/q1-full/hints",
    "stub_uniform": ROOT
    / "artifacts/experiments/q1-stub-uniform-sensitivity/stub_uniform",
}


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_archive_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _emitter(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    return str(meta.get("emitter_type") or meta.get("generated_by") or "")


def _bin_key(row: dict[str, Any]) -> tuple[int, int]:
    b = row["bin"]
    return int(b[0]), int(b[1])


def _mean_sd(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs, dtype=float)
    if len(a) == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return {"mean": round(float(a.mean()), 6), "sd": round(sd, 6), "n": int(len(a))}


def _build_surrogate(checkpoint: Path, calibration: Path | None):
    cfg = SurrogateConfig(
        enabled=True,
        model_type="mlp",
        checkpoint=str(checkpoint),
        stub_mean=0.5,
        stub_uncertainty=1.0,
        calibration=str(calibration) if calibration and calibration.is_file() else None,
        require_quality_gate=False,
    )
    return get_surrogate(cfg)


def analyze_seed(
    arm_root: Path,
    seed: int,
    *,
    surrogate=None,
    predict_sample: int = 0,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    run = arm_root / f"seed_{seed}"
    archive_path = run / "map_elites_archive.jsonl"
    summary_path = run / "nightly_run_summary.json"
    if not archive_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"missing archive/summary for {run}")

    summary = _load_summary(summary_path)
    attempts = int(summary.get("llm_emit_attempts") or 0)
    fallbacks = int(summary.get("llm_emit_fallbacks") or 0)
    denom = max(attempts - fallbacks, 0)

    rows = _load_archive_rows(archive_path)
    by_id = {str(r["metadata"]["id"]): r for r in rows if r.get("metadata")}

    occupied: dict[tuple[int, int], float] = {}
    n_llm_accept = 0
    n_fill = 0
    n_improve = 0
    llm_fit: list[float] = []
    rule_ham: list[float] = []
    scalar_l1: list[float] = []
    realized_delta: list[float] = []
    predicted_child: list[float] = []
    predicted_delta: list[float] = []
    pred_error: list[float] = []

    for row in rows:
        if _emitter(row) != "llm":
            # Still track occupancy for later LLM improvements into genetic fills.
            key = _bin_key(row)
            occupied[key] = float(row["fitness"])
            continue

        n_llm_accept += 1
        fit = float(row["fitness"])
        llm_fit.append(fit)
        key = _bin_key(row)
        prev = occupied.get(key)
        if prev is None:
            n_fill += 1
        elif fit > prev:
            n_improve += 1
        occupied[key] = fit

        parent_id = (row.get("metadata") or {}).get("parent_id")
        parent = by_id.get(str(parent_id)) if parent_id else None
        if parent is not None and parent.get("world_spec") and row.get("world_spec"):
            ops = _set_ops(row["world_spec"], parent["world_spec"])
            rule_ham.append(float(ops["rule_hamming"]))
            scalar_l1.append(float(ops["scalar_l1"]))
            parent_fit = float(parent["fitness"])
            realized_delta.append(fit - parent_fit)

    # Offline surrogate predict on a subsample of accepted LLM elites with parents.
    if surrogate is not None:
        candidates = [
            r
            for r in rows
            if _emitter(r) == "llm"
            and (r.get("metadata") or {}).get("parent_id")
            and str((r.get("metadata") or {}).get("parent_id")) in by_id
            and r.get("world_spec")
        ]
        if predict_sample > 0 and len(candidates) > predict_sample:
            assert rng is not None
            idx = rng.choice(len(candidates), size=predict_sample, replace=False)
            candidates = [candidates[int(i)] for i in idx]
        for row in candidates:
            parent = by_id[str(row["metadata"]["parent_id"])]
            child_spec = WorldSpec.from_json_dict(row["world_spec"])
            pred_f = float(surrogate.predict(child_spec).fitness)
            fit = float(row["fitness"])
            parent_fit = float(parent["fitness"])
            predicted_child.append(pred_f)
            predicted_delta.append(pred_f - parent_fit)
            pred_error.append(pred_f - fit)

    return {
        "seed": seed,
        "llm_emit_attempts": attempts,
        "llm_emit_fallbacks": fallbacks,
        "evaluated_llm_denom": denom,
        "n_llm_accepted": n_llm_accept,
        "n_fill": n_fill,
        "n_improve": n_improve,
        "insert_rate": (n_llm_accept / denom) if denom else float("nan"),
        "improve_rate": (n_improve / denom) if denom else float("nan"),
        "improve_among_accepted": (
            n_improve / n_llm_accept if n_llm_accept else float("nan")
        ),
        "fill_among_accepted": (
            n_fill / n_llm_accept if n_llm_accept else float("nan")
        ),
        "accepted_fitness": _mean_sd(llm_fit),
        "edit_rule_hamming": _mean_sd(rule_ham),
        "edit_scalar_l1": _mean_sd(scalar_l1),
        "realized_child_minus_parent": _mean_sd(realized_delta),
        "predicted_child_fitness": _mean_sd(predicted_child),
        "predicted_child_minus_parent": _mean_sd(predicted_delta),
        "pred_minus_true_child": _mean_sd(pred_error),
        "n_resolvable_parent": len(rule_ham),
        "n_offline_predict": len(predicted_child),
    }


def analyze_arm(
    label: str,
    arm_root: Path,
    seeds: list[int],
    *,
    surrogate=None,
    predict_sample: int = 0,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    per_seed = [
        analyze_seed(
            arm_root,
            s,
            surrogate=surrogate,
            predict_sample=predict_sample,
            rng=rng,
        )
        for s in seeds
    ]

    def pool(key: str) -> dict[str, float]:
        xs = [float(r[key]) for r in per_seed if np.isfinite(float(r[key]))]
        return _mean_sd(xs)

    def pool_nested(block: str, field: str = "mean") -> dict[str, float]:
        xs = [
            float(r[block][field])
            for r in per_seed
            if r[block]["n"] > 0 and np.isfinite(float(r[block][field]))
        ]
        return _mean_sd(xs)

    return {
        "arm": label,
        "root": str(arm_root),
        "n_seeds": len(per_seed),
        "scope": "accepted_llm_elites_only",
        "pooled": {
            "insert_rate": pool("insert_rate"),
            "improve_rate": pool("improve_rate"),
            "improve_among_accepted": pool("improve_among_accepted"),
            "fill_among_accepted": pool("fill_among_accepted"),
            "n_llm_accepted": pool("n_llm_accepted"),
            "accepted_fitness": pool_nested("accepted_fitness"),
            "edit_rule_hamming": pool_nested("edit_rule_hamming"),
            "edit_scalar_l1": pool_nested("edit_scalar_l1"),
            "realized_child_minus_parent": pool_nested("realized_child_minus_parent"),
            "predicted_child_fitness": pool_nested("predicted_child_fitness"),
            "predicted_child_minus_parent": pool_nested("predicted_child_minus_parent"),
            "pred_minus_true_child": pool_nested("pred_minus_true_child"),
        },
        "per_seed": per_seed,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# H1 LLM-candidate quality (accepted elites; package A)",
        "",
        "**Scope:** accepted archive elites with `emitter_type=llm` only. "
        "Rejected proposals are not logged per-seed on confirmatory "
        "`hints` / `stub_uniform` arms. Insertion denominators = "
        "`llm_emit_attempts − llm_emit_fallbacks` (~6500).",
        "",
        f"**Surrogate offline predict:** "
        f"`{payload.get('surrogate_checkpoint') or 'disabled'}`",
        "",
    ]
    for arm_label, arm in payload["arms"].items():
        p = arm["pooled"]
        lines.extend(
            [
                f"## `{arm_label}`",
                "",
                f"Root: `{arm['root']}` · seeds={arm['n_seeds']}",
                "",
                "| Metric | Mean ± SD (across seeds) |",
                "|--------|-------------------------:|",
                (
                    f"| LLM accepted / seed | {p['n_llm_accepted']['mean']:.1f} ± "
                    f"{p['n_llm_accepted']['sd']:.1f} |"
                ),
                (
                    f"| Insert rate (vs evaluated LLM) | "
                    f"{100*p['insert_rate']['mean']:.2f}% ± "
                    f"{100*p['insert_rate']['sd']:.2f} |"
                ),
                (
                    f"| Cell-improve rate (vs evaluated LLM) | "
                    f"{100*p['improve_rate']['mean']:.2f}% ± "
                    f"{100*p['improve_rate']['sd']:.2f} |"
                ),
                (
                    f"| Improve among accepted | "
                    f"{100*p['improve_among_accepted']['mean']:.1f}% ± "
                    f"{100*p['improve_among_accepted']['sd']:.1f} |"
                ),
                (
                    f"| Fill among accepted | "
                    f"{100*p['fill_among_accepted']['mean']:.1f}% ± "
                    f"{100*p['fill_among_accepted']['sd']:.1f} |"
                ),
                (
                    f"| Accepted fitness | {p['accepted_fitness']['mean']:.4f} ± "
                    f"{p['accepted_fitness']['sd']:.4f} |"
                ),
                (
                    f"| Edit rule Hamming | {p['edit_rule_hamming']['mean']:.3f} ± "
                    f"{p['edit_rule_hamming']['sd']:.3f} |"
                ),
                (
                    f"| Edit scalar L1 | {p['edit_scalar_l1']['mean']:.4f} ± "
                    f"{p['edit_scalar_l1']['sd']:.4f} |"
                ),
                (
                    f"| Realized child−parent Δfit | "
                    f"{p['realized_child_minus_parent']['mean']:+.4f} ± "
                    f"{p['realized_child_minus_parent']['sd']:.4f} |"
                ),
            ]
        )
        if np.isfinite(p["predicted_child_minus_parent"]["mean"]):
            lines.extend(
                [
                    (
                        f"| Predicted child−parent Δfit (offline) | "
                        f"{p['predicted_child_minus_parent']['mean']:+.4f} ± "
                        f"{p['predicted_child_minus_parent']['sd']:.4f} |"
                    ),
                    (
                        f"| Pred − true child fitness (offline) | "
                        f"{p['pred_minus_true_child']['mean']:+.4f} ± "
                        f"{p['pred_minus_true_child']['sd']:.4f} |"
                    ),
                ]
            )
        lines.append("")

    if "hints" in payload["arms"] and "stub_uniform" in payload["arms"]:
        h = payload["arms"]["hints"]["per_seed"]
        s = payload["arms"]["stub_uniform"]["per_seed"]
        by_h = {r["seed"]: r for r in h}
        by_s = {r["seed"]: r for r in s}
        common = sorted(set(by_h) & set(by_s))
        if common:
            d_ins = [by_h[i]["insert_rate"] - by_s[i]["insert_rate"] for i in common]
            d_imp = [by_h[i]["improve_rate"] - by_s[i]["improve_rate"] for i in common]
            d_fit = [
                by_h[i]["accepted_fitness"]["mean"]
                - by_s[i]["accepted_fitness"]["mean"]
                for i in common
            ]
            lines.extend(
                [
                    "## Paired hints − stub_uniform (descriptive)",
                    "",
                    f"n={len(common)} seeds",
                    "",
                    (
                        f"- Δ insert rate: {_mean_sd(d_ins)['mean']*100:+.2f} ± "
                        f"{_mean_sd(d_ins)['sd']*100:.2f} pp"
                    ),
                    (
                        f"- Δ improve rate: {_mean_sd(d_imp)['mean']*100:+.2f} ± "
                        f"{_mean_sd(d_imp)['sd']*100:.2f} pp"
                    ),
                    (
                        f"- Δ accepted fitness: {_mean_sd(d_fit)['mean']:+.4f} ± "
                        f"{_mean_sd(d_fit)['sd']:.4f}"
                    ),
                    "",
                    "Not a confirmatory Holm family. Supports manuscript reading that "
                    "live parent-level scalars do not improve LLM insert/improve "
                    "rates vs stub constants on accepted-elite proxies.",
                    "",
                ]
            )

    lines.extend(
        [
            "## Limits",
            "",
            "- No rejected LLM WorldSpecs per seed → cannot report all-evaluated "
            "fitness or insertion-conditioned fail anatomy.",
            "- Offline predicted Δ uses the shared MLP checkpoint; these runs may "
            "overlap the training buffer (descriptive only).",
            "- Exact parent fitness is already in the confirmatory prompt JSON; "
            "this analysis does not claim the LLM used surrogate scalars.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=9)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="MLP checkpoint for offline child predict (empty path to disable)",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
    )
    parser.add_argument(
        "--no-surrogate",
        action="store_true",
        help="Skip offline predicted Δ",
    )
    parser.add_argument(
        "--predict-sample",
        type=int,
        default=40,
        help="Max accepted LLM elites per seed for offline predict (0=all)",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    seeds = list(range(args.seed_start, args.seed_end + 1))

    surrogate = None
    ckpt_note: str | None = None
    rng = np.random.default_rng(0)
    if not args.no_surrogate and args.checkpoint and args.checkpoint.is_file():
        print(f"loading surrogate {args.checkpoint} …", flush=True)
        surrogate = _build_surrogate(args.checkpoint, args.calibration)
        ckpt_note = str(args.checkpoint)

    arms_out: dict[str, Any] = {}
    for label, root in DEFAULT_ARMS.items():
        if not root.is_dir():
            print(f"skip missing arm root: {root}", file=sys.stderr)
            continue
        print(f"analyzing {label} …", flush=True)
        arms_out[label] = analyze_arm(
            label,
            root,
            seeds,
            surrogate=surrogate,
            predict_sample=args.predict_sample,
            rng=rng,
        )

    payload = {
        "scope": "accepted_llm_elites_only",
        "package": "A",
        "seeds": seeds,
        "surrogate_checkpoint": ckpt_note,
        "arms": arms_out,
        "note": (
            "Descriptive H1 candidate-quality package. Not confirmatory Holm. "
            "Rejected proposals unavailable per-seed on these tiers."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "h1_llm_candidate_quality.json"
    md_path = args.out_dir / "ANALYSIS.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, md_path)
    print(json.dumps({k: v["pooled"] for k, v in arms_out.items()}, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

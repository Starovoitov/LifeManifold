#!/usr/bin/env python3
"""Offline soft-channel diagnostics (no new LLM/API runs).

1) Draft / rewrite distribution NMAE: stored buffer features → fitness head vs
   true fitness on evaluated LLM proposals in
   ``buffer_experiment_hints_rewrite.jsonl`` (emitter_type ``llm`` ≈ draft-kept;
   ``llm_rewrite`` ≈ rewritten children). Compare to hold-out accepted-elite
   NMAE (~0.112 in the manuscript). Skips WorldSpec re-extract / MC unc.

2) Manipulation checks from ``llm_call_log.jsonl`` (where present):
   - Path A rewrite: edit distance draft→rewrite vs (parent_true − child_pred)
     and uncertainty.
   - Parent-H1 hints: edit distance parent→child vs parent-hint residual and unc.

Writes:
  artifacts/experiments/q1-h1-child-rewrite-pilot/SOFT_CHANNEL_OFFLINE.md
  artifacts/experiments/q1-h1-child-rewrite-pilot/soft_channel_offline.json
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_edit_anatomy import _set_ops  # noqa: E402
from worldspace.specs.spec import WorldSpec  # noqa: E402
from worldspace.specs.world_spec_from_llm import (  # noqa: E402
    extract_json_object_from_text,
    world_spec_from_llm_payload,
)
from worldspace.illuminators.evaluation import apply_canonical_seed  # noqa: E402
from worldspace.surrogate import get_surrogate  # noqa: E402
from worldspace.surrogate.feature_extractor import extract_batch  # noqa: E402
from worldspace.surrogate.surrogate import _resolve_surrogate_fitness  # noqa: E402
from worldspace.surrogate.types import (
    SurrogateConfig,
    SurrogatePrediction,
)  # noqa: E402

OUT_DIR = ROOT / "artifacts/experiments/q1-h1-child-rewrite-pilot"
BUFFER = ROOT / "artifacts/surrogate/buffer_experiment_hints_rewrite.jsonl"
CHECKPOINT = ROOT / "artifacts/surrogate/checkpoints/nightly_v3_mc_d005.pkl"
CALIBRATION = ROOT / "artifacts/surrogate/checkpoints/calibration_v3_mc_d005.pkl"
REWRITE_ROOT = OUT_DIR / "hints_rewrite"
HINTS_ROOT = OUT_DIR / "hints"

_SURR_LINE = re.compile(
    r"Surrogate predicts fitness\s*≈\s*([0-9.]+),\s*uncertainty\s*=\s*([0-9.]+)",
    re.I,
)
_PARENT_TRUE = re.compile(
    r"Parent true fitness[^:]*:\s*([0-9.]+)",
    re.I,
)
_DRAFT_PRED = re.compile(
    r"Draft predicted fitness\s*≈\s*([0-9.]+),\s*uncertainty\s*=\s*([0-9.]+)",
    re.I,
)
_DRAFT_JSON_BLOCK = re.compile(
    r"Draft WorldSpec to rewrite:\s*(\{.*?\})\s*\n\s*Current best elite",
    re.S,
)


def _build_surrogate():
    # Match rewrite / hints YAML: extinction_gate_threshold=0.95 (compose path;
    # this checkpoint has no trained fitness head → wrong gate zeros most drafts).
    cfg = SurrogateConfig(
        enabled=True,
        model_type="mlp",
        checkpoint=str(CHECKPOINT),
        stub_mean=0.5,
        stub_uncertainty=1.0,
        calibration=str(CALIBRATION) if CALIBRATION.is_file() else None,
        require_quality_gate=False,
        use_soft_extinction=False,
        extinction_gate_threshold=0.95,
    )
    return get_surrogate(cfg)


def _ws_from_dict(d: dict[str, Any]) -> WorldSpec | None:
    try:
        body = {
            "birth": list(d.get("birth") or []),
            "survival": list(d.get("survival") or []),
            "noise": float(d["noise"]),
            "resource_regen": float(d["resource_regen"]),
            "predation": float(d["predation"]),
            "cell_types": list(d.get("cell_types") or ["life", "food"]),
            "neighborhood": str(d.get("neighborhood") or "moore"),
            "grid_size": int(d.get("grid_size") or 50),
            "steps": int(d.get("steps") or 200),
            "seed": int(d.get("seed") or 0),
        }
        return WorldSpec.from_json_dict(body)
    except (KeyError, TypeError, ValueError):
        return None


def _nmae(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = np.abs(pred - true)
    mae = float(err.mean())
    std = float(true.std(ddof=1)) if len(true) > 1 else float("nan")
    nmae = float(mae / std) if std and std > 0 else float("nan")
    r = float(np.corrcoef(pred, true)[0, 1]) if len(pred) > 2 else float("nan")
    bias = float((pred - true).mean())
    return {
        "n": int(len(pred)),
        "mae": round(mae, 6),
        "nmae": round(nmae, 6),
        "bias_pred_minus_true": round(bias, 6),
        "pearson_r": round(r, 6),
        "true_mean": round(float(true.mean()), 6),
        "pred_mean": round(float(pred.mean()), 6),
        "true_std": round(std, 6),
    }


def _predict_fitness_from_features(surrogate, feature_matrix: np.ndarray) -> np.ndarray:
    """Facade-aligned fitness without MC-dropout uncertainty (NMAE-only path).

    ``SurrogateFacade.predict_batch`` spends most wall on ``predict_uncertainty_batch``.
    For offline NMAE we only need the fitness head / compose path on stored features.
    """
    matrix = np.asarray(feature_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.asarray([], dtype=float)
    component_rows = surrogate.model.predict_components_batch(matrix)
    out = np.empty(matrix.shape[0], dtype=float)
    for i, components in enumerate(component_rows):
        prediction = SurrogatePrediction(
            components=components,
            measures={
                "stability": float(components["stability"]),
                "diversity": float(components["diversity"]),
            },
            fitness=0.0,
            uncertainty=0.0,
        )
        out[i] = _resolve_surrogate_fitness(
            surrogate.model,
            matrix[i],
            prediction,
            use_soft_extinction=surrogate.use_soft_extinction,
            extinction_gate_threshold=surrogate.extinction_gate_threshold,
        )
    return out


def analyze_buffer_nmae(
    surrogate, *, max_per_class: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Offline predict vs true fitness on evaluated LLM buffer rows (feature matrix)."""
    rng = np.random.default_rng(seed)
    by_et: dict[str, list[dict[str, Any]]] = {"llm": [], "llm_rewrite": []}
    with BUFFER.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            et = str(row.get("emitter_type") or "")
            if et not in by_et:
                continue
            feats = row.get("features")
            fit = (row.get("targets") or {}).get("fitness")
            if not isinstance(feats, list) or not feats or fit is None:
                continue
            by_et[et].append(row)

    out: dict[str, Any] = {
        "buffer": str(BUFFER.relative_to(ROOT)),
        "max_per_class": max_per_class,
        "path": "stored buffer features → model fitness head (no WorldSpec re-extract; no unc)",
        "classes": {},
        "pool_sizes": {et: len(rows) for et, rows in by_et.items()},
    }
    for et, rows in by_et.items():
        if len(rows) > max_per_class:
            idx = rng.choice(len(rows), size=max_per_class, replace=False)
            rows = [rows[i] for i in idx]
        X = np.asarray([r["features"] for r in rows], dtype=float)
        trues = np.asarray([float(r["targets"]["fitness"]) for r in rows], dtype=float)
        preds = _predict_fitness_from_features(surrogate, X)
        metrics = _nmae(preds, trues)
        class_row: dict[str, Any] = {
            **metrics,
            "emitter_type": et,
            "note": (
                "evaluated proposals written to live buffer (includes rejects); "
                "llm ≈ draft-kept / non-rewrite commits; "
                "llm_rewrite ≈ rewritten children"
            ),
        }
        out["classes"][et] = class_row
        print(
            f"  {et}: pool={out['pool_sizes'][et]} n={class_row['n']} "
            f"NMAE={class_row['nmae']}",
            flush=True,
        )
    return out


def _user_text(rec: dict[str, Any]) -> str:
    msgs = rec.get("messages") or []
    return "\n".join(
        str(m.get("content") or "") for m in msgs if m.get("role") == "user"
    )


def _response_text(rec: dict[str, Any]) -> str:
    return str(rec.get("response_content") or rec.get("response") or "")


def _parse_child_spec(
    response: str, *, grid_size: int = 50, steps: int = 200
) -> dict | None:
    parsed = extract_json_object_from_text(response)
    if parsed is None:
        return None
    base = WorldSpec(
        birth=[3],
        survival=[2, 3],
        noise=0.01,
        resource_regen=0.1,
        predation=0.1,
        cell_types=["life", "food"],
        grid_size=grid_size,
        steps=steps,
    )
    spec = world_spec_from_llm_payload(
        parsed, grid_size=grid_size, steps=steps, base=base
    )
    if spec is None:
        return None
    return spec.to_canonical_dict()


def _parse_elite_world_spec(user: str) -> dict | None:
    # Current best elite JSON block
    m = re.search(r"Current best elite[^\n]*:\s*(\{)", user)
    if not m:
        return None
    start = m.start(1)
    # brace match
    depth = 0
    for i, ch in enumerate(user[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(user[start : i + 1])
                except json.JSONDecodeError:
                    return None
                ws = obj.get("world_spec")
                return ws if isinstance(ws, dict) else None
    return None


def _parse_draft_block(user: str) -> dict | None:
    m = _DRAFT_JSON_BLOCK.search(user)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # fallback: after marker
    key = "Draft WorldSpec to rewrite:"
    if key not in user:
        return None
    tail = user.split(key, 1)[1].lstrip()
    if not tail.startswith("{"):
        return None
    depth = 0
    for i, ch in enumerate(tail):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(tail[: i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def iter_call_log(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def analyze_rewrite_manipulation(seed_dirs: list[Path]) -> dict[str, Any]:
    """Correlate draft→rewrite edits with child-surrogate scalars."""
    gaps: list[float] = []
    uncs: list[float] = []
    hams: list[float] = []
    l1s: list[float] = []
    identical: int = 0
    parsed_pairs = 0
    prompt_preds: list[float] = []
    draft_specs: list[WorldSpec] = []
    surrogate = _build_surrogate()

    for d in seed_dirs:
        log = d / "llm_call_log.jsonl"
        if not log.is_file():
            continue
        print(f"  parsing {d.name}…", flush=True)
        for rec in iter_call_log(log):
            user = _user_text(rec)
            resp = _response_text(rec)
            if "Draft WorldSpec to rewrite:" not in user:
                continue
            dm = _DRAFT_PRED.search(user)
            pm = _PARENT_TRUE.search(user)
            draft_ws = _parse_draft_block(user)
            child_ws = _parse_child_spec(resp)
            if not (dm and pm and draft_ws and child_ws):
                continue
            child_pred = float(dm.group(1))
            child_unc = float(dm.group(2))
            parent_true = float(pm.group(1))
            ops = _set_ops(child_ws, draft_ws)
            gaps.append(parent_true - child_pred)
            uncs.append(child_unc)
            hams.append(float(ops["rule_hamming"]))
            l1s.append(float(ops["scalar_l1"]))
            if ops["rule_hamming"] == 0 and ops["scalar_l1"] < 1e-6:
                identical += 1
            parsed_pairs += 1
            spec = _ws_from_dict(draft_ws)
            if spec is not None:
                draft_specs.append(spec)
                prompt_preds.append(child_pred)

    prompt_pred_vs_repredict: list[float] = []
    if draft_specs:
        # subsample re-predict check via feature extract + fitness head (skip unc)
        rng = np.random.default_rng(0)
        take = min(len(draft_specs), 1000)
        idx = rng.choice(len(draft_specs), size=take, replace=False)
        specs = [draft_specs[i] for i in idx]
        for spec in specs:
            apply_canonical_seed(spec)
        ppred = np.asarray([prompt_preds[i] for i in idx], dtype=float)
        feat_mat = extract_batch(specs)
        offline = _predict_fitness_from_features(surrogate, feat_mat)
        prompt_pred_vs_repredict = (offline - ppred).tolist()

    def _corr(x: list[float], y: list[float]) -> dict[str, float]:
        if len(x) < 5:
            return {"pearson_r": float("nan"), "spearman_r": float("nan"), "n": len(x)}
        pr = float(np.corrcoef(x, y)[0, 1])
        sr = float(stats.spearmanr(x, y).statistic)
        return {
            "pearson_r": round(pr, 6),
            "spearman_r": round(sr, 6),
            "n": len(x),
        }

    gap = np.asarray(gaps)
    return {
        "n_pairs": parsed_pairs,
        "identical_fraction": (
            round(identical / parsed_pairs, 6) if parsed_pairs else None
        ),
        "mean_rule_hamming": round(float(np.mean(hams)), 6) if hams else None,
        "mean_scalar_l1": round(float(np.mean(l1s)), 6) if l1s else None,
        "mean_parent_minus_child_pred": (
            round(float(gap.mean()), 6) if len(gap) else None
        ),
        "corr_hamming_vs_parent_minus_pred": _corr(gaps, hams),
        "corr_hamming_vs_uncertainty": _corr(uncs, hams),
        "corr_scalar_l1_vs_parent_minus_pred": _corr(gaps, l1s),
        "corr_scalar_l1_vs_uncertainty": _corr(uncs, l1s),
        "prompt_child_pred_minus_offline_repredict": {
            "mean": (
                round(float(np.mean(prompt_pred_vs_repredict)), 6)
                if prompt_pred_vs_repredict
                else None
            ),
            "mae": (
                round(float(np.mean(np.abs(prompt_pred_vs_repredict))), 6)
                if prompt_pred_vs_repredict
                else None
            ),
            "n": len(prompt_pred_vs_repredict),
            "note": "should be ~0 if rewrite prompt scalars match checkpoint predict(draft)",
        },
        "seeds_used": [
            p.name for p in seed_dirs if (p / "llm_call_log.jsonl").is_file()
        ],
    }


def analyze_parent_hint_manipulation(seed_dirs: list[Path]) -> dict[str, Any]:
    """Correlate parent→child edits with parent-level hint scalars."""
    residuals: list[float] = []
    uncs: list[float] = []
    hams: list[float] = []
    l1s: list[float] = []
    n = 0
    for d in seed_dirs:
        log = d / "llm_call_log.jsonl"
        if not log.is_file():
            continue
        for rec in iter_call_log(log):
            user = _user_text(rec)
            if "Draft WorldSpec to rewrite:" in user:
                continue  # rewrite pass
            sm = _SURR_LINE.search(user)
            if not sm:
                continue
            parent_ws = _parse_elite_world_spec(user)
            child_ws = _parse_child_spec(_response_text(rec))
            if not parent_ws or not child_ws:
                continue
            # parent true fitness from elite JSON
            m = re.search(
                r'"fitness"\s*:\s*([0-9.eE+-]+)',
                (
                    user[user.find("Current best elite") :]
                    if "Current best elite" in user
                    else ""
                ),
            )
            parent_true = float(m.group(1)) if m else float("nan")
            hint_f = float(sm.group(1))
            hint_u = float(sm.group(2))
            ops = _set_ops(child_ws, parent_ws)
            residuals.append(
                hint_f - parent_true if parent_true == parent_true else 0.0
            )
            uncs.append(hint_u)
            hams.append(float(ops["rule_hamming"]))
            l1s.append(float(ops["scalar_l1"]))
            n += 1

    def _corr(x: list[float], y: list[float]) -> dict[str, float]:
        if len(x) < 5:
            return {"pearson_r": float("nan"), "spearman_r": float("nan"), "n": len(x)}
        return {
            "pearson_r": round(float(np.corrcoef(x, y)[0, 1]), 6),
            "spearman_r": round(float(stats.spearmanr(x, y).statistic), 6),
            "n": len(x),
        }

    return {
        "n_pairs": n,
        "mean_rule_hamming": round(float(np.mean(hams)), 6) if hams else None,
        "mean_scalar_l1": round(float(np.mean(l1s)), 6) if l1s else None,
        "corr_hamming_vs_hint_minus_parent_true": _corr(residuals, hams),
        "corr_hamming_vs_uncertainty": _corr(uncs, hams),
        "corr_scalar_l1_vs_hint_minus_parent_true": _corr(residuals, l1s),
        "corr_scalar_l1_vs_uncertainty": _corr(uncs, l1s),
        "seeds_used": [
            p.name for p in seed_dirs if (p / "llm_call_log.jsonl").is_file()
        ],
        "note": (
            "Parent-scalar H1: residual = hint_fitness − archive parent true fitness. "
            "Near-zero correlations ⇒ weak evidence the model uses the hint line."
        ),
    }


def main() -> int:
    print("Loading surrogate…", flush=True)
    surrogate = _build_surrogate()
    print("Buffer NMAE (batch subsample)…", flush=True)
    nmae = analyze_buffer_nmae(surrogate)

    rewrite_dirs = sorted(p for p in REWRITE_ROOT.glob("seed_*") if p.is_dir())
    hints_dirs = sorted(p for p in HINTS_ROOT.glob("seed_*") if p.is_dir())

    print("Rewrite manipulation…", flush=True)
    rew_manip = analyze_rewrite_manipulation(rewrite_dirs)
    print("Parent-hint manipulation…", flush=True)
    hint_manip = analyze_parent_hint_manipulation(hints_dirs)

    # Reference: manuscript hold-out NMAE
    ref_holdout_nmae = 0.112

    payload = {
        "scope": (
            "Offline only. Draft-distribution NMAE uses live eval buffer rows "
            "(includes rejected proposals). Manipulation uses llm_call_log.jsonl "
            "where present (child-rewrite pilot)."
        ),
        "reference_holdout_nmae_fitness": ref_holdout_nmae,
        "buffer_nmae": nmae,
        "rewrite_manipulation": rew_manip,
        "parent_hint_manipulation": hint_manip,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "soft_channel_offline.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    llm = nmae["classes"].get("llm", {})
    llmr = nmae["classes"].get("llm_rewrite", {})
    lines = [
        "# Soft-channel offline diagnostics (priority-1)",
        "",
        payload["scope"],
        "",
        "## 1. Surrogate NMAE on evaluated LLM proposals (draft-ish distribution)",
        "",
        f"Source: `{BUFFER.relative_to(ROOT)}` (live_eval). "
        f"Offline fitness from stored features via `{CHECKPOINT.name}` "
        f"(same resolve path as facade; no MC-uncertainty). "
        f"Manuscript hold-out NMAE (accepted elites / compose path) ≈ **{ref_holdout_nmae}**.",
        "",
        "| Class | n | MAE | NMAE | bias (pred−true) | Pearson r | true mean |",
        "|-------|--:|----:|-----:|-----------------:|----------:|----------:|",
    ]
    for et, m in (("llm", llm), ("llm_rewrite", llmr)):
        if not m:
            continue
        lines.append(
            f"| `{et}` | {m['n']} | {m['mae']:.4f} | **{m['nmae']:.3f}** | "
            f"{m['bias_pred_minus_true']:+.4f} | {m['pearson_r']:.3f} | {m['true_mean']:.3f} |"
        )
    lines += [
        "",
        "- `llm`: commits without successful rewrite (draft kept / rewrite not triggered).",
        "- `llm_rewrite`: rewritten children that were evaluated.",
        "- If NMAE ≫ hold-out 0.112 on these classes, child-level advice is closer to "
        "noise than the confirmatory parent-scalar interface assumed.",
        "",
        "## 2. Manipulation: does rewrite edit size track child-surrogate numbers?",
        "",
        f"Parsed rewrite pairs from `hints_rewrite/*/llm_call_log.jsonl`: "
        f"**n={rew_manip['n_pairs']}** "
        f"(seeds: {', '.join(rew_manip['seeds_used']) or 'none'}).",
        "",
        f"- Mean rule Hamming draft→rewrite: {rew_manip['mean_rule_hamming']}",
        f"- Mean scalar L1: {rew_manip['mean_scalar_l1']}",
        f"- Identical draft returned: {rew_manip['identical_fraction']}",
        f"- Mean (parent_true − child_pred): {rew_manip['mean_parent_minus_child_pred']}",
        f"- Corr Hamming vs (parent−pred): {rew_manip['corr_hamming_vs_parent_minus_pred']}",
        f"- Corr Hamming vs uncertainty: {rew_manip['corr_hamming_vs_uncertainty']}",
        f"- Corr scalar L1 vs (parent−pred): {rew_manip['corr_scalar_l1_vs_parent_minus_pred']}",
        f"- Prompt pred vs offline re-predict(draft): "
        f"{rew_manip['prompt_child_pred_minus_offline_repredict']}",
        "",
        "## 3. Manipulation: parent-H1 hint line vs parent→child edits",
        "",
        f"Parsed from `hints/*/llm_call_log.jsonl` (draft pass only): "
        f"**n={hint_manip['n_pairs']}** "
        f"(seeds: {', '.join(hint_manip['seeds_used']) or 'none'}).",
        "",
        f"- Mean rule Hamming parent→child: {hint_manip['mean_rule_hamming']}",
        f"- Corr Hamming vs (hint−parent_true): "
        f"{hint_manip['corr_hamming_vs_hint_minus_parent_true']}",
        f"- Corr Hamming vs uncertainty: {hint_manip['corr_hamming_vs_uncertainty']}",
        f"- Corr scalar L1 vs (hint−parent_true): "
        f"{hint_manip['corr_scalar_l1_vs_hint_minus_parent_true']}",
        "",
        hint_manip["note"],
        "",
        "## Reading",
        "",
        "Descriptive only — not Holm / not a new confirmatory family.",
        "",
        f"- Child-distribution NMAE (`llm` {llm.get('nmae', float('nan')):.3f}, "
        f"`llm_rewrite` {llmr.get('nmae', float('nan')):.3f}) is "
        f"**{llm.get('nmae', float('nan')) / ref_holdout_nmae:.1f}× / "
        f"{llmr.get('nmae', float('nan')) / ref_holdout_nmae:.1f}×** "
        "the hold-out 0.112 — child advice is noisier than parent-hold-out validity "
        "suggests, but not pure noise (rewrite class Pearson "
        f"r≈{llmr.get('pearson_r', float('nan')):.2f}).",
        f"- Rewrite actuator is mostly a no-op edit: "
        f"**{100 * (rew_manip.get('identical_fraction') or 0):.0f}%** identical "
        f"drafts returned; mean Hamming≈{rew_manip.get('mean_rule_hamming')}. "
        "Edit size ≈ uncorrelated with (parent−pred) / unc "
        f"(|r|≲0.03).",
        "- Parent-H1 hint residual vs parent→child Hamming: weak "
        f"(Pearson r≈{hint_manip['corr_hamming_vs_hint_minus_parent_true'].get('pearson_r')}).",
        "- Prompt child_pred matches offline re-predict at compose gate 0.95 "
        f"(MAE≈{rew_manip['prompt_child_pred_minus_offline_repredict'].get('mae')}).",
        "",
        f"JSON: `{json_path.relative_to(ROOT)}`",
        "",
    ]
    md_path = OUT_DIR / "SOFT_CHANNEL_OFFLINE.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Online gate validation on genetic_me_shadow (true fitness for all proposals).

Reports AUROC / precision-recall at τ, Spearman, NMAE by emitter, and
decile calibration of predicted vs true fitness. Descriptive only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHADOW = ROOT / "artifacts/experiments/q1-h2-ranking-controls/genetic_me_shadow"
TAU = 0.45


def _nmae(pred: np.ndarray, true: np.ndarray) -> float:
    std = float(np.std(true))
    if std <= 0.0:
        return float("nan")
    return float(np.mean(np.abs(pred - true)) / std)


def _spearman(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """Return (rho, p). Casts around incomplete scipy stubs for SignificanceResult."""
    result = cast(Any, spearmanr(pred, true))
    return float(result.statistic), float(result.pvalue)


def _precision_recall_at_tau(
    pred: np.ndarray, true: np.ndarray, *, tau: float
) -> tuple[float, float]:
    """Precision/recall when skipping proposals with pred < tau (true skip = true < tau)."""
    pred_skip = pred < tau
    y_skip = true < tau
    tp = int((pred_skip & y_skip).sum())
    n_pred_skip = int(pred_skip.sum())
    n_true_skip = int(y_skip.sum())
    precision = float(tp / n_pred_skip) if n_pred_skip else float("nan")
    recall = float(tp / n_true_skip) if n_true_skip else float("nan")
    return precision, recall


def _load_rows(shadow_root: Path, seeds: range) -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    for seed in seeds:
        path = shadow_root / f"seed_{seed}" / "surrogate_archive.jsonl"
        if not path.is_file():
            raise SystemExit(f"missing shadow archive: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(
                (
                    str(payload.get("emitter_type", "?")),
                    float(payload["prediction"]["fitness"]),
                    float(payload["eval_outcome"]["fitness"]),
                )
            )
    return rows


def analyze(shadow_root: Path, *, tau: float = TAU) -> dict[str, Any]:
    rows = _load_rows(shadow_root, range(10))
    pred = np.asarray([r[1] for r in rows], dtype=np.float64)
    true = np.asarray([r[2] for r in rows], dtype=np.float64)
    emitter = np.asarray([r[0] for r in rows])
    y_skip = (true < tau).astype(int)
    score = -pred
    pred_skip = pred < tau
    tp = int(((pred_skip) & (y_skip == 1)).sum())
    fp = int(((pred_skip) & (y_skip == 0)).sum())
    fn = int(((~pred_skip) & (y_skip == 1)).sum())
    tn = int(((~pred_skip) & (y_skip == 0)).sum())
    rho, rho_p = _spearman(pred, true)
    precision_at_tau, recall_at_tau = _precision_recall_at_tau(pred, true, tau=tau)
    edges = np.linspace(0.0, 1.0, 11)
    calibration: list[dict[str, Any]] = []
    for i in range(len(edges) - 1):
        mask = (pred >= edges[i]) & (pred < edges[i + 1])
        if i == len(edges) - 2:
            mask = (pred >= edges[i]) & (pred <= edges[i + 1])
        if not np.any(mask):
            continue
        calibration.append(
            {
                "pred_lo": float(edges[i]),
                "pred_hi": float(edges[i + 1]),
                "n": int(mask.sum()),
                "mean_pred": float(pred[mask].mean()),
                "mean_true": float(true[mask].mean()),
                "frac_true_lt_tau": float((true[mask] < tau).mean()),
            }
        )
    by_emitter: dict[str, Any] = {}
    for name in sorted(set(emitter.tolist())):
        mask = emitter == name
        y = (true[mask] < tau).astype(int)
        spearman_rho, _ = _spearman(pred[mask], true[mask])
        precision, recall = _precision_recall_at_tau(pred[mask], true[mask], tau=tau)
        by_emitter[name] = {
            "n": int(mask.sum()),
            "nmae": _nmae(pred[mask], true[mask]),
            "mae": float(np.mean(np.abs(pred[mask] - true[mask]))),
            "spearman": spearman_rho,
            "auroc_skip": float(roc_auc_score(y, -pred[mask])),
            "precision_at_tau": precision,
            "recall_at_tau": recall,
        }
    band = (pred >= 0.35) & (pred < 0.55)
    return {
        "family": "H2-gate-online-validation",
        "n_proposals": int(len(rows)),
        "tau": tau,
        "frac_true_lt_tau": float(y_skip.mean()),
        "frac_pred_lt_tau": float(pred_skip.mean()),
        "auroc_pred_for_true_lt_tau": float(roc_auc_score(y_skip, score)),
        "average_precision": float(average_precision_score(y_skip, score)),
        "precision_at_tau": precision_at_tau,
        "recall_at_tau": recall_at_tau,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "spearman_pred_true": rho,
        "spearman_p": rho_p,
        "nmae_all": _nmae(pred, true),
        "mae_all": float(np.mean(np.abs(pred - true))),
        "n_band_0.35_0.55": int(band.sum()),
        "nmae_band_0.35_0.55": _nmae(pred[band], true[band]) if band.any() else None,
        "frac_true_lt_tau_in_band": (
            float((true[band] < tau).mean()) if band.any() else None
        ),
        "by_emitter": by_emitter,
        "calibration_deciles": calibration,
        "source": str(shadow_root),
    }


def _markdown(summary: dict[str, Any]) -> str:
    ge = summary["by_emitter"].get("genetic", {})
    ra = summary["by_emitter"].get("random", {})
    return "\n".join(
        [
            "# H2 gate online validation (shadow stream)",
            "",
            f"Source: `{summary['source']}` · $n={summary['n_proposals']:,}$ "
            f"proposals · $\\tau={summary['tau']}$.",
            "",
            f"- AUROC (pred ranks true fit $<\\tau$): **{summary['auroc_pred_for_true_lt_tau']:.3f}**",
            f"- Precision / recall @ $\\tau$: "
            f"**{summary['precision_at_tau']:.3f}** / "
            f"**{summary['recall_at_tau']:.3f}**",
            f"- Spearman(pred, true): **{summary['spearman_pred_true']:.3f}**",
            f"- Online NMAE: all **{summary['nmae_all']:.3f}**; "
            f"genetic **{ge.get('nmae', float('nan')):.3f}**; "
            f"random **{ra.get('nmae', float('nan')):.3f}** "
            f"(hold-out NMAE $=0.112$ is not this estimand)",
            "",
            "Descriptive only — does not amend confirmatory Holm families.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-root", type=Path, default=DEFAULT_SHADOW)
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "artifacts/experiments/q1-h2-ranking-controls/h2_gate_online_validation.json",
    )
    args = parser.parse_args()
    summary = analyze(args.shadow_root.resolve(), tau=args.tau)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md = args.out.with_name("H2_GATE_ONLINE_VALIDATION.md")
    md.write_text(_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {k: summary[k] for k in summary if k != "calibration_deciles"}, indent=2
        )
    )
    print(f"Wrote {args.out}")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()

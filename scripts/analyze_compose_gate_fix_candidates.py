#!/usr/bin/env python3
"""Offline compose-gate fix-candidate replay on q1-full/filter logs.

No new simulator runs. Uses logged ``surrogate_archive.jsonl`` predictions to:

1. Localize D1 to the gray zone ``p_ext ∈ [0.5, 0.95)``.
2. Replay confidence-gated skip (force-eval gray zone / skip only if
   ``p_ext ≥ 0.95``).
3. Replay soft extinction recompose with ``min_fit`` retargeted to match
   production skip (~33.5%).
4. Reference: align both compose gates to 0.95 (D1 → 0 by construction).

These are diagnostic / policy candidates for journal-extension compose repair,
not confirmatory H3 unlocks.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_compose_gate_live import (
    DEFAULT_FILTER_ROOT,
    GATE_A,
    GATE_B,
    MAX_UNC,
    discover_seed_archives,
    would_skip,
    _iter_jsonl,
    _prediction,
)
from worldspace.surrogate.utils import compute_fitness_from_prediction

FILTER_ROOT = DEFAULT_FILTER_ROOT
OUT_JSON = ROOT / "artifacts/surrogate/compose_gate_fix_candidates.json"
OUT_MD = ROOT / "artifacts/experiments/q1-full/COMPOSE_GATE_FIX_CANDIDATES.md"

PRODUCTION_TAU = 0.45
TARGET_SKIP = 0.335  # production hard@0.95 skip rate
SOFT_TAU_GRID: tuple[float, ...] = tuple(
    round(x * 0.01, 2) for x in range(5, 96)
)  # 0.05 .. 0.95


@dataclass(frozen=True)
class ProposalView:
    """One logged proposal with recomposed fitness variants."""

    p_ext: float
    uncertainty: float
    force_empty: bool
    fit_hard_a: float
    fit_hard_b: float
    fit_soft: float
    logged_skip: bool

    @property
    def in_gray_zone(self) -> bool:
        return GATE_A <= self.p_ext < GATE_B


def _iter_proposals(path: Path) -> Iterator[ProposalView]:
    for record in _iter_jsonl(path):
        pred = _prediction(record)
        yield ProposalView(
            p_ext=float(pred.components["early_extinction_prob"]),
            uncertainty=float(pred.uncertainty),
            force_empty=record.get("decision_reason") == "empty_bin_explore",
            fit_hard_a=float(
                compute_fitness_from_prediction(pred, extinction_gate_threshold=GATE_A)
            ),
            fit_hard_b=float(
                compute_fitness_from_prediction(pred, extinction_gate_threshold=GATE_B)
            ),
            fit_soft=float(
                compute_fitness_from_prediction(pred, use_soft_extinction=True)
            ),
            logged_skip=record.get("decision") == "skip",
        )


def _skip(fitness: float, view: ProposalView, *, tau: float) -> bool:
    return would_skip(
        fitness,
        view.uncertainty,
        min_predicted_fitness=tau,
        max_uncertainty_to_skip=MAX_UNC,
        force_eval_empty=view.force_empty,
    )


def _d1_and_skip(
    views: list[ProposalView],
    *,
    skip_a_fn,
    skip_b_fn,
) -> dict[str, Any]:
    n = len(views)
    if n == 0:
        return {"n": 0, "d1": None, "skip_a": None, "skip_b": None}
    div = skip_a = skip_b = 0
    for v in views:
        a = bool(skip_a_fn(v))
        b = bool(skip_b_fn(v))
        div += int(a != b)
        skip_a += int(a)
        skip_b += int(b)
    return {
        "n": n,
        "d1": round(div / n, 4),
        "skip_rate_gate_0.5": round(skip_a / n, 4),
        "skip_rate_gate_0.95": round(skip_b / n, 4),
        "skip_rate_pct_gate_0.95": round(100.0 * skip_b / n, 2),
    }


def localize_gray_zone(views: list[ProposalView], *, tau: float) -> dict[str, Any]:
    """D1 on all / in-band / out-of-band at production τ."""

    def sa(v: ProposalView) -> bool:
        return _skip(v.fit_hard_a, v, tau=tau)

    def sb(v: ProposalView) -> bool:
        return _skip(v.fit_hard_b, v, tau=tau)

    in_band = [v for v in views if v.in_gray_zone]
    out_band = [v for v in views if not v.in_gray_zone]
    return {
        "tau": tau,
        "all": _d1_and_skip(views, skip_a_fn=sa, skip_b_fn=sb),
        "gray_zone_p_ext_in_[0.5,0.95)": {
            **_d1_and_skip(in_band, skip_a_fn=sa, skip_b_fn=sb),
            "share_of_proposals": round(len(in_band) / len(views), 4) if views else 0.0,
        },
        "outside_gray_zone": {
            **_d1_and_skip(out_band, skip_a_fn=sa, skip_b_fn=sb),
            "share_of_proposals": (
                round(len(out_band) / len(views), 4) if views else 0.0
            ),
        },
    }


def confidence_gated_policies(
    views: list[ProposalView], *, tau: float
) -> dict[str, Any]:
    """Force-eval gray zone, or allow skip only when p_ext ≥ 0.95."""

    def prod_a(v: ProposalView) -> bool:
        return _skip(v.fit_hard_a, v, tau=tau)

    def prod_b(v: ProposalView) -> bool:
        return _skip(v.fit_hard_b, v, tau=tau)

    def force_eval_gray_a(v: ProposalView) -> bool:
        return False if v.in_gray_zone else prod_a(v)

    def force_eval_gray_b(v: ProposalView) -> bool:
        return False if v.in_gray_zone else prod_b(v)

    def confident_extinct_a(v: ProposalView) -> bool:
        return prod_a(v) if v.p_ext >= GATE_B else False

    def confident_extinct_b(v: ProposalView) -> bool:
        return prod_b(v) if v.p_ext >= GATE_B else False

    # Among production hard@0.95 skips, how many are gray-zone?
    prod_skips = [v for v in views if prod_b(v)]
    gz_of_skips = sum(1 for v in prod_skips if v.in_gray_zone)

    return {
        "tau": tau,
        "production_hard": _d1_and_skip(views, skip_a_fn=prod_a, skip_b_fn=prod_b),
        "force_eval_gray_zone": _d1_and_skip(
            views, skip_a_fn=force_eval_gray_a, skip_b_fn=force_eval_gray_b
        ),
        "skip_only_if_p_ext_ge_0.95": _d1_and_skip(
            views, skip_a_fn=confident_extinct_a, skip_b_fn=confident_extinct_b
        ),
        "production_skips_from_gray_zone": {
            "n_production_skips_hard_0.95": len(prod_skips),
            "n_from_gray_zone": gz_of_skips,
            "frac_from_gray_zone": (
                round(gz_of_skips / len(prod_skips), 4) if prod_skips else None
            ),
        },
        "note": (
            "Both confidence policies drive D1→0 by refusing to skip ambiguous "
            "p_ext; skip rate falls sharply vs production (~33.5%→~12%)."
        ),
    }


def _soft_skip_rate(views: list[ProposalView], *, tau: float) -> float:
    if not views:
        return 0.0
    skips = sum(1 for v in views if _skip(v.fit_soft, v, tau=tau))
    return skips / len(views)


def retarget_soft_tau(
    views: list[ProposalView],
    *,
    target_skip: float = TARGET_SKIP,
    grid: tuple[float, ...] = SOFT_TAU_GRID,
) -> dict[str, Any]:
    """Find soft-compose τ closest to production skip rate."""
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for tau in grid:
        rate = _soft_skip_rate(views, tau=tau)
        row = {
            "tau": tau,
            "skip_rate": round(rate, 4),
            "skip_rate_pct": round(100.0 * rate, 2),
            "abs_err_vs_target": round(abs(rate - target_skip), 4),
        }
        rows.append(row)
        if best is None or row["abs_err_vs_target"] < best["abs_err_vs_target"]:
            best = row

    assert best is not None
    tau_star = float(best["tau"])

    def soft_skip(v: ProposalView) -> bool:
        return _skip(v.fit_soft, v, tau=tau_star)

    def hard95(v: ProposalView) -> bool:
        return _skip(v.fit_hard_b, v, tau=PRODUCTION_TAU)

    # Soft ignores extinction gate → D1 soft@0.5 vs soft@0.95 is trivially 0.
    n = len(views)
    soft_vs_hard = sum(1 for v in views if soft_skip(v) != hard95(v)) / n if n else 0.0

    return {
        "target_skip_rate": target_skip,
        "production_tau_hard": PRODUCTION_TAU,
        "soft_tau_at_production_0.45": {
            "tau": PRODUCTION_TAU,
            "skip_rate": round(_soft_skip_rate(views, tau=PRODUCTION_TAU), 4),
            "skip_rate_pct": round(
                100.0 * _soft_skip_rate(views, tau=PRODUCTION_TAU), 2
            ),
        },
        "retargeted": best,
        "soft_vs_hard0.95_skip_divergence_at_retarget": round(soft_vs_hard, 4),
        "d1_soft_gate0.5_vs_0.95": 0.0,
        "d1_soft_note": (
            "Soft extinction ignores the extinction gate, so D1 between gates "
            "is 0 by construction — not evidence that soft repairs D1."
        ),
        "grid_extrema": {
            "min_skip_pct": min(r["skip_rate_pct"] for r in rows),
            "max_skip_pct": max(r["skip_rate_pct"] for r in rows),
        },
    }


def align_gates_reference(views: list[ProposalView], *, tau: float) -> dict[str, Any]:
    """Both gates at 0.95 → D1 = 0; hold-out alignment candidate."""

    def sb(v: ProposalView) -> bool:
        return _skip(v.fit_hard_b, v, tau=tau)

    return {
        "tau": tau,
        "both_gates_0.95": _d1_and_skip(views, skip_a_fn=sb, skip_b_fn=sb),
        "note": (
            "Aligning offline hold-out compose gate to live 0.95 zeroes D1 by "
            "construction; requires hold-out re-evaluation / protocol amendment, "
            "not necessarily a new QD run."
        ),
    }


def load_all_views(filter_root: Path = FILTER_ROOT) -> list[ProposalView]:
    archives = discover_seed_archives(filter_root)
    views: list[ProposalView] = []
    for path in archives:
        views.extend(_iter_proposals(path))
    return views


def run_analysis(filter_root: Path = FILTER_ROOT) -> dict[str, Any]:
    views = load_all_views(filter_root)
    return {
        "source": "offline_compose_gate_fix_candidates",
        "filter_root": str(filter_root.relative_to(ROOT)),
        "n_proposals": len(views),
        "n_seeds": len(discover_seed_archives(filter_root)),
        "production_tau": PRODUCTION_TAU,
        "gates": [GATE_A, GATE_B],
        "gray_zone_localization": localize_gray_zone(views, tau=PRODUCTION_TAU),
        "confidence_gated": confidence_gated_policies(views, tau=PRODUCTION_TAU),
        "soft_recompose": retarget_soft_tau(views),
        "align_gates_reference": align_gates_reference(views, tau=PRODUCTION_TAU),
        "readout": (
            "Gray zone accounts for all D1 mass (out-of-band D1=0). "
            "Confidence-gated skip drives D1→0 but cuts skip ~33%→~12%. "
            "Soft@τ=0.45 over-skips (~96%); retarget soft τ to match production "
            "skip before comparing policies. Aligning gates to 0.95 is the "
            "direct compose repair for confirmatory H3."
        ),
    }


def _md(payload: dict[str, Any]) -> str:
    gz = payload["gray_zone_localization"]
    cg = payload["confidence_gated"]
    soft = payload["soft_recompose"]
    lines = [
        "# Compose-gate fix candidates (offline replay)",
        "",
        f"Artifact: `{OUT_JSON.relative_to(ROOT)}`",
        f"Source: `{payload['filter_root']}` ({payload['n_proposals']} proposals, "
        f"{payload['n_seeds']} seeds)",
        "",
        "## 1. Gray-zone localization (τ=0.45)",
        "",
        f"- All proposals: D1={gz['all']['d1']}",
        f"- Gray zone p_ext in [0.5, 0.95): "
        f"D1={gz['gray_zone_p_ext_in_[0.5,0.95)']['d1']} "
        f"(share {100*gz['gray_zone_p_ext_in_[0.5,0.95)']['share_of_proposals']:.1f}%)",
        f"- Outside gray zone: D1={gz['outside_gray_zone']['d1']} "
        f"(share {100*gz['outside_gray_zone']['share_of_proposals']:.1f}%)",
        "",
        "## 2. Confidence-gated skip policies",
        "",
        "| Policy | D1 | Skip @ gate 0.95 |",
        "|--------|----|------------------|",
        f"| Production hard | {cg['production_hard']['d1']} | "
        f"{cg['production_hard']['skip_rate_pct_gate_0.95']}% |",
        f"| Force-eval gray zone | {cg['force_eval_gray_zone']['d1']} | "
        f"{cg['force_eval_gray_zone']['skip_rate_pct_gate_0.95']}% |",
        f"| Skip only if p_ext >= 0.95 | "
        f"{cg['skip_only_if_p_ext_ge_0.95']['d1']} | "
        f"{cg['skip_only_if_p_ext_ge_0.95']['skip_rate_pct_gate_0.95']}% |",
        "",
        f"Of production skips, "
        f"{100*(cg['production_skips_from_gray_zone']['frac_from_gray_zone'] or 0):.1f}% "
        "are gray-zone proposals.",
        "",
        "## 3. Soft recompose",
        "",
        f"- Soft @ τ=0.45 skip: "
        f"{soft['soft_tau_at_production_0.45']['skip_rate_pct']}% "
        "(over-skips vs production 33.5%)",
        f"- Retargeted soft τ={soft['retargeted']['tau']}: "
        f"skip {soft['retargeted']['skip_rate_pct']}% "
        f"(target {100*soft['target_skip_rate']:.1f}%)",
        f"- Soft vs hard@0.95 skip divergence at retarget: "
        f"{soft['soft_vs_hard0.95_skip_divergence_at_retarget']}",
        f"- D1 soft gate 0.5 vs 0.95: {soft['d1_soft_gate0.5_vs_0.95']} "
        "(trivial — soft ignores gate)",
        "",
        "## 4. Align gates reference",
        "",
        f"- Both gates 0.95: D1="
        f"{payload['align_gates_reference']['both_gates_0.95']['d1']}",
        "",
        "## Readout",
        "",
        payload["readout"],
        "",
        "Run: `uv run python scripts/analyze_compose_gate_fix_candidates.py`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    payload = run_analysis()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(_md(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "readout": payload["readout"],
                **{
                    k: payload[k]
                    for k in (
                        "gray_zone_localization",
                        "confidence_gated",
                        "soft_recompose",
                        "align_gates_reference",
                    )
                },
            },
            indent=2,
        )[:4000]
    )
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

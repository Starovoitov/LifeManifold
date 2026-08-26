# Sphere Phase A — genetic `min_fitness_frontier` vs uniform

**Tier:** `q1-sphere-genetic-policy`
**Protocol:** [`Q1_RQ1_SPHERE_DOMAIN.md`](../../../Q1_RQ1_SPHERE_DOMAIN.md)
**Status:** seeds **0–9 DONE** (2026-08-26). **GO** for Phase B.
**Comparator:** paired `genetic` (uniform) in this tier (not Sphere H2 `me_uniform`).

Does **not** amend Holm, CA H1, maze empty 2×2, or Sphere H2. Not Holm, not TOST.

## Decision (locked cut: 5{,}000 proposals, empty $100\times100$)

| Arm | Coverage @ 5k % |
|-----|-----------------|
| `genetic` (uniform) | 19.38 ± 0.31 |
| **`genetic_minfit`** | **13.59 ± 0.46** |
| Δ minfit − uniform | **−5.79 ± 0.55 pp** (10/10) |

Protocol GO: $\lvert\mathrm{mean}\,\Delta\rvert \ge 5$ pp **and** 10/10 same sign.

Policy moves Sphere coverage in the headroom regime, so a stub/hints 2×2 can
test the H1 leftover *on this evaluator*. The gap is an order of magnitude
smaller than maze empty genetic (−40.19 pp @ 5k). Do not port pp.

## Per-seed Δ coverage @ 5k (pp)

| Seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|---|
| Uniform % | 19.21 | 19.61 | 19.56 | 19.34 | 18.89 | 19.05 | 19.93 | 19.67 | 19.06 | 19.50 |
| Minfit % | 13.96 | 12.77 | 13.84 | 13.49 | 13.98 | 13.52 | 13.58 | 13.89 | 12.73 | 14.11 |

Reproduce: `python scripts/analyze_sphere_genetic_policy.py`

## Phase B

Authorized by this GO. **Read** 2026-08-27 (**40/40**). Policy
$+2.68$/$+2.47$ pp; leftover $+0.00$/$-0.21$ pp. See
[`../q1-rq1-sphere-factorial/ANALYSIS.md`](../q1-rq1-sphere-factorial/ANALYSIS.md).

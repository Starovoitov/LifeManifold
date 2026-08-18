# Maze RQ1 Phase B — LLM 2×2 (empty archive, 5k)

**Tier:** `q1-rq1-maze-factorial`
**Protocol:** [`Q1_RQ1_SECOND_DOMAIN.md`](../../../Q1_RQ1_SECOND_DOMAIN.md)
**Status:** seeds **0–9 DONE** (2026-08-18). **40/40.**
**Model:** dated `gpt-4o-mini-2024-07-18`; call logs on all 40 runs.

Does **not** amend Holm H5, locked maze arms at 32.5k, CA RQ1, or genetic H2.
Not Holm, not TOST. Does not identify CA Holm $+15.09$. Empty archive is
not occupancy-matched to the CA $971$-niche / $38.84\%$ floor.

## Primary cut (terminal coverage @ 5{,}000 proposals)

| Cell | Prompt | Policy | Coverage % | Filled / 900 | QD-score |
|------|--------|--------|------------|--------------|----------|
| `llm_stub_minfit` | stub | minfit | **5.10 ± 0.83** | 45.9 ± 7.5 | 21.2 ± 2.7 |
| `llm_stub_uniform` | stub | uniform | **44.20 ± 3.52** | 397.8 ± 31.7 | 226.4 ± 23.1 |
| `llm_hints_minfit` | live | minfit | **5.12 ± 0.81** | 46.1 ± 7.3 | 21.3 ± 2.7 |
| `llm_hints_uniform` | live | uniform | **44.43 ± 3.68** | 399.9 ± 33.1 | 230.4 ± 25.3 |

All 40: 5000 evaluations, 0 skips, `prompt_version=c5def71acc84f0db`.
LLM emit attempts $=2940$/seed; parse success $1.0$; fallbacks $\approx 0$.

## Paired contrasts (n=10, seeds 0–9)

| Contrast | Mean Δ ± SD | Sign $+$/−/$0$ |
|----------|-------------|----------------|
| policy @ stub (`uniform` − `minfit`) | **+39.10 ± 3.38 pp** | **10/0/0** |
| policy @ live | **+39.31 ± 3.97 pp** | **10/0/0** |
| leftover @ minfit (live − stub) | **+0.02 ± 0.14 pp** | 5/2/3 |
| leftover @ uniform | **+0.23 ± 3.42 pp** | 5/4/1 |
| interaction (leftover@uni − leftover@minfit) | $+0.21 \pm 3.35$ pp | 5/5/0 |

Off-diagonal bundle `hints_uniform` − `stub_minfit` $= +39.33$ pp and equals
$\Delta_{\mathrm{policy}\mid\mathrm{stub}}+\Delta_{\mathrm{soft}\mid\mathrm{uniform}}$
seed-wise (cell-mean identity).

Archives are **not** bit-identical at minfit (0/10), unlike the CA
calendar-blocked factorial. Coverage still matches to $0.02$ pp.

## Reading

Same qualitative inversion as CA: **policy dominates, parent-level scalars
do not.** Matching `target_selection` leaves a leftover consistent with
seed noise (split signs at uniform; minfit pinned at ${\sim}5\%$).
Matching prompt and swapping policy moves ${\approx}{+}39$ pp on either
soft level (10/10).

Magnitudes are **not** portable. Genetic Phase A @ 5k was
**−40.19 ± 4.13 pp** (minfit $16.79\%$ vs frozen uniform $56.98\%$).
The LLM policy gap is the same order (${\approx}39$ pp), not the CA
${\approx}15$ pp continuation gap. LLM sits **below** genetic at both
policies at this cut (uniform $44.2$ vs $57.0$; minfit $5.1$ vs $16.8$) —
this is not an LLM-vs-genetic ranking and not a portable pp claim.

Leftover @ uniform SD ($3.42$ pp) is an order of magnitude larger than
the CA leftover. Do not read $+0.23$ as a tight scalar-null identification;
read it as **no second ${\sim}39$ pp channel**.

H5 `llm_stub`/`llm_hints` @ 32.5k remain the wrong cells for this story:
uniform saturates the 900-cell maze archive, so the locked hints−stub
AUC ($+1.5$ pp) had no headroom to show a policy collapse.

## Operational

Cat-killed workers 2026-08-18; relaunch 10:26 wiped incomplete
`llm_stub_minfit/seed_6` and `llm_hints_uniform/seed_5` and re-ran from
empty. Worker-1 HTTP 520 on 2026-08-17 was retried after generator
$520$–$524$ handling; that seed was wiped, not resumed.

Reproduce: `python scripts/analyze_rq1_maze_factorial.py`
